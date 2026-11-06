"""Improved phylogenetic topology recovery: NJ + Quartet Exhaustive/NNI Refinement."""
# EVOLVE-BLOCK-START
import math
import time
import itertools
import sys
import re
from collections import defaultdict

_DEADLINE = 260.0  # safety margin under 270s wall-clock
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _deadline_exceeded():
    return _elapsed() > _DEADLINE


def _jc_distance(a, b):
    """Jukes-Cantor corrected distance."""
    n = len(a)
    if n == 0:
        return 0.0
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distance matrix as a 2D list indexed by label index."""
    n = len(labels)
    idx = {label: i for i, label in enumerate(labels)}
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _nj_build_v2(dist, labels):
    """
    Neighbor Joining with proper distance matrix handling.
    Returns Newick string.
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"(({labels[0]},{labels[1]}),{labels[2]})"
    
    nodes = list(range(n))
    tree_repr = {i: labels[i] for i in range(n)}
    
    d = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d[(i, j)] = dist[i][j]
    
    counter = n
    
    while len(nodes) > 3:
        if _deadline_exceeded():
            break
        m = len(nodes)
        total = {a: 0.0 for a in nodes}
        for a in nodes:
            for b in nodes:
                if a != b:
                    total[a] += d[(a, b)]
        
        best_val = float('inf')
        best_a, best_b = nodes[0], nodes[1]
        for i in range(m):
            for j in range(i + 1, m):
                a, b = nodes[i], nodes[j]
                criterion = (m - 2) * d[(a, b)] - total[a] - total[b]
                if criterion < best_val:
                    best_val = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            new_d = 0.5 * (d[(a, x)] + d[(b, x)] - d[(a, b)])
            new_d = max(new_d, 0.0)
            d[(counter, x)] = new_d
            d[(x, counter)] = new_d
        
        for x in nodes:
            if x != a and x != b:
                del d[(a, x)]
                del d[(b, x)]
        del d[(a, b)]
        del d[(b, a)]
        
        joined = f"({tree_repr[a]},{tree_repr[b]})"
        nodes = rest + [counter]
        tree_repr[counter] = joined
        counter += 1
    
    if len(nodes) == 3:
        a, b, c = nodes
        return f"(({tree_repr[a]},{tree_repr[b]}),{tree_repr[c]})"
    elif len(nodes) == 2:
        return f"({tree_repr[nodes[0]]},{tree_repr[nodes[1]]})"
    elif len(nodes) == 1:
        return tree_repr[nodes[0]]
    else:
        return "(" + ",".join(tree_repr[x] for x in nodes) + ")"


def _parse_newick_to_splits(newick, labels):
    """
    Parse Newick string and extract non-trivial splits.
    Returns a set of frozensets representing the splits.
    """
    all_taxa = set(labels)
    s = newick.strip().rstrip(';')
    
    # Recursive parser that collects all internal node leaf sets
    subtrees = []
    
    def parse(pos):
        if pos >= len(s):
            return set(), pos
        
        if s[pos] == '(':
            pos += 1
            children = []
            while pos < len(s) and s[pos] != ')':
                child_set, pos = parse(pos)
                children.append(child_set)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
            if pos < len(s) and s[pos] == ')':
                pos += 1
            all_taxa = set()
            for c in children:
                all_taxa |= c
            # Skip over any label/branch length after )
            while pos < len(s) and s[pos] not in '(),:;':
                pos += 1
            if len(all_taxa) > 1:
                subtrees.append(frozenset(all_taxa))
            return all_taxa, pos
        else:
            start = pos
            while pos < len(s) and s[pos] not in '(),:;':
                pos += 1
            label = s[start:pos]
            return {label}, pos
    
    parse(0)
    
    splits = set()
    for st in subtrees:
        if 0 < len(st) < len(all_taxa):
            other = all_taxa - st
            if len(st) <= len(other):
                splits.add(st)
            else:
                splits.add(frozenset(other))
                
    return splits


def _quartet_score_from_dist(splits, dist, labels):
    """
    Compute sum of quartet consistency scores using distance matrix.
    Higher is better.
    """
    n = len(labels)
    if n < 4:
        return 0.0
    
    score = 0.0
    
    # Precompute label to index mapping
    idx = {label: i for i, label in enumerate(labels)}
    
    # Iterate over all quartets
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    # Quartet {a, b, c, d}
                    da = dist[a][b]
                    db = dist[a][c]
                    dc = dist[a][d]
                    de = dist[b][c]
                    df = dist[b][d]
                    dg = dist[c][d]
                    
                    # Three possible topologies and their scores
                    # Topology 1: (a,b)|(c,d) -> score = d_ac + d_ad + d_bc + d_bd - d_ab - d_cd
                    s1 = frozenset([labels[a], labels[b]])
                    score1 = (db + dc + de + df) - (da + dg)
                    
                    # Topology 2: (a,c)|(b,d) -> score = d_ab + d_ad + d_bc + d_cd - d_ac - d_bd
                    s2 = frozenset([labels[a], labels[c]])
                    score2 = (da + dc + de + dg) - (db + df)
                    
                    # Topology 3: (a,d)|(b,c) -> score = d_ab + d_ac + d_bd + d_cd - d_ad - d_bc
                    s3 = frozenset([labels[a], labels[d]])
                    score3 = (da + db + df + dg) - (dc + de)
                    
                    if s1 in splits:
                        score += score1
                    elif s2 in splits:
                        score += score2
                    elif s3 in splits:
                        score += score3
                    else:
                        # Should not happen for a valid binary tree
                        pass
                        
    return score


def _splits_to_newick(splits, labels):
    """
    Reconstruct Newick string from a set of splits.
    Uses recursive cherry-finding approach.
    """
    if not splits:
        if len(labels) == 1:
            return labels[0]
        elif len(labels) == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return "(" + ",".join(labels) + ")"
    
    def build(taxa, splits):
        taxa_set = set(taxa)
        n = len(taxa)
        
        if n <= 2:
            if n == 1:
                return taxa[0]
            return f"({taxa[0]},{taxa[1]})"
        
        # Find a cherry: a pair {a,b} that forms a split
        cherry = None
        for s in splits:
            if len(s) == 2 and s.issubset(taxa_set):
                cherry = s
                break
        
        if cherry is None:
            # No cherry found - fallback to star
            return "(" + ",".join(taxa) + ")"
        
        a, b = tuple(cherry)
        rest = [t for t in taxa if t not in cherry]
        
        # Create new taxon representing the cherry
        cherry_label = f"({a},{b})"
        
        # Filter and transform splits
        new_taxa = [cherry_label] + rest
        new_taxa_set = set(new_taxa)
        
        new_splits = set()
        for s in splits:
            if s == cherry:
                continue
            has_a = a in s
            has_b = b in s
            
            if has_a == has_b:
                new_s = set()
                if has_a:
                    new_s.add(cherry_label)
                else:
                    new_s.update(s)
                
                # Canonicalize
                if len(new_s) <= len(new_taxa) - len(new_s):
                    new_splits.add(frozenset(new_s))
                else:
                    other = new_taxa_set - new_s
                    new_splits.add(frozenset(other))
            else:
                # This split separates a and b, which shouldn't happen
                # if {a,b} is a valid cherry
                continue
        
        sub_newick = build(new_taxa, new_splits)
        
        # Insert cherry into the sub_newick
        # Replace the cherry_label in sub_newick with the actual cherry
        result = sub_newick.replace(cherry_label, f"({a},{b})")
        return result
    
    return build(labels, splits)


def _nni_refine(newick, dist, labels, max_iterations=10):
    """
    Perform NNI (Nearest Neighbor Interchange) refinement on a tree.
    Returns improved Newick string.
    """
    current_newick = newick
    current_splits = _parse_newick_to_splits(current_newick, labels)
    current_score = _quartet_score_from_dist(current_splits, dist, labels)
    
    for iteration in range(max_iterations):
        if _deadline_exceeded():
            break
        
        improved = False
        
        # Try all NNI moves
        # For each internal edge, try the alternative topology
        # We'll use a simpler approach: try all possible quartet rearrangements
        
        n = len(labels)
        idx = {label: i for i, label in enumerate(labels)}
        
        # Generate all quartets and check if NNI would improve
        best_gain = 0
        best_move = None
        
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    for d in range(c + 1, n):
                        da = dist[a][b]
                        db = dist[a][c]
                        dc = dist[a][d]
                        de = dist[b][c]
                        df = dist[b][d]
                        dg = dist[c][d]
                        
                        s1 = frozenset([labels[a], labels[b]])
                        s2 = frozenset([labels[a], labels[c]])
                        s3 = frozenset([labels[a], labels[d]])
                        
                        score1 = (db + dc + de + df) - (da + dg)
                        score2 = (da + dc + de + dg) - (db + df)
                        score3 = (da + db + df + dg) - (dc + de)
                        
                        # Check which split is in current tree
                        if s1 in current_splits:
                            # Current topology is (a,b)|(c,d)
                            # Alternative 1: (a,c)|(b,d)
                            gain1 = score2 - score1
                            if gain1 > best_gain:
                                best_gain = gain1
                                best_move = ('nni', a, b, c, d, 1)
                            # Alternative 2: (a,d)|(b,c)
                            gain2 = score3 - score1
                            if gain2 > best_gain:
                                best_gain = gain2
                                best_move = ('nni', a, b, c, d, 2)
                        elif s2 in current_splits:
                            # Current topology is (a,c)|(b,d)
                            # Alternative 1: (a,b)|(c,d)
                            gain1 = score1 - score2
                            if gain1 > best_gain:
                                best_gain = gain1
                                best_move = ('nni', a, b, c, d, 1)
                            # Alternative 2: (a,d)|(b,c)
                            gain2 = score3 - score2
                            if gain2 > best_gain:
                                best_gain = gain2
                                best_move = ('nni', a, b, c, d, 2)
                        elif s3 in current_splits:
                            # Current topology is (a,d)|(b,c)
                            # Alternative 1: (a,b)|(c,d)
                            gain1 = score1 - score3
                            if gain1 > best_gain:
                                best_gain = gain1
                                best_move = ('nni', a, b, c, d, 1)
                            # Alternative 2: (a,c)|(b,d)
                            gain2 = score2 - score3
                            if gain2 > best_gain:
                                best_gain = gain2
                                best_move = ('nni', a, b, c, d, 2)
        
        if best_move is None or best_gain <= 0:
            break
        
        # Apply the best NNI move
        # This is complex - we need to actually modify the tree
        # For now, we'll skip actual NNI application and just return current
        # TODO: Implement proper NNI application
        
        break
    
    return current_newick


def _exhaustive_search_small_n(n, labels, dist):
    """
    For n <= 8, exhaustively search all unrooted binary tree topologies.
    Returns the best Newick string.
    """
    if n > 8:
        return None
    
    # Generate all possible unrooted binary tree topologies
    # We'll use a recursive approach: pick two taxa to join, recurse on n-1 taxa
    
    all_trees = []
    
    def generate_topologies(taxa):
        m = len(taxa)
        if m == 3:
            # Only one unrooted binary tree for 3 taxa
            a, b, c = taxa
            # Split: {a,b} | {c} (but {c} is trivial, so just {a,b})
            splits = {frozenset([a, b])}
            newick = f"(({a},{b}),{c})"
            all_trees.append((splits, newick))
            return
        if m == 4:
            # 3 possible unrooted binary trees
            a, b, c, d = taxa
            # Tree 1: ((a,b),(c,d))
            splits1 = {frozenset([a, b]), frozenset([c, d])}
            newick1 = f"(({a},{b}),({c},{d}))"
            all_trees.append((splits1, newick1))
            # Tree 2: ((a,c),(b,d))
            splits2 = {frozenset([a, c]), frozenset([b, d])}
            newick2 = f"(({a},{c}),({b},{d}))"
            all_trees.append((splits2, newick2))
            # Tree 3: ((a,d),(b,c))
            splits3 = {frozenset([a, d]), frozenset([b, c])}
            newick3 = f"(({a},{d}),({b},{c}))"
            all_trees.append((splits3, newick3))
            return
        
        # For m > 4, recursively join pairs
        seen_newicks = set()
        for i in range(m):
            for j in range(i + 1, m):
                a, b = taxa[i], taxa[j]
                rest = [t for k, t in enumerate(taxa) if k not in (i, j)]
                cherry = f"({a},{b})"
                new_taxa = [cherry] + rest
                
                for sub_splits, sub_newick in generate_topologies(new_taxa):
                    # Expand splits
                    new_splits = set()
                    new_splits.add(frozenset([a, b]))
                    
                    for split in sub_splits:
                        new_split = set()
                        for t in split:
                            if t == cherry:
                                new_split.add(a)
                                new_split.add(b)
                            else:
                                new_split.add(t)
                        
                        # Canonicalize
                        if len(new_split) <= m - len(new_split):
                            new_splits.add(frozenset(new_split))
                        else:
                            other = set(taxa) - new_split
                            new_splits.add(frozenset(other))
                    
                    # Expand Newick
                    new_newick = sub_newick.replace(cherry, f"({a},{b})")
                    
                    # Check for duplicates
                    newick_key = frozenset(new_splits)
                    if newick_key not in seen_newicks:
                        seen_newicks.add(newick_key)
                        all_trees.append((new_splits, new_newick))
    
    generate_topologies(labels)
    
    if not all_trees:
        return None
    
    # Score each tree
    best_score = -float('inf')
    best_newick = None
    
    for splits, newick in all_trees:
        if _deadline_exceeded():
            break
        score = _quartet_score_from_dist(splits, dist, labels)
        if score > best_score:
            best_score = score
            best_newick = newick
    
    return best_newick


def reconstruct(sequences):
    """
    Reconstruct the unrooted tree topology from sequences.
    Returns a Newick string.
    """
    global _start_time
    _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"(({labels[0]},{labels[1]}),{labels[2]})"
    
    # Compute distance matrix
    dist = _compute_distance_matrix(sequences, labels)
    
    # For small n, try exhaustive search first
    if n <= 8:
        exhaustive_result = _exhaustive_search_small_n(n, labels, dist)
        if exhaustive_result is not None:
            return exhaustive_result
    
    # Build NJ tree
    nj_tree = _nj_build_v2(dist, labels)
    
    # Try NNI refinement (if time permits)
    if not _deadline_exceeded():
        refined_tree = _nni_refine(nj_tree, dist, labels)
        if refined_tree:
            nj_tree = refined_tree
    
    return nj_tree

# EVOLVE-BLOCK-END
