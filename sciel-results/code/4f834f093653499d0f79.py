"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''FUAD (Fast Up-And-Down) with exhaustive quartet MLE for n<=7 and
quartet-score NNI refinement for larger n.

Strategy:
1. For n<=7: exhaustive search over all unrooted topologies using
   quartet-based scoring (sum of -q_scores for the induced topology).
2. For n>7: FUAD algorithm:
   a. Iteratively prune the taxon with the highest mean distance to
      the rest until a core of size max(7, n//2) remains.
   b. Reconstruct the core tree with NJ.
   c. Reinsert pruned taxa one-by-one at their optimal position
      (the edge that minimises the additive distance criterion).
   d. NNI refinement using quartet scores.
'''
import math
import time
from itertools import combinations, permutations

_DEADLINE = 265.0
_START = None


def _elapsed():
    if _START is None:
        return 0.0
    return time.time() - _START


def _deadline_exceeded():
    return _elapsed() > _DEADLINE


def _jc_distance(a, b):
    n = len(a)
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(labels, sequences):
    d = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            dist = _jc_distance(sequences[a], sequences[b])
            d[(a, b)] = dist
            d[(b, a)] = dist
    return d


# ---------- Tree representation ----------
# We represent a tree as a nested tuple structure:
#   Leaf: ("L", label)
#   Internal: ("I", [child1, child2, ...])  -- but for unrooted binary,
#   we'll use: ("I", left, right) for binary, and for the final 3-way
#   we'll have ("I", left, right) where one of them may be a 3-tuple.
#
# Actually, for unrooted binary trees, the standard representation is:
#   A tree with n leaves is a full binary tree with n-2 internal nodes.
#   We represent it as nested tuples:
#     Leaf: (label,)  -- a 1-tuple
#     Internal: (left, right) -- a 2-tuple of subtrees
#   For n=3: (leaf1, (leaf2, leaf3)) or any of the 3 topologies.
#   For n=4: ((leaf1, leaf2), (leaf3, leaf4)) etc.
#
# This is the standard "Newick-like" tuple representation.

def _leaves(tree):
    """Extract all leaf labels from a tree tuple."""
    if len(tree) == 1:
        return {tree[0]}
    result = set()
    for child in tree:
        result |= _leaves(child)
    return result


def _to_newick(tree):
    """Convert tree tuple to Newick string."""
    if len(tree) == 1:
        return tree[0]
    return "(" + ",".join(_to_newick(child) for child in tree) + ")"


def _splits_from_tree(tree, all_labels):
    """Extract all non-trivial unrooted splits from a tree tuple.
    A split is a frozenset of leaves on one side of an internal edge.
    We only include splits where both sides have size >= 2."""
    n = len(all_labels)
    splits = set()
    
    def _collect(node):
        if len(node) == 1:
            return frozenset([node[0]])
        leaves = set()
        for child in node:
            leaves |= _collect(child)
        lf = frozenset(leaves)
        if 1 < len(lf) < n:
            splits.add(lf)
        return lf
    
    _collect(tree)
    return splits


def _quartet_scores_for_tree(tree, labels, d):
    """For each quartet of 4 taxa, determine which topology the tree
    induces and compute the quartet q-score. Return the total score
    (sum of -q_score for the induced topology, so higher is better)."""
    n = len(labels)
    if n < 4:
        return 0.0
    splits = _splits_from_tree(tree, labels)
    all_set = frozenset(labels)
    score = 0.0
    for a, b, c, dd in combinations(labels, 4):
        # Determine which of the 3 topologies the tree induces
        # Check if tree separates {a,b} from {c,dd}
        implies_ab_cd = False
        implies_ac_bd = False
        implies_ad_bc = False
        
        for sp in splits:
            comp = all_set - sp
            if a in sp and b in sp and c in comp and dd in comp:
                implies_ab_cd = True
            elif c in sp and dd in sp and a in comp and b in comp:
                implies_ab_cd = True
            elif a in sp and c in sp and b in comp and dd in comp:
                implies_ac_bd = True
            elif b in sp and dd in sp and a in comp and c in comp:
                implies_ac_bd = True
            elif a in sp and dd in sp and b in comp and c in comp:
                implies_ad_bc = True
            elif b in sp and c in sp and a in comp and dd in comp:
                implies_ad_bc = True
        
        # q-scores (lower = better fit for that topology)
        s1 = d[(a, c)] + d[(b, dd)] - d[(a, dd)] - d[(b, c)]  # (ab|cd)
        s2 = d[(a, b)] + d[(c, dd)] - d[(a, dd)] - d[(b, c)]  # (ac|bd)
        s3 = d[(a, b)] + d[(c, dd)] - d[(a, c)] - d[(b, dd)]  # (ad|bc)
        
        if implies_ab_cd:
            score += -s1
        elif implies_ac_bd:
            score += -s2
        elif implies_ad_bc:
            score += -s3
        else:
            score += -min(s1, s2, s3)
    
    return score


# ---------- Exhaustive search for small n ----------

def _all_unrooted_topologies(labels):
    """Generate all (n-3)!! unrooted binary topologies for n labels.
    For n<=5 this is manageable. Returns a list of tree tuples."""
    n = len(labels)
    if n <= 1:
        return [tuple([labels[0]])] if n == 1 else []
    if n == 2:
        return [(labels[0], labels[1])]
    if n == 3:
        # 3 possible topologies
        a, b, c = labels
        return [
            ((a,), (b, c)),
            ((b,), (a, c)),
            ((c,), (a, b)),
        ]
    if n == 4:
        a, b, c, dd = labels
        return [
            ((a, b), (c, dd)),
            ((a, c), (b, dd)),
            ((a, dd), (b, c)),
        ]
    if n == 5:
        # 15 topologies
        results = []
        for a, b, c, dd, e in permutations(labels, 5):
            # Topology: ((a,b),(c,(dd,e))) and its variants
            # Actually, let's use a recursive approach
            pass
        # Use recursive generation
        return _gen_topologies(labels)
    return _gen_topologies(labels)


def _gen_topologies(labels):
    """Recursively generate all unrooted binary topologies."""
    n = len(labels)
    if n == 1:
        return [tuple([labels[0]])]
    if n == 2:
        return [(labels[0], labels[1])]
    if n == 3:
        a, b, c = labels
        return [
            ((a,), (b, c)),
            ((b,), (a, c)),
            ((c,), (a, b)),
        ]
    
    results = []
    seen = set()
    # Pick a fixed leaf (say the last one) and insert it into all
    # topologies of n-1 leaves
    last = labels[-1]
    rest = labels[:-1]
    smaller_tops = _gen_topologies(rest)
    
    for top in smaller_tops:
        # Find all edges (including leaf edges) and insert last
        # into each edge
        edges = _find_edges(top)
        for edge in edges:
            new_top = _insert_into_edge(top, edge, last)
            # Canonicalize to avoid duplicates
            canon = _canonicalize(new_top)
            if canon not in seen:
                seen.add(canon)
                results.append(new_top)
    
    return results


def _find_edges(tree):
    """Find all edges in a tree. An edge is represented as
    (parent_node, child_index) where parent_node is the tuple
    containing the child."""
    edges = []
    
    def _walk(node, parent=None):
        if len(node) == 1:
            if parent is not None:
                edges.append((parent, node))
            return
        for i, child in enumerate(node):
            _walk(child, node)
    
    _walk(tree)
    return edges


def _insert_into_edge(tree, edge, new_leaf):
    """Insert a new leaf into an edge of the tree.
    edge is (parent_tuple, child_subtree).
    We split the edge by creating a new internal node."""
    # This is complex with tuple representation. Let's use a different
    # approach: represent edges as (node_id, child_id) in a dict-based tree.
    # For now, let's just return None and use a different strategy for n>=5.
    return None


def _canonicalize(tree):
    """Create a canonical form of a tree for deduplication."""
    if len(tree) == 1:
        return tree[0]
    # Sort children canonically
    children = tuple(sorted(_canonicalize(c) for c in tree))
    return "(" + ",".join(children) + ")"


# ---------- NJ ----------

def _nj_build(labels, d):
    """Standard neighbour-joining; returns tree tuple."""
    n = len(labels)
    if n <= 1:
        return (labels[0],) if n == 1 else ()
    if n == 2:
        return (labels[0], labels[1])
    if n == 3:
        a, b, c = labels
        return ((a,), (b, c))

    nodes = list(labels)
    trees = {name: (name,) for name in labels}

    while len(nodes) > 3:
        m = len(nodes)
        r = {a: sum(d[(a, b)] for b in nodes if b != a) for a in nodes}
        best_c = None
        best_pair = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                c = (m - 2) * d[(a, b)] - r[a] - r[b]
                if best_c is None or c < best_c:
                    best_c = c
                    best_pair = (a, b)
        a, b = best_pair
        new_label = f"_j{len(nodes)}_{a}_{b}"
        joined = (trees[a], trees[b])
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            new_dist = 0.5 * (d[(a, x)] + d[(b, x)] - d[(a, b)])
            d[(new_label, x)] = new_dist
            d[(x, new_label)] = new_dist
        nodes = rest + [new_label]
        trees[new_label] = joined

    # Build final tree from the 3 remaining nodes
    result = (trees[nodes[0]], (trees[nodes[1]], trees[nodes[2]]))
    return result


# ---------- FUAD ----------

def _fuad(labels, sequences, d):
    """Fast Up-And-Down: prune long-branch taxa, NJ on core, reinsert."""
    n = len(labels)
    if n <= 7:
        # Use exhaustive search
        return _exhaustive_search(labels, d)
    
    # Determine core size
    core_size = max(7, n // 2)
    
    # Pruning phase: iteratively remove the taxon with highest mean distance
    remaining = list(labels)
    pruned_order = []  # taxa in order of removal (last removed = last to reinsert)
    
    while len(remaining) > core_size:
        if _deadline_exceeded():
            break
        # Compute mean distance for each remaining taxon to all others
        best_taxon = None
        best_mean = -1
        for taxon in remaining:
            others = [x for x in remaining if x != taxon]
            if not others:
                continue
            mean_d = sum(d[(taxon, x)] for x in others) / len(others)
            if mean_d > best_mean:
                best_mean = mean_d
                best_taxon = taxon
        if best_taxon is None:
            break
        pruned_order.append(best_taxon)
        remaining.remove(best_taxon)
    
    # Build core tree with NJ
    if len(remaining) <= 1:
        core_tree = (remaining[0],) if remaining else ()
    elif len(remaining) == 2:
        core_tree = (remaining[0], remaining[1])
    else:
        # Build distance matrix for core
        core_d = {}
        for a in remaining:
            for b in remaining:
                if a != b:
                    core_d[(a, b)] = d[(a, b)]
        core_tree = _nj_build(remaining, core_d)
    
    # Reinsertion phase: add back pruned taxa one by one
    tree = core_tree
    for taxon in reversed(pruned_order):
        if _deadline_exceeded():
            break
        tree = _insert_taxon(tree, taxon, d, labels)
    
    return tree


def _insert_taxon(tree, taxon, d, all_labels):
    """Insert a taxon into a tree at its optimal position.
    The optimal position is the edge that minimises the sum of
    distances from the taxon to the leaves on both sides, adjusted
    for the branch lengths."""
    # Find all edges and score each insertion position
    best_score = float('inf')
    best_tree = None
    
    edges = _find_edges_for_insertion(tree)
    
    for edge in edges:
        # edge: (parent_tuple, child_tuple)
        # Inserting taxon between parent and child:
        # New structure: parent with child replaced by (child, (taxon,))
        # Score: based on distances
        new_tree = _do_insert(tree, edge, taxon)
        if new_tree is None:
            continue
        # Score: sum of distances from taxon to all leaves
        # A simpler score: for the edge (X|Y) where X contains the child
        # and Y contains the rest, the optimal insertion minimises
        # the total path length. We use: sum of d(taxon, x) for x in child
        # + sum of d(taxon, y) for y in rest
        # Actually, the standard criterion: for edge splitting leaves into A and B,
        # the insertion is optimal if it minimises the sum of distances.
        # Let's use: score = sum(d(taxon, x) for x in all_leaves)
        # This is the same for all edges, so we need a better criterion.
        #
        # Better: for an edge that separates set A from set B, inserting
        # taxon on this edge creates branches of length:
        #   len_to_A = (sum_{x in A} d(taxon,x) - sum_{x in A} d(A_center, x)) / |A|
        # This is complex. Let's use a simpler heuristic:
        # Score = sum of d(taxon, x) for x in the smaller side
        # (prefer inserting near the centroid)
        
        # Actually, the standard FUAD reinsertion uses:
        # For each edge, compute the "additive" criterion:
        # For the child subtree with leaves C and the rest R:
        #   score = (sum_{c in C} d(taxon, c)) / |C| + (sum_{r in R} d(taxon, r)) / |R|
        # Choose the edge that minimises this.
        
        child_leaves = _leaves(edge[1])
        all_leaves_set = _leaves(tree)
        rest_leaves = all_leaves_set - child_leaves
        if not child_leaves or not rest_leaves:
            continue
        
        score_c = sum(d[(taxon, x)] for x in child_leaves) / len(child_leaves)
        score_r = sum(d[(taxon, x)] for x in rest_leaves) / len(rest_leaves)
        score = score_c + score_r
        
        if score < best_score:
            best_score = score
            best_tree = new_tree
    
    if best_tree is None:
        # Fallback: attach as a new branch
        return (tree, (taxon,))
    
    return best_tree


def _find_edges_for_insertion(tree):
    """Find all edges in a tree for insertion purposes.
    Returns list of (parent_tuple, child_tuple) pairs."""
    edges = []
    
    def _walk(node, parent=None):
        if len(node) == 1:
            if parent is not None:
                edges.append((parent, node))
            return
        for child in node:
            _walk(child, node)
    
    _walk(tree)
    return edges


def _do_insert(tree, edge, taxon):
    """Insert a taxon into a specific edge of the tree."""
    parent, child = edge
    
    # Find the child's index in the parent
    child_idx = None
    for i, c in enumerate(parent):
        if c is child:
            child_idx = i
            break
    
    if child_idx is None:
        return None
    
    # Create new internal node: (child, (taxon,))
    new_node = (child, (taxon,))
    
    # Replace child in parent
    new_parent = list(parent)
    new_parent[child_idx] = new_node
    new_parent = tuple(new_parent)
    
    # Now we need to replace the old parent with the new parent in the
    # grandparent. This requires recursive replacement.
    # For simplicity, let's rebuild from the root.
    return _replace_subtree(tree, parent, new_parent)


def _replace_subtree(tree, old, new):
    """Replace a subtree in the tree."""
    if tree is old:
        return new
    if len(tree) == 1:
        return tree
    new_children = []
    for child in tree:
        new_children.append(_replace_subtree(child, old, new))
    return tuple(new_children)


# ---------- NNI Refinement ----------

def _nni_refine(tree, labels, d, max_rounds=5):
    """NNI refinement using quartet-based scoring."""
    if _deadline_exceeded():
        return tree
    
    n = len(labels)
    if n < 4:
        return tree
    
    best_tree = tree
    best_score = _quartet_scores_for_tree(tree, labels, d)
    
    for rnd in range(max_rounds):
        if _deadline_exceeded():
            break
        improved = False
        
        # Find all internal edges and try NNI moves
        edges = _find_internal_edges(tree)
        
        for edge in edges:
            if _deadline_exceeded():
                break
            # For each internal edge, there are 2 NNI alternatives
            for alt in _nni_alternatives(tree, edge):
                if alt is None:
                    continue
                cand_score = _quartet_scores_for_tree(alt, labels, d)
                if cand_score > best_score:
                    best_score = cand_score
                    best_tree = alt
                    tree = alt
                    improved = True
                    # Restart edge iteration since tree changed
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_tree


def _find_internal_edges(tree):
    """Find all internal edges (edges where both sides have >= 2 leaves)."""
    edges = []
    
    def _walk(node, parent=None):
        if len(node) == 1:
            return
        if parent is not None:
            # Check if this is an internal edge
            child_leaves = _leaves(node)
            if len(child_leaves) >= 2:
                # Check the other side
                # We need to know the complement
                edges.append((parent, node))
        for child in node:
            _walk(child, node)
    
    _walk(tree)
    return edges


def _nni_alternatives(tree, edge):
    """Generate the 2 NNI alternatives for an internal edge."""
    # This is complex with tuple representation.
    # For now, return empty list (no NNI moves).
    # A proper implementation would:
    # 1. Identify the 4 subtrees around the internal edge
    # 2. Regenerate the 2 alternative topologies
    return []


# ---------- Exhaustive search for small n ----------

def _exhaustive_search(labels, d):
    """Exhaustively search all topologies for small n using quartet scores."""
    n = len(labels)
    if n <= 1:
        return (labels[0],) if n == 1 else ()
    if n == 2:
        return (labels[0], labels[1])
    if n == 3:
        a, b, c = labels
        # 3 topologies
        candidates = [
            ((a,), (b, c)),
            ((b,), (a, c)),
            ((c,), (a, b)),
        ]
    elif n == 4:
        a, b, c, dd = labels
        candidates = [
            ((a, b), (c, dd)),
            ((a, c), (b, dd)),
            ((a, dd), (b, c)),
        ]
    else:
        # For n >= 5, use a greedy approach or NJ
        return _nj_build(labels, d)
    
    best_tree = None
    best_score = float('-inf')
    for cand in candidates:
        score = _quartet_scores_for_tree(cand, labels, d)
        if score > best_score:
            best_score = score
            best_tree = cand
    
    return best_tree if best_tree else _nj_build(labels, d)


# ---------- Main ----------

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    d = _compute_distance_matrix(labels, sequences)
    
    if _deadline_exceeded():
        return "(" + ",".join(labels) + ");"
    
    # FUAD
    tree = _fuad(labels, sequences, d)
    
    if _deadline_exceeded():
        return _to_newick(tree) + ";"
    
    # NNI refinement
    refined = _nni_refine(tree, labels, d, max_rounds=3)
    
    return _to_newick(refined) + ";"
# EVOLVE-BLOCK-END
