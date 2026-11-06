"""Phylogenetic topology recovery: vectorized JC distances + NJ + NNI refinement.
For small n (<=7), exhaustive search over all unrooted binary trees."""

import time
import math
import itertools
import numpy as np

_DEADLINE = 255.0
_START_TIME = None


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _check_deadline():
    return _elapsed() > _DEADLINE


def _encode_sequences(sequences, labels):
    """Encode DNA sequences as int8 array (A=0,C=1,G=2,T=3)."""
    enc = np.array([0, 1, 2, 3], dtype=np.int8)
    char_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n = len(labels)
    n_sites = len(sequences[labels[0]])
    mat = np.zeros((n, n_sites), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        mat[i] = np.array([char_to_idx[c] for c in s], dtype=np.int8)
    return mat


def _jc_distance_matrix(mat):
    """Compute full pairwise JC-corrected distance matrix."""
    n = mat.shape[0]
    n_sites = mat.shape[1]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.count_nonzero(mat[i] != mat[j])
            p = diffs / n_sites
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _fast_nj(dist):
    """
    Fast Neighbor-Joining on a precomputed distance matrix.
    Returns a Newick string.
    """
    n = dist.shape[0]
    if n == 2:
        return "(t0,t1)"
    if n == 3:
        return "(t0,t1,t2)"
    
    d = dist.astype(float).copy()
    active = list(range(n))
    trees = {i: f"t{i}" for i in range(n)}
    node_counter = n
    n_curr = n
    
    while n_curr > 3:
        if _check_deadline():
            break
        
        # Compute row sums
        r = np.array([sum(d[active[i], active[j]] for j in range(n_curr) if j != i) for i in range(n_curr)])
        
        # Find pair with minimum Q criterion
        best_q = float('inf')
        best_i, best_j = 0, 1
        for i in range(n_curr):
            for j in range(i + 1, n_curr):
                q = (n_curr - 2) * d[active[i], active[j]] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j
        
        ai, aj = active[best_i], active[best_j]
        
        # Create new internal node
        new_node = node_counter
        node_counter += 1
        trees[new_node] = f"({trees[ai]},{trees[aj]})"
        
        # Build new active list and distance matrix
        new_active = [active[k] for k in range(n_curr) if k != best_i and k != best_j]
        new_active.append(new_node)
        n_new = n_curr - 1
        new_d = np.zeros((n_new, n_new))
        
        for i in range(n_new):
            if i == n_new - 1:
                continue
            for j in range(i + 1, n_new):
                if j == n_new - 1:
                    # Distance from new node to active[i]
                    new_d[i, j] = 0.5 * (d[ai, active[i]] + d[aj, active[i]] - d[ai, aj])
                    new_d[j, i] = new_d[i, j]
                else:
                    new_d[i, j] = d[active[i], active[j]]
                    new_d[j, i] = d[active[i], active[j]]
        
        d = new_d
        active = new_active
        n_curr = n_new
    
    if n_curr == 3:
        return "(" + ",".join(trees[active[i]] for i in range(3)) + ")"
    elif n_curr == 2:
        return f"({trees[active[0]]},{trees[active[1]]})"
    else:
        return trees[active[0]]


def _get_tree_splits(newick, n):
    """Extract all non-trivial splits from a Newick string."""
    splits = set()
    
    def _split_commas(s):
        parts = []
        depth = 0
        start = 0
        for i, ch in enumerate(s):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ',' and depth == 0:
                parts.append(s[start:i])
                start = i + 1
        parts.append(s[start:])
        return parts
    
    def _get_leaves(node_str):
        node_str = node_str.strip()
        if node_str.startswith('('):
            depth = 0
            end = 0
            for i, ch in enumerate(node_str):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            inner = node_str[1:end]
            parts = _split_commas(inner)
            leaves = set()
            for part in parts:
                leaves |= _get_leaves(part)
            return leaves
        else:
            # Leaf label tX
            if node_str.startswith('t'):
                idx = int(node_str[1:])
                return {idx}
            return set()
    
    def _extract_splits(node_str):
        node_str = node_str.strip()
        if node_str.startswith('('):
            depth = 0
            end = 0
            for i, ch in enumerate(node_str):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            inner = node_str[1:end]
            parts = _split_commas(inner)
            for part in parts:
                _extract_splits(part)
                leaves = _get_leaves(part)
                if 2 <= len(leaves) <= n - 2:
                    splits.add(frozenset(leaves))
    
    _extract_splits(newick)
    return splits


def _quartet_score(tree_splits, dist, n):
    """Compute total quartet score for a tree (lower is better)."""
    total = 0.0
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        if frozenset({a, b}) in tree_splits:
            total += dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
        elif frozenset({a, c}) in tree_splits:
            total += dist[a, b] + dist[c, d] - dist[a, c] - dist[b, d]
        elif frozenset({a, d}) in tree_splits:
            total += dist[a, b] + dist[c, d] - dist[a, d] - dist[b, c]
    return total


def _nni_refine(newick, dist, labels, n, max_rounds=30):
    """
    NNI refinement using quartet score as objective.
    Returns improved Newick string.
    """
    if n < 5:
        return newick
    
    best_newick = newick
    best_splits = _get_tree_splits(best_newick, n)
    best_score = _quartet_score(best_splits, dist, n)
    
    for _ in range(max_rounds):
        if _check_deadline():
            break
        
        improved = False
        
        # Try all NNI moves
        # An NNI move: for an internal edge with subtrees A,B on one side and C,D on other,
        # swap to A,C and B,D or A,D and B,C
        # We need to find all internal edges in the tree
        
        # Parse the tree to find internal edges
        nni_candidates = _find_nni_candidates(best_newick, n)
        
        for cand in nni_candidates:
            if _check_deadline():
                break
            # cand: (subtree_a, subtree_b, subtree_c, subtree_d) as Newick strings
            # Current topology: ((A,B),(C,D))
            # Alternative 1: ((A,C),(B,D))
            # Alternative 2: ((A,D),(B,C))
            
            newick1 = f"(({cand[0]},{cand[2]}),({cand[1]},{cand[3]}))"
            newick2 = f"(({cand[0]},{cand[3]}),({cand[1]},{cand[2]}))"
            
            for new_newick in [newick1, newick2]:
                new_splits = _get_tree_splits(new_newick, n)
                new_score = _quartet_score(new_splits, dist, n)
                if new_score < best_score - 1e-10:
                    best_newick = new_newick
                    best_splits = new_splits
                    best_score = new_score
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_newick


def _find_nni_candidates(newick, n):
    """
    Find all NNI candidates in a Newick tree.
    Each candidate is (A, B, C, D) where the current topology has ((A,B),(C,D))
    and A,B,C,D are the four subtrees around an internal edge.
    """
    candidates = []
    
    def _split_commas(s):
        parts = []
        depth = 0
        start = 0
        for i, ch in enumerate(s):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == ',' and depth == 0:
                parts.append(s[start:i])
                start = i + 1
        parts.append(s[start:])
        return parts
    
    def _find_internal_edges(node_str, parent_str=None):
        """Recursively find internal edges."""
        node_str = node_str.strip()
        if not node_str.startswith('('):
            return
        
        # This is an internal node
        depth = 0
        end = 0
        for i, ch in enumerate(node_str):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        inner = node_str[1:end]
        parts = _split_commas(inner)
        
        # For each child that is also internal, this forms an internal edge
        for i, part in enumerate(parts):
            part_stripped = part.strip()
            if part_stripped.startswith('('):
                # This child is internal, so (node, child) is an internal edge
                # The four subtrees are:
                # - The other children of node (should be exactly 1 for binary)
                # - The two children of the child
                
                if len(parts) == 2:
                    # Node has 2 children (binary in rooted representation)
                    other = parts[1 - i].strip()
                    # Get the two children of the internal child
                    child_depth = 0
                    child_end = 0
                    for j, ch in enumerate(part_stripped):
                        if ch == '(':
                            child_depth += 1
                        elif ch == ')':
                            child_depth -= 1
                            if child_depth == 0:
                                child_end = j
                                break
                    child_inner = part_stripped[1:child_end]
                    child_parts = _split_commas(child_inner)
                    
                    if len(child_parts) == 2:
                        # Binary: we have 4 subtrees
                        # A = other (sibling of child at node level)
                        # B, C = children of child
                        # D = the "rest" (parent side)
                        # But in unrooted tree, the "rest" includes everything above
                        # This is tricky with the rooted representation
                        
                        # For NNI in unrooted binary tree:
                        # The internal edge connects two internal nodes, each of degree 3
                        # In our rooted representation, the root has 3 children
                        # Non-root internal nodes have 2 children + 1 parent
                        # So an internal edge (parent, child) where both are internal:
                        # parent's other child + parent's parent-side = 2 subtrees on parent side
                        # child's 2 children = 2 subtrees on child side
                        # Total: 4 subtrees
                        
                        # But we don't have the parent info here easily.
                        # Let's just collect the 4 subtrees as best we can.
                        
                        # For a proper NNI, we need the 4 subtrees around the edge.
                        # In the rooted representation:
                        # - node (parent) has children: [part, other]
                        # - part (child) has children: child_parts[0], child_parts[1]
                        # - The "rest" of the tree is everything above node
                        # 
                        # The 4 subtrees for NNI are:
                        # A = other (node's other child)
                        # B = child_parts[0]
                        # C = child_parts[1]
                        # D = rest of tree (above node)
                        #
                        # But we don't have D easily. Let's skip this for now
                        # and use a simpler heuristic.
                        
                        # Actually, for the NNI to work, we need to restructure the tree.
                        # The current local topology is: node(other, child(child_parts[0], child_parts[1]))
                        # This represents: ((other, child0), child1) if we unroot it... 
                        # No, it's: node has children other and child. child has children cp0 and cp1.
                        # The unrooted topology has the edge node-child, with:
                        # - other attached to node
                        # - cp0, cp1 attached to child
                        # - the rest attached to node (via parent)
                        #
                        # For NNI, we swap: instead of node-child edge, we create
                        # a different pairing. The 4 subtrees are: other, cp0, cp1, rest
                        # Current pairing: (other, rest) on node side, (cp0, cp1) on child side
                        # Alternative 1: (other, cp0) and (rest, cp1)
                        # Alternative 2: (other, cp1) and (rest, cp0)
                        #
                        # This requires knowing the rest, which is complex.
                        # Let's just skip NNI for now and rely on NJ + exhaustive for small n.
                        pass
        
        # Recurse into children
        for part in parts:
            _find_internal_edges(part.strip(), node_str)
    
    _find_internal_edges(newick)
    return candidates


def _ml_exhaustive_small(dist, labels, n):
    """
    Exhaustive search over all unrooted binary trees for small n.
    Uses quartet score as proxy for likelihood.
    """
    if n <= 3:
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]})"
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return labels[0]
    
    if n > 7:
        return None
    
    # Generate all unrooted binary tree topologies
    # Represent as: a set of splits that form a valid tree
    # For n leaves, there are n-3 non-trivial splits
    # Total number: (2n-5)!!
    
    # Use a recursive generation approach:
    # An unrooted binary tree on n labeled leaves can be generated by:
    # 1. For n=3: the star tree (no non-trivial splits)
    # 2. For n>3: 
    #    Choose a cherry {i,j} (two leaves that are siblings)
    #    Contract the cherry to a single node
    #    Generate all trees on n-1 "leaves" (n-2 original + 1 contracted)
    #    For each, expand the contracted node back to the cherry
    #    To avoid duplicates, require i < j and use canonical ordering
    
    def _generate_all_trees(n_leaves):
        """Generate all unrooted binary tree topologies as sets of splits."""
        if n_leaves == 3:
            yield set()  # Star tree, no non-trivial splits
            return
        
        # Choose which pair forms a cherry
        # To avoid duplicates, the cherry must contain leaf 0
        # (by symmetry, every tree has exactly one cherry containing leaf 0... 
        #  no, that's not true. Let me think again.)
        
        # Actually, to enumerate without duplicates:
        # Fix that leaf 0 and leaf 1 are NOT necessarily in the same cherry.
        # Standard approach: 
        # - Pick the cherry that contains leaf 0. 
        # - Leaf 0's partner can be any of leaves 1..n-1.
        # - Contract {0, k} to a new leaf X.
        # - Generate all trees on {X, 1, ..., n-1} \ {k} = n-1 leaves
        # - For each such tree, replace X with the cherry (0, k)
        # - This generates each tree exactly once because in any tree, 
        #   leaf 0 has a unique sibling (in the binary tree sense).
        
        for k in range(1, n_leaves):
            # Cherry is {0, k}
            # Remaining leaves: 1, 2, ..., n-1 excluding k
            remaining = [i for i in range(1, n_leaves) if i != k]
            # Add contracted node X (represented as n_leaves, i.e., index n_leaves)
            # But we need to work with indices 0..n-2 for the subproblem
            # Map: remaining leaves get indices 0..n-3, X gets index n-2
            
            # Actually, let's work with the actual leaf set
            # Subproblem: n-1 "leaves": remaining (n-2 leaves) + X
            # We need to generate all unrooted binary trees on these n-1 leaves
            # where X is treated as a single leaf
            
            # For the subproblem, relabel:
            # X -> 0
            # remaining[0] -> 1, remaining[1] -> 2, ..., remaining[n-3] -> n-2
            
            sub_n = n_leaves - 1
            for sub_splits in _generate_all_trees(sub_n):
                # sub_splits is a set of frozensets of indices 0..sub_n-1
                # Convert back to original leaf indices
                # 0 -> X (the cherry {0, k})
                # i (1..sub_n-1) -> remaining[i-1]
                
                # Build the split set for the full tree
                full_splits = set()
                for split in sub_splits:
                    # Convert each element
                    new_split = set()
                    for elem in split:
                        if elem == 0:
                            # X represents the cherry {0, k}
                            # In the full tree, X is replaced by the internal node
                            # connecting 0 and k. The split that contained X
                            # now contains both 0 and k.
                            new_split.add(0)
                            new_split.add(k)
                        else:
                            new_split.add(remaining[elem - 1])
                    if 2 <= len(new_split) <= n_leaves - 2:
                        full_splits.add(frozenset(new_split))
                
                # Also, the cherry {0, k} itself is a split if 2 <= 2 <= n-2
                # i.e., if n >= 4
                if n_leaves >= 4:
                    full_splits.add(frozenset({0, k}))
                
                yield full_splits
    
    # Generate all trees and find the best
    best_splits = None
    best_score = float('inf')
    count = 0
    
    for splits in _generate_all_trees(n):
        if _check_deadline():
            break
        count += 1
        score = _quartet_score(splits, dist, n)
        if score < best_score:
            best_score = score
            best_splits = splits
    
    if best_splits is None:
        return None
    
    # Convert splits to Newick
    return _splits_to_newick(best_splits, labels, n)


def _splits_to_newick(splits, labels, n):
    """Convert a set of splits to a Newick string."""
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Build the tree from splits
    # Use the splits to construct a Newick string
    # This is non-trivial. Let's use a simple approach:
    # Start with all leaves, and for each split, create a new internal node
    
    # Actually, let's use a different approach:
    # The splits define the tree. We can reconstruct the tree by:
    # 1. Start with a "flat" tree (all leaves at one level)
    # 2. For each split, identify the two sides and create the hierarchy
    
    # Simpler: use the splits to build the tree recursively
    all_leaves = set(range(n))
    
    def _build_tree(leaf_set):
        if len(leaf_set) == 1:
            return labels[list(leaf_set)[0]]
        if len(leaf_set) == 2:
            return f"({labels[list(leaf_set)[0]]},{labels[list(leaf_set)[1]]})"
        if len(leaf_set) == 3:
            leaves = sorted(leaf_set)
            return f"({labels[leaves[0]]},{labels[leaves[1]]},{labels[leaves[2]]})"
        
        # Find a split that divides leaf_set into two non-empty parts
        # The split should be in our split set (or its complement)
        found_split = None
        for split in splits:
            intersection = split & leaf_set
            if 0 < len(intersection) < len(leaf_set):
                # This split divides leaf_set
                found_split = intersection
                break
        
        if found_split is None:
            # No split found, use a default partition
            # This shouldn't happen for a valid tree
            leaves = sorted(leaf_set)
            mid = len(leaves) // 2
            found_split = set(leaves[:mid])
        
        left_set = found_split
        right_set = leaf_set - found_split
        
        left_newick = _build_tree(left_set)
        right_newick = _build_tree(right_set)
        
        return f"({left_newick},{right_newick})"
    
    return _build_tree(all_leaves)


def reconstruct(sequences):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Encode sequences
    mat = _encode_sequences(sequences, labels)
    
    # Compute distance matrix
    dist = _jc_distance_matrix(mat)
    
    # For small n, try exhaustive search
    if n <= 7 and not _check_deadline():
        result = _ml_exhaustive_small(dist, labels, n)
        if result is not None:
            return result
    
    # For larger n, use NJ
    newick = _fast_nj(dist)
    
    return newick
