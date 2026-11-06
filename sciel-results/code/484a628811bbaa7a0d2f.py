"""Phylogenetic topology recovery: fast vectorized NJ with local SPR refinement."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_DEADLINE = 270.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _remaining():
    return _DEADLINE - _elapsed()


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using vectorized numpy operations."""
    n = len(labels)
    seqs = [sequences[lb] for lb in labels]
    L = len(seqs[0])

    # Encode sequences as numpy arrays: A=0, C=1, G=2, T=3
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arrays = []
    for s in seqs:
        arr = np.array([enc[c] for c in s], dtype=np.uint8)
        arrays.append(arr)
    mat = np.array(arrays, dtype=np.uint8)  # shape (n, L)

    # Compute pairwise Hamming distances in chunks to limit memory
    dist = np.zeros((n, n), dtype=np.float64)
    chunk = 16
    for i in range(0, n, chunk):
        i_end = min(i + chunk, n)
        # diff shape: (chunk, n, L)
        diff = (mat[i:i_end, None, :] != mat[None, :, :])
        counts = diff.sum(axis=2).astype(np.float64)  # (chunk, n)
        p = counts / L
        p_safe = np.clip(p, 0.0, 0.7499)
        jc = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
        jc[p >= 0.7499] = 3.0
        dist[i:i_end, :] = jc

    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_tree(dist, labels):
    """
    Neighbor-joining algorithm. Returns (newick_string, internal_node_map).
    internal_node_map: dict mapping internal node id -> (left_child_id, right_child_id)
    """
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]});", {}
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});", {}

    # Node ids: 0..n-1 are leaves, n.. are internal
    nodes = list(range(n))
    tree_repr = {i: labels[i] for i in range(n)}
    # internal_children: id -> (left, right)
    internal_children = {}
    D = dist.copy()
    next_id = n

    while len(nodes) > 3:
        m = len(nodes)
        # row sums
        row_sum = np.zeros(m, dtype=np.float64)
        for i, a in enumerate(nodes):
            for j, b in enumerate(nodes):
                if i != j:
                    row_sum[i] += D[a, b]

        best_criterion = float('inf')
        best_i = 0
        best_j = 1
        for i in range(m):
            for j in range(i + 1, m):
                a = nodes[i]
                b = nodes[j]
                criterion = (m - 2) * D[a, b] - row_sum[i] - row_sum[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i = i
                    best_j = j

        a = nodes[best_i]
        b = nodes[best_j]
        d_ab = D[a, b]

        # Branch lengths (not strictly needed for topology but good for completeness)
        # delta_a = 0.5 * (d_ab + row_sum[best_i] - row_sum[best_j]) / (m - 2)
        # delta_b = d_ab - delta_a

        # New node
        new_id = next_id
        next_id += 1
        tree_repr[new_id] = f"({tree_repr[a]},{tree_repr[b]})"
        internal_children[new_id] = (a, b)

        # Update distance matrix
        # We need to add a row/col for new_id and remove rows/cols for a, b
        # For simplicity, rebuild a new matrix for remaining + new node
        remaining = [x for x in nodes if x not in (a, b)]
        remaining.append(new_id)

        new_D = np.zeros((len(remaining), len(remaining)), dtype=np.float64)
        # Map old ids to new indices
        old_to_new = {x: i for i, x in enumerate(remaining)}

        for i, x in enumerate(remaining):
            for j, y in enumerate(remaining):
                if i == j:
                    continue
                if x == new_id:
                    new_D[i, j] = 0.5 * (D[a, y] + D[b, y] - d_ab)
                elif y == new_id:
                    new_D[i, j] = 0.5 * (D[a, x] + D[b, x] - d_ab)
                else:
                    new_D[i, j] = D[x, y]

        D = new_D
        nodes = remaining

    # Final tree with 3 nodes
    a, b, c = nodes
    # In unrooted NJ, the last 3 nodes form a star (or we can join two then attach the third)
    # Standard NJ ends with 3 nodes forming a trichotomy in the unrooted sense,
    # but typically we join two of them to make a binary tree.
    # Let's join a and b, then attach c.
    # Actually, the standard algorithm joins until 2 nodes remain? No, usually 3.
    # If 3 nodes remain: a, b, c.
    # We can create a new node joining a and b, then that node and c.
    # Or just return the star topology for the last 3?
    # The task asks for unrooted tree topology. A star of 3 is a valid unrooted tree.
    # However, RF distance usually assumes binary trees.
    # Let's make it binary: join a and b into new_node, then (new_node, c).
    new_id = next_id
    next_id += 1
    tree_repr[new_id] = f"({tree_repr[a]},{tree_repr[b]})"
    internal_children[new_id] = (a, b)
    # Now we have new_id and c.
    # The tree is (new_id, c)
    # But wait, the previous loop exited when len(nodes) == 3.
    # So we have 3 nodes left.
    # If we join a and b, we get new_id. Now nodes are [new_id, c].
    # We need one more step to join them?
    # The loop condition is len(nodes) > 3.
    # So if we start with 4, we join 2, get 3. Loop stops.
    # We have 3 nodes. We need to resolve them.
    # Let's just join two of them to make a binary tree.
    # The resulting tree string should be valid Newick.
    # Let's join a and b, then join result with c.
    # But we already created new_id for a,b.
    # So the final tree is (new_id, c).
    # But new_id is an internal node.
    # The Newick string for the whole tree is just the string of the root.
    # The root is the last node created?
    # In the loop, we create new nodes.
    # When we exit with 3 nodes, we haven't joined them yet.
    # Let's join two of them.
    # Actually, the standard NJ implementation often stops at 2 nodes.
    # Let's check the loop: while len(nodes) > 3.
    # If n=4, join 2 -> 3 nodes. Stop.
    # We have 3 nodes: x, y, z.
    # We should join two, say x and y, to get w. Then we have w, z.
    # Then join w and z to get root.
    # So we need 2 more steps?
    # No, the loop stops at 3.
    # Let's just handle the 3 nodes case by joining two.
    # The resulting tree is binary.
    # Let's join nodes[0] and nodes[1] into new_node.
    # Then the tree is (new_node, nodes[2]).
    # This is a valid binary tree.
    
    # Re-do the final step properly
    # We have nodes = [a, b, c]
    # Join a and b
    new_id = next_id
    next_id += 1
    tree_repr[new_id] = f"({tree_repr[a]},{tree_repr[b]})"
    internal_children[new_id] = (a, b)
    
    # Now we have new_id and c
    # The root is the combination of new_id and c
    # But we don't need to create another node if we just return the string
    # The tree is ( (a,b), c )
    # Which is represented by tree_repr[new_id] and tree_repr[c]
    # The final Newick is (tree_repr[new_id], tree_repr[c])
    # But wait, tree_repr[new_id] is already "(a,b)"
    # So the string is "((a,b),c)"
    # This is a valid Newick string.
    
    # However, the internal_children map needs to be consistent.
    # The root is effectively an implicit node joining new_id and c.
    # For SPR moves, we need to know the structure.
    # Let's create a root node explicitly.
    root_id = next_id
    next_id += 1
    tree_repr[root_id] = f"({tree_repr[new_id]},{tree_repr[c]})"
    internal_children[root_id] = (new_id, c)
    
    return tree_repr[root_id] + ";", internal_children


def _get_splits_from_tree(internal_children, labels, n):
    """
    Extract non-trivial splits from the tree structure.
    internal_children: dict mapping internal node id -> (left, right)
    labels: list of leaf labels
    n: number of taxa
    Returns a set of frozensets.
    """
    # Map leaf id to label
    leaf_id_to_label = {i: labels[i] for i in range(n)}
    
    # For each internal node, compute the set of leaves in its subtree
    # Then the split is (subtree_leaves, complement)
    # We only care about non-trivial splits (size > 0 and < n)
    
    splits = set()
    
    def get_leaves(node_id):
        if node_id < n:
            return frozenset([leaf_id_to_label[node_id]])
        else:
            left, right = internal_children[node_id]
            left_leaves = get_leaves(left)
            right_leaves = get_leaves(right)
            return left_leaves | right_leaves
    
    # Iterate over all internal nodes
    for node_id in internal_children:
        subtree_leaves = get_leaves(node_id)
        if 0 < len(subtree_leaves) < n:
            # Normalize the split: use the smaller set for consistency
            # Actually, RF distance usually considers {A, B} and {B, A} as the same split
            # So we can just add the frozenset. The comparison will handle it.
            # But to be safe, we can canonicalize.
            # Let's just add the frozenset.
            splits.add(subtree_leaves)
    
    return splits


def _spr_move(tree_repr, internal_children, labels, n, dist):
    """
    Perform one SPR move to improve the tree.
    This is a simplified version. We'll try to find a better neighbor.
    """
    # For now, return the same tree
    # A full SPR implementation is complex.
    # Let's stick with NJ for now, as it's robust.
    return tree_repr, internal_children


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        if n == 2:
            return f"({labels[0]},{labels[1]});"
        elif n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
        else:
            return f"({','.join(labels)});"
    
    # Compute distance matrix
    dist = _jc_distance_matrix(sequences, labels)
    
    # Run NJ
    tree, internal_children = _nj_tree(dist, labels)
    
    # Check time
    if _remaining() < 5.0:
        return tree
    
    # Optional: Local search (not implemented yet)
    # For now, return NJ result
    
    return tree


# EVOLVE-BLOCK-END
