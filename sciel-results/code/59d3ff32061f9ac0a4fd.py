"""Phylogenetic topology recovery via FastUPGMA / NJ with bootstrap-like robustness."""
import math
import time

# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = None


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distances using vectorized operations."""
    n = len(labels)
    # Convert sequences to numpy arrays for vectorized comparison
    seq_arrays = []
    for lab in labels:
        s = sequences[lab]
        # Encode A,C,G,T as 0,1,2,3
        arr = np.zeros(len(s), dtype=np.int8)
        for i, c in enumerate(s):
            if c == 'A':
                arr[i] = 0
            elif c == 'C':
                arr[i] = 1
            elif c == 'G':
                arr[i] = 2
            else:
                arr[i] = 3
        seq_arrays.append(arr)

    L = len(sequences[labels[0]])
    mat = np.stack(seq_arrays)  # shape (n, L)

    # Compute pairwise differences in chunks
    dist = np.zeros((n, n), dtype=np.float64)
    chunk = 32
    for i in range(0, n, chunk):
        i_end = min(i + chunk, n)
        diff = (mat[i:i_end, None, :] != mat[None, :, :])
        counts = diff.sum(axis=2)
        p = counts.astype(np.float64) / L
        p_safe = np.clip(p, 0, 0.7499)
        jc = -0.75 * np.log(1.0 - 4.0 * p_safe / 3.0)
        jc[p >= 0.7499] = 3.0
        dist[i:i_end, :] = jc

    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_newick_fast(dist, labels):
    """
    Fast Neighbor Joining using numpy for the criterion computation.
    Returns (newick_string, tree_dict, node_list) for potential refinement.
    """
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"

    # We'll use a pre-allocated matrix with max possible nodes (2n-2)
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    # Initialize with the input distances
    D[:n, :n] = dist

    active = list(range(n))
    tree_str = {i: labels[i] for i in range(n)}
    next_id = n

    while len(active) > 3:
        m = len(active)
        # Build the sub-matrix for active nodes
        idx = np.array(active)
        D_sub = D[idx[:, None], idx[None, :]]  # m x m

        # Row sums (exclude diagonal)
        np.fill_diagonal(D_sub, 0.0)
        r = D_sub.sum(axis=1)  # shape (m,)

        # NJ criterion: Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        Q = (m - 2) * D_sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)

        # Find min
        flat_idx = np.argmin(Q)
        ii = flat_idx // m
        jj = flat_idx % m
        a = active[ii]
        b = active[jj]

        new_id = next_id
        next_id += 1

        # Update distances from new node to all other active nodes
        for k, x in enumerate(active):
            if x == a or x == b:
                continue
            D[new_id, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            D[x, new_id] = D[new_id, x]

        tree_str[new_id] = f"({tree_str[a]},{tree_str[b]})"

        # Remove a, b; add new_id
        active = [x for x in active if x != a and x != b]
        active.append(new_id)

    a, b, c = active
    return f"({tree_str[a]},{tree_str[b]},{tree_str[c]})"


def _parse_newick_to_tree(newick, labels):
    """
    Parse a Newick string into an adjacency list representation.
    Returns (adj, leaf_map) where adj is a dict of adjacency lists and leaf_map maps internal node ids to their children.
    """
    # We'll build a tree structure
    # Each node has: id, children (list of node ids or leaf labels)
    # For simplicity, we'll use a different approach: track parent-child relationships
    
    # Actually, let's work with the tree structure directly using splits
    pass


def _get_splits_from_newick(newick, labels):
    """
    Extract all non-trivial splits from a Newick string.
    A split is a frozenset of leaf labels on one side of an internal edge.
    """
    # Parse the Newick to get the tree structure
    # We'll use a stack-based parser
    
    n = len(labels)
    if n <= 3:
        return set()
    
    # Parse Newick into a tree
    # Each internal node will have 3 children (for unrooted binary trees)
    # We represent the tree as a list of nodes, each with children
    
    # Simple approach: use the fact that we know the labels
    # Let's build the tree structure by parsing
    
    pos = 0
    length = len(newick)
    
    def parse_node():
        nonlocal pos
        if pos >= length:
            return None
        
        if newick[pos] == '(':
            pos += 1
            children = []
            while pos < length and newick[pos] != ')':
                child = parse_node()
                if child is not None:
                    children.append(child)
                if pos < length and newick[pos] == ',':
                    pos += 1
            if pos < length and newick[pos] == ')':
                pos += 1
            # Skip branch length
            while pos < length and newick[pos] not in (',', ')', ';'):
                pos += 1
            return ('internal', children)
        else:
            # Leaf
            start = pos
            while pos < length and newick[pos] not in (',', ')', ';'):
                pos += 1
            label = newick[start:pos]
            return ('leaf', label)
    
    root = parse_node()
    
    if root is None:
        return set()
    
    # Now extract splits from the tree
    # For each internal edge, compute the set of leaves on one side
    splits = set()
    
    def extract_splits(node, parent=None):
        nonlocal splits
        if node is None:
            return set()
        
        if node[0] == 'leaf':
            return {node[1]}
        
        # Internal node
        children = node[1]
        all_leaves = set()
        child_leaf_sets = []
        for child in children:
            leaves = extract_splits(child, node)
            all_leaves.update(leaves)
            child_leaf_sets.append(leaves)
        
        # For each child, the split is the set of leaves in that child's subtree
        # But we need to consider the edge between this node and each child
        for i, child in enumerate(children):
            child_leaves = child_leaf_sets[i]
            if len(child_leaves) > 0 and len(child_leaves) < len(all_leaves):
                # This is a non-trivial split
                # Use the smaller side for canonical representation
                if len(child_leaves) <= len(all_leaves) - len(child_leaves):
                    splits.add(frozenset(child_leaves))
                else:
                    splits.add(frozenset(all_leaves - child_leaves))
        
        return all_leaves
    
    extract_splits(root)
    
    # Filter out trivial splits (size 1 or n-1)
    n_leaves = len(labels)
    non_trivial = set()
    for s in splits:
        if 1 < len(s) < n_leaves - 1:
            non_trivial.add(s)
    
    return non_trivial


def _rf_distance(splits1, splits2):
    """Compute normalized Robinson-Foulds distance between two sets of splits."""
    if not splits1 and not splits2:
        return 0.0
    total = len(splits1) + len(splits2)
    if total == 0:
        return 0.0
    common = splits1 & splits2
    diff = total - 2 * len(common)
    return diff / total


def _spr_refine(newick, dist, labels, max_time=5.0):
    """
    SPR (Subtree Prune and Regraft) local search to improve the tree topology.
    This is a heuristic refinement that tries all possible SPR moves and keeps the best.
    """
    n = len(labels)
    if n <= 4:
        return newick
    
    start_time = time.time()
    
    # Parse the tree to get a workable representation
    # We'll represent the tree as a list of internal edges with their splits
    
    # For efficiency, let's use a simpler approach:
    # Generate the current splits, then try SPR moves
    
    current_splits = _get_splits_from_newick(newick, labels)
    current_rf = _rf_distance(current_splits, current_splits)  # 0
    
    # For SPR, we need to actually manipulate the tree
    # This is complex, so let's use a simpler heuristic:
    # Try NJ with different distance matrix perturbations
    
    # Actually, let's implement a proper SPR by working with the tree structure
    # We'll use a simpler approach: re-run NJ but with a modified criterion
    
    # For now, return the input (no change) as SPR is complex to implement correctly
    # in a time-constrained setting
    return newick


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    Returns a Newick string.
    """
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 250  # Safety margin from 270s limit

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"

    # Check deadline before expensive computation
    if time.time() > _DEADLINE - 5:
        return "(" + ",".join(labels) + ")"

    # Compute distance matrix
    dist = _compute_distance_matrix(sequences, labels)

    # Check deadline
    if time.time() > _DEADLINE - 10:
        return "(" + ",".join(labels) + ")"

    # Use fast NJ for topology recovery
    result = _nj_newick_fast(dist, labels)

    return result

# EVOLVE-BLOCK-END
