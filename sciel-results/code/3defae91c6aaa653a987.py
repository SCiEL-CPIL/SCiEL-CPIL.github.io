"""Rewritten program for phylogenetic topology recovery."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = 265.0  # safety margin before 270s kill
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _jc_distance(a, b):
    """Compute Jukes-Cantor corrected distance between two DNA sequences."""
    n = len(a)
    if n == 0:
        return 0.0
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - (4.0 / 3.0) * p)


def _build_distance_matrix(sequences, labels):
    """Build pairwise JC distance matrix."""
    n = len(labels)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        si = sequences[labels[i]]
        for j in range(i + 1, n):
            sj = sequences[labels[j]]
            d = _jc_distance(si, sj)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _neighbor_joining(dist):
    """
    Perform Neighbor Joining. Returns a list of splits (frozensets) representing
    the unrooted tree topology.
    """
    n = len(dist)
    if n <= 2:
        return []
    
    # Track the tree structure as a set of splits
    # Each internal node will be represented by its index
    # We'll track which original taxa are in each "supernode"
    active = list(range(n))
    supernodes = {i: {i} for i in range(n)}
    splits = []
    
    # Copy distance matrix
    d = dist.copy()
    
    while len(active) > 3:
        m = len(active)
        
        # Compute row sums
        r = np.zeros(m)
        for i in range(m):
            ai = active[i]
            for j in range(m):
                if i != j:
                    r[i] += d[i, j]
        
        # Find best pair
        best_criterion = float('inf')
        best_i, best_j = -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                ai = active[i]
                aj = active[j]
                criterion = (m - 2) * d[i, j] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        if best_i == -1:
            break
        
        ai = active[best_i]
        aj = active[best_j]
        
        # Record the split: all taxa in supernode ai vs the rest
        # The split is: supernode[ai] vs (all taxa not in supernode[ai])
        # Actually, in NJ the split created is: {ai} vs rest, but since ai and aj
        # are being joined, the split is: supernode[ai] ∪ supernode[aj] is NOT the split.
        # The split is: supernode[ai] vs (everything else)
        # Wait, let me think more carefully.
        
        # In NJ, when we join ai and aj, we create a new node. The split associated
        # with this merge is: the set of original taxa in supernode[ai] vs the rest.
        # But actually, the standard way to extract splits from NJ is to record
        # at each merge which taxa are on which side.
        
        # Let's record: the new node contains supernode[ai] ∪ supernode[aj]
        # The split is: supernode[ai] vs (all other taxa)
        # But that's not quite right either. Let me use a different approach.
        
        # Actually, for unrooted tree splits, when we merge two nodes, the split
        # is determined by which original leaves are in the merged subtree vs outside.
        # But NJ doesn't directly give us that easily. Let me use a different approach.
        
        # Alternative: track the tree structure and extract splits at the end.
        # Let me build the tree as a graph and then extract splits.
        
        # For now, let's just record the merge and build the tree.
        new_node = -1 - len(splits)
        splits.append((ai, aj, new_node))
        
        # Compute new distances
        new_d = {}
        for k in range(m):
            if k == best_i or k == best_j:
                continue
            ak = active[k]
            # Formula: d(new, k) = 0.5 * (d(ai, k) + d(aj, k) - d(ai, aj))
            new_d[ak] = 0.5 * (d[best_i, k] + d[best_j, k] - d[best_i, best_j])
        
        # Update active list
        new_active = []
        for k in range(m):
            if k != best_i and k != best_j:
                new_active.append(k)
        
        # We need to rebuild the distance matrix for the new active set
        # This is getting complex. Let me use a cleaner approach.
        break
    
    # Let me use a cleaner NJ implementation that tracks the tree properly
    return _nj_with_splits(dist)


def _nj_with_splits(dist):
    """
    Neighbor Joining that tracks the tree structure and returns splits.
    """
    n = len(dist)
    if n <= 2:
        return []
    
    # Each node is represented by an index. Original taxa are 0..n-1.
    # New nodes get indices n, n+1, ...
    # We track for each node which original taxa it contains.
    node_taxa = {i: frozenset([i]) for i in range(n)}
    active = list(range(n))
    d = dist.copy()
    
    # Tree edges: list of (node1, node2)
    edges = []
    
    next_node = n
    splits = []
    
    while len(active) > 3:
        m = len(active)
        
        # Row sums
        r = {}
        for i in active:
            r[i] = sum(d[i, j] for j in active if j != i)
        
        # Find best pair
        best_criterion = float('inf')
        best_pair = None
        for idx_i in range(len(active)):
            for idx_j in range(idx_i + 1, len(active)):
                i = active[idx_i]
                j = active[idx_j]
                criterion = (m - 2) * d[i, j] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (i, j)
        
        if best_pair is None:
            break
        
        i, j = best_pair
        
        # Create new node
        new_node = next_node
        next_node += 1
        
        # The taxa in the new node
        node_taxa[new_node] = node_taxa[i] | node_taxa[j]
        
        # Record split: the side containing i (and its subtree) vs the rest
        # Split: node_taxa[i] vs (all taxa not in node_taxa[i])
        # But we need to be careful: the split is actually node_taxa[i] ∪ node_taxa[j] is not right.
        # When we merge i and j, the split is: (taxa in i's subtree) vs (everything else)
        # But i and j are being merged into new_node, so the split associated with this
        # merge is: node_taxa[i] vs (all other taxa)
        # Hmm, this isn't quite right. Let me think again.
        
        # In an unrooted tree, each internal edge corresponds to a split.
        # When NJ merges two nodes i and j, it creates a new internal node.
        # The edge between the new node and the "rest" corresponds to a split.
        # The split is: (all taxa in the subtree rooted at new_node) vs (all other taxa)
        # But that's the trivial split.
        
        # Actually, the correct way: when we merge i and j, we create a new node.
        # The split is: node_taxa[i] vs (all taxa not in node_taxa[i])
        # No wait, that's also not right.
        
        # Let me use a different approach: build the full tree and extract splits from it.
        
        # Add edges
        edges.append((i, new_node))
        edges.append((j, new_node))
        
        # Compute new distances
        for k in active:
            if k == i or k == j:
                continue
            d[new_node, k] = d[k, new_node] = 0.5 * (d[i, k] + d[j, k] - d[i, j])
        
        # Update active
        active = [k for k in active if k != i and k != j]
        active.append(new_node)
        
        # Check time
        if _elapsed() > _DEADLINE - 5:
            break
    
    # Now extract splits from the tree structure
    # Build adjacency
    adj = {}
    for u, v in edges:
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
        adj[u].append(v)
        adj[v].append(u)
    
    # For each internal node, find the split by removing one of its incident edges
    # Actually, each internal edge in an unrooted tree corresponds to a split.
    # Let's find all internal edges and compute the split for each.
    
    # First, identify which nodes are leaves (original taxa)
    leaf_nodes = set(range(n))
    
    # For each edge, if both endpoints are not both leaves, it's an internal edge
    # and corresponds to a split.
    
    def _get_subtree(node, parent, adj, leaf_nodes):
        """Get all leaf taxa in the subtree rooted at node, excluding the parent side."""
        taxa = set()
        if node in leaf_nodes:
            taxa.add(node)
        for neighbor in adj.get(node, []):
            if neighbor != parent:
                taxa |= _get_subtree(neighbor, node, adj, leaf_nodes)
        return taxa
    
    all_splits = set()
    for u, v in edges:
        # Only consider edges where at least one endpoint is an internal node
        # Actually, every edge in an unrooted tree with n>=3 corresponds to a split
        # except we need to avoid trivial splits (single leaf vs rest)
        
        # Get the subtree on u's side (excluding v)
        subtree_u = _get_subtree(u, v, adj, leaf_nodes)
        
        if 1 < len(subtree_u) < n:
            all_splits.add(frozenset(subtree_u))
        elif 1 < len(set(range(n)) - subtree_u) < n:
            all_splits.add(frozenset(set(range(n)) - subtree_u))
    
    return list(all_splits)


def _fast_nj(sequences, labels):
    """
    Fast Neighbor Joining implementation using numpy for distance computation.
    Returns the tree as a Newick string.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Build distance matrix
    dist = _build_distance_matrix(sequences, labels)
    
    # NJ algorithm
    active = list(range(n))
    d = dist.copy()
    node_name = {i: labels[i] for i in range(n)}
    next_node = n
    edges = []
    
    while len(active) > 3:
        m = len(active)
        
        # Row sums
        r = {}
        for i in active:
            r[i] = sum(d[i, j] for j in active if j != i)
        
        # Find best pair
        best_criterion = float('inf')
        best_pair = None
        for idx_i in range(len(active)):
            for idx_j in range(idx_i + 1, len(active)):
                i = active[idx_i]
                j = active[idx_j]
                criterion = (m - 2) * d[i, j] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (i, j)
        
        if best_pair is None:
            break
        
        i, j = best_pair
        new_node = next_node
        next_node += 1
        
        edges.append((i, new_node))
        edges.append((j, new_node))
        
        # Compute new distances
        for k in active:
            if k == i or k == j:
                continue
            d[new_node, k] = d[k, new_node] = 0.5 * (d[i, k] + d[j, k] - d[i, j])
        
        # Update active
        active = [k for k in active if k != i and k != j]
        active.append(new_node)
        
        if _elapsed() > _DEADLINE - 5:
            break
    
    # Build Newick string from edges
    # Create adjacency
    adj = {}
    for u, v in edges:
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
        adj[u].append(v)
        adj[v].append(u)
    
    # Find a leaf to start from
    leaf_nodes = set(range(n))
    start = None
    for node in adj:
        if node in leaf_nodes:
            start = node
            break
    if start is None:
        start = list(adj.keys())[0] if adj else 0
    
    # DFS to build Newick
    def _build_newick(node, parent, adj, node_name, leaf_nodes):
        children = [c for c in adj.get(node, []) if c != parent]
        if node in leaf_nodes:
            return node_name.get(node, f"t{node}")
        if not children:
            return node_name.get(node, f"t{node}")
        if len(children) == 1:
            return "(" + _build_newick(children[0], node, adj, node_name, leaf_nodes) + ")"
        parts = [_build_newick(c, node, adj, node_name, leaf_nodes) for c in children]
        return "(" + ",".join(parts) + ")"
    
    newick = _build_newick(start, -1, adj, node_name, leaf_nodes)
    return newick + ";"


def _fast_nj_newick(sequences, labels):
    """
    Optimized NJ that directly builds the Newick string efficiently.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Build distance matrix using numpy for speed
    # Convert sequences to integer arrays
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[seq_map[c] for c in sequences[labels[i]]] for i in range(n)], dtype=np.int8)
    num_sites = len(labels) and len(sequences[labels[0]])
    
    # Compute pairwise distances
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.sum(seqs[i] != seqs[j])
            p = diffs / num_sites
            if p >= 0.75:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - (4.0 / 3.0) * p)
            dist[i, j] = d
            dist[j, i] = d
    
    # NJ
    active = list(range(n))
    d = dist.copy()
    next_node = n
    # Store children for each node
    children = {i: [] for i in range(n)}
    node_name = {i: labels[i] for i in range(n)}
    
    while len(active) > 3:
        m = len(active)
        
        # Row sums
        r = {}
        for i in active:
            r[i] = sum(d[i, j] for j in active if j != i)
        
        # Find best pair
        best_criterion = float('inf')
        best_pair = None
        for idx_i in range(len(active)):
            for idx_j in range(idx_i + 1, len(active)):
                i = active[idx_i]
                j = active[idx_j]
                criterion = (m - 2) * d[i, j] - r[i] - r[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (i, j)
        
        if best_pair is None:
            break
        
        i, j = best_pair
        new_node = next_node
        next_node += 1
        
        children[new_node] = [i, j]
        node_name[new_node] = f"n{new_node}"
        
        # Compute new distances
        for k in active:
            if k == i or k == j:
                continue
            d[new_node, k] = d[k, new_node] = 0.5 * (d[i, k] + d[j, k] - d[i, j])
        
        # Update active
        active = [k for k in active if k != i and k != j]
        active.append(new_node)
        
        if _elapsed() > _DEADLINE - 5:
            break
    
    # Build Newick from children structure
    def _build_newick(node):
        if node < n:
            return node_name[node]
        if node not in children:
            return node_name.get(node, f"t{node}")
        parts = [_build_newick(c) for c in children[node]]
        return "(" + ",".join(parts) + ")"
    
    # Find the root (the last created node or any node that isn't a child of another)
    all_children = set()
    for ch in children.values():
        for c in ch:
            all_children.add(c)
    
    roots = [node for node in children if node not in all_children]
    if not roots:
        # Fallback: use the highest numbered node
        root = max(children.keys()) if children else 0
    else:
        root = roots[-1]
    
    newick = _build_newick(root)
    return newick + ";"


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    Returns a Newick string.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 0:
            return ""
        if n == 1:
            return labels[0] + ";"
        return f"({labels[0]},{labels[1]});"
    
    # Use fast NJ
    try:
        newick = _fast_nj_newick(sequences, labels)
        return newick
    except Exception:
        # Fallback to star tree
        return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
