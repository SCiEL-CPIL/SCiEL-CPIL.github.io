"""Optimized Neighbor-Joining with vectorized distance computation and NNI refinement."""
import numpy as np
import math


def _compute_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected pairwise distance matrix using NumPy vectorization."""
    n = len(labels)
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    if n == 0:
        return np.zeros((0, 0))
    if n == 1:
        return np.zeros((1, 1))

    # Convert to numpy array of characters for vectorized comparison
    # Using 'S1' dtype allows element-wise comparison
    mat = np.array([list(s) for s in seq_list], dtype='S1')
    
    # Compute pairwise differences
    # Broadcasting: (n, 1, L) vs (1, n, L) -> (n, n, L)
    diff = (mat[:, None, :] != mat[None, :, :]).sum(axis=2).astype(np.float64)
    
    # Compute p = diff / length
    p = diff / length
    
    # Jukes-Cantor correction: d = -0.75 * ln(1 - 4p/3)
    # Handle p >= 0.75 (saturation) by capping
    np.clip(p, 0, 0.7499, out=p)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(d, 0.0)
    
    return d


def _neighbor_joining(d):
    """
    Neighbor-Joining algorithm on a distance matrix d of shape (n, n).
    Returns a list of edges (u, v, length) and the number of leaves n.
    """
    n = d.shape[0]
    if n == 1:
        return []
    if n == 2:
        return [(0, 1, d[0, 1])]
    
    D = d.copy()
    edges = []
    active = list(range(n))
    next_node_id = n
    
    while len(active) > 2:
        m = len(active)
        
        # Compute row sums for active nodes
        # Extract submatrix
        D_sub = D[np.ix_(active, active)]
        r = D_sub.sum(axis=1)
        
        # Compute Q matrix
        # Q[i, j] = (m - 2) * D[i, j] - r[i] - r[j]
        Q = (m - 2) * D_sub - r[:, None] - r[None, :]
        
        # Find min Q in upper triangle
        # Mask lower triangle and diagonal
        mask = np.triu(np.ones((m, m), dtype=bool), k=1)
        Q_masked = np.where(mask, Q, np.inf)
        
        min_idx = np.unravel_index(np.argmin(Q_masked), Q_masked.shape)
        i_idx, j_idx = min_idx
        
        a = active[i_idx]
        b = active[j_idx]
        
        # Compute branch lengths
        delta = (r[i_idx] - r[j_idx]) / (2.0 * (m - 2))
        len_a = 0.5 * D[a, b] + 0.5 * delta
        len_b = 0.5 * D[a, b] - 0.5 * delta
        
        if len_a < 0: len_a = 0.0
        if len_b < 0: len_b = 0.0
        
        # Add edges
        edges.append((a, next_node_id, len_a))
        edges.append((b, next_node_id, len_b))
        
        # Update active list
        active.remove(a)
        active.remove(b)
        active.append(next_node_id)
        
        # Update distances for new node
        new_id = next_node_id
        D = np.vstack([D, np.zeros(D.shape[0] + 1)])
        D = np.hstack([D, np.zeros((D.shape[0], 1))])
        
        for x in active:
            if x == new_id: continue
            val = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            if val < 0: val = 0.0
            D[new_id, x] = val
            D[x, new_id] = val
            
        next_node_id += 1
        
    # Connect last two
    if len(active) == 2:
        i, j = active
        dist = D[i, j]
        if dist < 0: dist = 0.0
        edges.append((i, j, dist))
        
    return edges


def _build_adjacency(edges, n_nodes):
    """Build adjacency list from edges."""
    adj = {i: [] for i in range(n_nodes)}
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def _tree_distances(adj, n_leaves, n_nodes):
    """Compute pairwise distance matrix for a tree."""
    dist = np.zeros((n_leaves, n_leaves))
    for i in range(n_leaves):
        # BFS/DFS from i
        visited = {i: 0.0}
        stack = [i]
        while stack:
            u = stack.pop()
            for v, w in adj[u]:
                if v not in visited:
                    visited[v] = visited[u] + w
                    stack.append(v)
        for j in range(n_leaves):
            dist[i, j] = visited[j]
    return dist


def _sse(D_obs, D_tree):
    """Sum of Squared Errors."""
    return np.sum((D_obs - D_tree) ** 2)


def _nni_refine(edges, D_obs, n_leaves, max_iters=10):
    """Refine tree topology using NNI moves to minimize SSE."""
    n_nodes = 2 * n_leaves - 2
    if n_leaves < 4:
        return edges
    
    current_edges = edges[:]
    best_sse = _sse(D_obs, _tree_distances(_build_adjacency(current_edges, n_nodes), n_leaves, n_nodes))
    
    for _ in range(max_iters):
        improved = False
        adj = _build_adjacency(current_edges, n_nodes)
        
        # Identify internal edges
        # In a binary tree, internal nodes have degree 3.
        # An edge is internal if both endpoints have degree > 1.
        internal_edges = []
        for u, v, w in current_edges:
            if len(adj[u]) > 1 and len(adj[v]) > 1:
                internal_edges.append((u, v, w))
                
        for u, v, w in internal_edges:
            # Get neighbors
            neighbors_u = [x for x, _ in adj[u] if x != v]
            neighbors_v = [x for x, _ in adj[v] if x != u]
            
            if len(neighbors_u) != 2 or len(neighbors_v) != 2:
                continue
                
            a, b = neighbors_u
            c, d = neighbors_v
            
            # Try NNI Move 1: Swap b and c
            # Remove (u,b) and (v,c)
            # Add (u,c) and (v,b)
            
            # Create a copy of edges
            new_edges = []
            for eu, ev, ew in current_edges:
                if (eu == u and ev == b) or (eu == b and ev == u):
                    continue
                if (eu == v and ev == c) or (eu == c and ev == v):
                    continue
                new_edges.append((eu, ev, ew))
            
            # Find lengths of removed edges
            len_ub = 0.0
            len_vc = 0.0
            for eu, ev, ew in current_edges:
                if (eu == u and ev == b) or (eu == b and ev == u):
                    len_ub = ew
                if (eu == v and ev == c) or (eu == c and ev == v):
                    len_vc = ew
            
            new_edges.append((u, c, len_vc)) # Use len_vc for u-c? Or len_ub? Let's use the length of the edge we're replacing in terms of "position".
            # Actually, it's better to just use the length of the edge that is being "moved".
            # The edge u-b is moved to u-c? No, the connection changes.
            # Let's just assign the length of the removed edge to the new one.
            new_edges.append((v, b, len_ub))
            
            # Evaluate
            new_adj = _build_adjacency(new_edges, n_nodes)
            new_dist = _tree_distances(new_adj, n_leaves, n_nodes)
            new_sse = _sse(D_obs, new_dist)
            
            if new_sse < best_sse - 1e-9:
                best_sse = new_sse
                current_edges = new_edges
                improved = True
                # Restart the loop or continue? Continue is fine, but adj is now stale.
                # To be safe, break and restart the pass? Or just update adj?
                # Updating adj is complex. Let's just break and restart the pass.
                break
                
            # Try NNI Move 2: Swap b and d
            new_edges2 = []
            for eu, ev, ew in current_edges:
                if (eu == u and ev == b) or (eu == b and ev == u):
                    continue
                if (eu == v and ev == d) or (eu == d and ev == v):
                    continue
                new_edges2.append((eu, ev, ew))
            
            len_vd = 0.0
            for eu, ev, ew in current_edges:
                if (eu == v and ev == d) or (eu == d and ev == v):
                    len_vd = ew
            
            new_edges2.append((u, d, len_vd))
            new_edges2.append((v, b, len_ub))
            
            new_adj2 = _build_adjacency(new_edges2, n_nodes)
            new_dist2 = _tree_distances(new_adj2, n_leaves, n_nodes)
            new_sse2 = _sse(D_obs, new_dist2)
            
            if new_sse2 < best_sse - 1e-9:
                best_sse = new_sse2
                current_edges = new_edges2
                improved = True
                break
                
        if not improved:
            break
            
    return current_edges


def _edges_to_newick(edges, n_leaves):
    """Convert edge list to Newick string."""
    if n_leaves == 1:
        return "t0;"
    if n_leaves == 2:
        return "(t0,t1);"
        
    n_nodes = 2 * n_leaves - 2
    adj = _build_adjacency(edges, n_nodes)
    
    # Root the tree at node 0
    # DFS to build Newick
    visited = set()
    
    def dfs(node, parent):
        visited.add(node)
        children = []
        for neighbor, weight in adj[node]:
            if neighbor != parent:
                child_str = dfs(neighbor, node)
                children.append(f"{child_str}:{weight:.4f}")
                
        if node < n_leaves:
            return f"t{node}"
        else:
            if len(children) == 0:
                return f"t{node}" # Should not happen for internal
            elif len(children) == 1:
                return f"({children[0]})"
            else:
                return f"({','.join(children)})"
                
    newick = dfs(0, -1)
    return newick + ";"


def reconstruct(sequences):
    """
    Reconstruct unrooted tree topology from DNA sequences.
    
    Args:
        sequences: dict mapping taxon labels (t0, t1, ...) to DNA strings
        
    Returns:
        Newick string
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Compute distance matrix
    D = _compute_distance_matrix(sequences, labels)
    
    # Run Neighbor-Joining
    edges = _neighbor_joining(D)
    
    # Refine with NNI
    edges = _nni_refine(edges, D, n)
    
    # Convert to Newick
    newick = _edges_to_newick(edges, n)
    
    return newick
