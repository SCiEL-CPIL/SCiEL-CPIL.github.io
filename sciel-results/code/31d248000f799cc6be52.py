import math
import time
import numpy as np
from itertools import combinations

def _jc_distance_matrix(seqs):
    """Compute JC-corrected distance matrix from list of numpy char arrays."""
    n = len(seqs)
    L = len(seqs[0])
    # Encode sequences as 0,1,2,3 for A,C,G,T
    # 0: A, 1: C, 2: G, 3: T
    # We want to count mismatches.
    # Vectorized mismatch count:
    # seqs is list of np arrays of chars.
    # Convert to int arrays for faster comparison?
    # Actually, np arrays of chars are fine.
    
    D = np.zeros((n, n))
    # Pairwise comparison
    # For n=32, L=2000, this is 32*32*2000 = 2M ops, very fast.
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.sum(seqs[i] != seqs[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
    return D

def _nj_build_tree(D, labels):
    """
    Build initial tree using Neighbor Joining.
    Returns a list of edges (u, v) and a mapping from node id to label or children.
    Leaves are 0..n-1. Internal nodes are n..2n-3.
    """
    n = len(labels)
    if n <= 3:
        # Handle small n separately
        if n == 1:
            return []
        elif n == 2:
            return [(0, 1)]
        else:
            # Star tree for 3 taxa? Or binary?
            # NJ requires binary. For 3, it's a single internal node with 3 leaves?
            # Unrooted binary tree for 3 leaves is not possible (needs 2 internal nodes? No, 3 leaves = 1 internal node if rooted, but unrooted 3-leaf tree is a star).
            # Standard NJ produces a binary tree. For n=3, it's a star?
            # Actually, for n=3, the tree has 1 internal node connected to 3 leaves.
            # But our NNI logic assumes binary internal nodes (degree 3).
            # Let's return a star for n=3.
            return [(0, 3), (1, 3), (2, 3)]
            
    # Standard NJ
    # D is n x n
    # nodes: list of current node ids
    # tree_edges: list of (u, v)
    # We will track the current set of nodes and their "effective" distances.
    
    D = D.copy()
    nodes = list(range(n))
    edges = []
    
    next_node_id = n
    
    while len(nodes) > 2:
        k = len(nodes)
        # Calculate row sums
        row_sums = D.sum(axis=1)
        # Q matrix
        # Q[i, j] = (k - 2) * D[i, j] - row_sums[i] - row_sums[j]
        Q = (k - 2) * D - row_sums[:, None] - row_sums[None, :]
        
        # Find min Q
        min_val = np.inf
        min_i, min_j = 0, 0
        for i in range(k):
            for j in range(i + 1, k):
                if Q[i, j] < min_val:
                    min_val = Q[i, j]
                    min_i, min_j = i, j
                    
        a_idx, b_idx = min_i, min_j
        a_node, b_node = nodes[a_idx], nodes[b_idx]
        
        new_node = next_node_id
        next_node_id += 1
        
        edges.append((a_node, new_node))
        edges.append((b_node, new_node))
        
        # Update distances
        # New distance from new_node to any other node m:
        # d(new, m) = 0.5 * (d(a, m) + d(b, m) - d(a, b))
        
        keep_idxs = [idx for idx in range(k) if idx not in (a_idx, b_idx)]
        new_k = k - 1
        
        new_D = np.zeros((new_k + 1, new_k + 1)) # +1 for new node
        
        # Copy distances between kept nodes
        for i, ki in enumerate(keep_idxs):
            for j, kj in enumerate(keep_idxs):
                new_D[i, j] = D[ki, kj]
                
        # Calculate distances to new node
        for i, ki in enumerate(keep_idxs):
            dist = 0.5 * (D[ki, a_idx] + D[ki, b_idx] - D[a_idx, b_idx])
            new_D[i, new_k] = dist
            new_D[new_k, i] = dist
            
        # New node distance to itself is 0
        new_D[new_k, new_k] = 0
        
        D = new_D
        nodes = [nodes[k] for k in keep_idxs] + [new_node]
        
    # Final 2 nodes
    if len(nodes) == 2:
        edges.append((nodes[0], nodes[1]))
        
    return edges

def _get_subtree_leaves(edges, node, n):
    """
    Get the set of leaf indices (0..n-1) in the subtree rooted at 'node', 
    assuming the tree is rooted at 'node' and we ignore the parent edge.
    This is a BFS/DFS.
    """
    # Build adjacency list
    adj = {i: [] for i in range(2 * n - 2)} # Max nodes
    for u, v in edges:
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS
    visited = {node}
    queue = [node]
    leaves = set()
    
    while queue:
        curr = queue.pop(0)
        if curr < n:
            leaves.add(curr)
        for neighbor in adj.get(curr, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                
    return leaves

def _nni_optimize(D, edges, labels, n, deadline):
    """
    Perform NNI search to optimize topology.
    """
    start_time = time.time()
    
    # Build adjacency and leaf sets for all nodes
    def update_leaf_sets(edges):
        adj = {i: [] for i in range(2 * n - 2)}
        for u, v in edges:
            if u not in adj: adj[u] = []
            if v not in adj: adj[v] = []
            adj[u].append(v)
            adj[v].append(u)
            
        leaf_sets = {}
        for node in adj:
            if node < n:
                leaf_sets[node] = {node}
            else:
                # BFS
                visited = {node}
                queue = [node]
                leaves = set()
                while queue:
                    curr = queue.pop(0)
                    if curr < n:
                        leaves.add(curr)
                    for neighbor in adj.get(curr, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                leaf_sets[node] = leaves
        return adj, leaf_sets

    adj, leaf_sets = update_leaf_sets(edges)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate over all internal edges
        # An internal edge is one where both endpoints are >= n
        # Or more generally, any edge that is not a leaf edge.
        # In a binary tree, internal nodes have degree 3.
        # We can iterate over all edges and check if they are internal.
        
        edges_to_check = list(edges)
        for idx, (u, v) in enumerate(edges_to_check):
            if time.time() > deadline:
                break
                
            # Check if this is an internal edge
            # In a binary tree with n leaves, there are n-3 internal edges.
            # An edge is internal if both u and v are internal nodes (>= n) OR
            # if it's the only edge connecting two parts of the tree that both have >1 leaves.
            # Simpler: u and v are both internal nodes?
            # Not necessarily. The edge between a leaf and an internal node is not NNI-able.
            # NNI applies to internal edges.
            # In a binary tree, an edge is internal if it is not incident to a leaf.
            if u < n or v < n:
                continue
                
            # Get neighbors of u and v
            # u is connected to v, and two other nodes (a, b)
            # v is connected to u, and two other nodes (c, d)
            
            neighbors_u = [x for x in adj[u] if x != v]
            neighbors_v = [x for x in adj[v] if x != u]
            
            if len(neighbors_u) != 2 or len(neighbors_v) != 2:
                continue # Not a valid internal edge in a binary tree?
                
            a, b = neighbors_u
            c, d = neighbors_v
            
            # Current topology: (A, B) - (C, D)
            # A = subtree at a, B = subtree at b, C = subtree at c, D = subtree at d
            
            # Pick representatives
            # We need to pick one leaf from each subtree to compute quartet distances.
            # To be robust, we could pick the leaf that minimizes/maximizes distance?
            # Standard NNI uses the "center" or just any representative.
            # Using the centroid or just the first leaf is common.
            # Let's pick the leaf with the smallest distance to the node?
            # Or just the first one in the set.
            
            rep_a = min(leaf_sets[a]) if leaf_sets[a] else a
            rep_b = min(leaf_sets[b]) if leaf_sets[b] else b
            rep_c = min(leaf_sets[c]) if leaf_sets[c] else c
            rep_d = min(leaf_sets[d]) if leaf_sets[d] else d
            
            # Compute quartet scores
            # d_ab + d_cd
            s1 = D[rep_a, rep_b] + D[rep_c, rep_d]
            # d_ac + d_bd
            s2 = D[rep_a, rep_c] + D[rep_b, rep_d]
            # d_ad + d_bc
            s3 = D[rep_a, rep_d] + D[rep_b, rep_c]
            
            min_score = min(s1, s2, s3)
            
            # Current topology corresponds to s1 (A,B vs C,D)
            # If s1 is not the minimum, we should swap.
            
            if min_score != s1:
                # Determine which swap to perform
                # If s2 is min: Swap B and C?
                # Topology: (A, C) - (B, D)
                # Edges to remove: (u, b), (v, c)
                # Edges to add: (u, c), (v, b)
                
                if min_score == s2:
                    # Swap b and c
                    new_edges = []
                    for e in edges:
                        if e == (u, b) or e == (b, u):
                            continue
                        if e == (v, c) or e == (c, v):
                            continue
                        new_edges.append(e)
                    new_edges.append((u, c))
                    new_edges.append((v, b))
                    
                elif min_score == s3:
                    # Swap b and d
                    # Topology: (A, D) - (B, C)
                    # Edges to remove: (u, b), (v, d)
                    # Edges to add: (u, d), (v, b)
                    
                    new_edges = []
                    for e in edges:
                        if e == (u, b) or e == (b, u):
                            continue
                        if e == (v, d) or e == (d, v):
                            continue
                        new_edges.append(e)
                    new_edges.append((u, d))
                    new_edges.append((v, b))
                    
                else:
                    continue # Should not happen
                    
                edges = new_edges
                adj, leaf_sets = update_leaf_sets(edges)
                improved = True
                
    return edges

def _edges_to_newick(edges, labels, n):
    """Convert edge list to Newick string."""
    if n == 1:
        return labels[0] + ";"
        
    # Build adjacency
    adj = {i: [] for i in range(2 * n - 2)}
    for u, v in edges:
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append(v)
        adj[v].append(u)
        
    # Find a leaf to start rooting
    # Pick leaf 0
    root = 0
    # We need to traverse from root.
    # To make it unrooted, we can pick any edge as the "root edge" or just traverse.
    # Standard Newick is rooted. We can root at leaf 0.
    
    def to_newick(node, parent):
        children = [x for x in adj[node] if x != parent]
        if not children:
            # Leaf
            return labels[node]
        elif len(children) == 1:
            # This shouldn't happen in a binary tree unless it's a path, 
            # but in unrooted binary tree, internal nodes have 3 neighbors.
            # If we root at a leaf, the root has 1 child.
            # The next node might have 2 children (if it's an internal node) or 1 (if it's a leaf? no).
            # Actually, if we root at a leaf, the tree becomes rooted.
            # Internal nodes will have 2 children (one parent, one child? No, 3 neighbors total).
            # So in the rooted view, internal nodes have 2 children.
            
            child_str = to_newick(children[0], node)
            return f"({child_str})"
        else:
            child_strs = [to_newick(c, node) for c in children]
            return "(" + ",".join(child_strs) + ")"
            
    newick = to_newick(root, -1)
    return newick + ";"

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
        
    # Convert sequences to numpy arrays
    seqs = [np.array(list(sequences[label]), dtype='U1') for label in labels]
    
    # Calculate distance matrix
    D = _jc_distance_matrix(seqs)
    
    # Build initial tree with NJ
    edges = _nj_build_tree(D, labels)
    
    # Optimize with NNI
    if time.time() < deadline - 10.0: # Leave 10s for output
        edges = _nni_optimize(D, edges, labels, n, deadline - 5.0)
        
    # Convert to Newick
    newick = _edges_to_newick(edges, labels, n)
    return newick

