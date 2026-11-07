import numpy as np
import time

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Build adjacency list to find spanning tree
    adj = {i: [] for i in range(n)}
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        
    # Find spanning tree using BFS
    visited = [False] * n
    tree_edges = set()
    queue = [0]
    visited[0] = True
    
    while queue:
        u = queue.pop(0)
        for v, idx in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edges.add(idx)
                queue.append(v)
                
    non_tree_edges = [i for i in range(num_edges) if i not in tree_edges]
    k = len(non_tree_edges)
    
    # Map edge indices to positions in the sign array
    # We will construct the adjacency matrix for each configuration
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)
    
    # Pre-allocate the batch of adjacency matrices
    # Shape: (2^k, n, n)
    num_configs = 1 << k
    # Note: For m=9, k=10, num_configs=1024. n=18. 1024*18*18 ~ 330k floats. Very small.
    A_batch = np.zeros((num_configs, n, n), dtype=np.float64)
    
    # Fill tree edges (always +1)
    for idx in tree_edges:
        u, v = edge_u[idx], edge_v[idx]
        # Set for all configurations
        A_batch[:, u, v] = 1.0
        A_batch[:, v, u] = 1.0
        
    # Fill non-tree edges based on masks
    # We need to map each mask to its configuration index
    # Mask bit i corresponds to non_tree_edges[i]
    for mask in range(num_configs):
        # Initialize with tree edges (already done, but we need to add non-tree)
        # Actually, A_batch is currently 0 except tree edges.
        # We need to add non-tree edges for this specific mask.
        # It is more efficient to build the base matrix once and add deltas.
        # But A_batch is already allocated.
        pass
    
    # Better approach: Build a base matrix with tree edges, and a list of non-tree edge pairs.
    A_base = np.zeros((n, n), dtype=np.float64)
    for idx in tree_edges:
        u, v = edge_u[idx], edge_v[idx]
        A_base[u, v] = 1.0
        A_base[v, u] = 1.0
        
    # List of (u, v) for non-tree edges
    non_tree_pairs = []
    for idx in non_tree_edges:
        u, v = edge_u[idx], edge_v[idx]
        non_tree_pairs.append((u, v))
        
    # Generate all configurations
    radii = np.zeros(num_configs, dtype=np.float64)
    
    # If k is small, we can just loop. If k is larger, we might need to be smarter.
    # Here k <= 10, so looping is fine, but we want to be distinct.
    # We will use the batched eigvalsh approach for speed and distinctness.
    
    # Re-allocate A_batch properly
    # A_batch[i] = A_base + sum_{j in non_tree} sign_{i,j} * (E_{u_j v_j} + E_{v_j u_j})
    
    # Create a tensor of edge contributions
    # For each non-tree edge j, we have a matrix M_j which is 0 everywhere except (u_j, v_j) and (v_j, u_j) are 1.
    # We can stack these into a 3D array of shape (k, n, n)
    M_non_tree = np.zeros((k, n, n), dtype=np.float64)
    for j, (u, v) in enumerate(non_tree_pairs):
        M_non_tree[j, u, v] = 1.0
        M_non_tree[j, v, u] = 1.0
        
    # Generate signs for all masks
    # signs_batch[i, j] is the sign of non-tree edge j in configuration i
    # We want -1 or 1.
    # Mask bit j is 1 -> sign -1? Or 1?
    # Let's say bit 1 -> -1, bit 0 -> 1.
    # signs = 1 - 2 * bits
    
    # Create bit matrix
    # bits shape: (num_configs, k)
    bits = np.zeros((num_configs, k), dtype=np.int8)
    for mask in range(num_configs):
        for j in range(k):
            if (mask >> j) & 1:
                bits[mask, j] = 1
    # Convert to signs: 1 -> 1, 0 -> -1? 
    # Let's map 1 -> -1, 0 -> 1 to have variety.
    # sign = 1 if bit=0, -1 if bit=1.
    # sign = 1 - 2*bit
    
    signs_batch = 1.0 - 2.0 * bits # Shape (num_configs, k), values 1 or -1
    
    # Compute A_batch = A_base + signs_batch @ M_non_tree
    # signs_batch (num_configs, k) @ M_non_tree (k, n, n) -> (num_configs, n, n)
    # This uses tensordot or einsum.
    A_batch = A_base[np.newaxis, :, :] + np.einsum('ck,cij->kij', signs_batch, M_non_tree)
    
    # Compute eigenvalues for all matrices
    eigvals = np.linalg.eigvalsh(A_batch) # Shape (num_configs, n)
    radii = np.max(np.abs(eigvals), axis=1)
    
    # Sort indices by radius
    sorted_indices = np.argsort(radii)
    
    # Check validity in order of increasing radius
    def is_valid_lift_fast(signs):
        # signs is a 1D array of length num_edges
        # Build lift graph
        lift_adj = {i: set() for i in range(2 * n)}
        
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                lift_adj[u * 2].add(v * 2)
                lift_adj[v * 2].add(u * 2)
                lift_adj[u * 2 + 1].add(v * 2 + 1)
                lift_adj[v * 2 + 1].add(u * 2 + 1)
            else:
                lift_adj[u * 2].add(v * 2 + 1)
                lift_adj[v * 2 + 1].add(u * 2)
                lift_adj[u * 2 + 1].add(v * 2)
                lift_adj[v * 2].add(u * 2 + 1)
        
        # BFS
        visited = [False] * (2 * n)
        queue = [0]
        visited[0] = True
        count = 1
        while queue:
            u = queue.pop(0)
            for v in lift_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    count += 1
                    queue.append(v)
        
        return count == 2 * n

    # Iterate through candidates
    for idx in sorted_indices:
        # Reconstruct signs for this configuration
        # Tree edges are +1
        # Non-tree edges are signs_batch[idx]
        signs = np.ones(num_edges, dtype=np.int8)
        
        # Set non-tree edges
        for j, edge_idx in enumerate(non_tree_edges):
            signs[edge_idx] = int(signs_batch[idx, j])
            
        if is_valid_lift_fast(signs):
            return signs
            
    # Fallback
    # If no valid lift found (should not happen), return all +1 except one
    signs = np.ones(num_edges, dtype=np.int8)
    if num_edges > 0:
        signs[0] = -1
    return signs

