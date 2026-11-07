import numpy as np
import time

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Build adjacency list with edge indices
    adj = {i: [] for i in range(n)}
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        
    # Find a spanning tree using BFS
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
                
    # Identify non-tree edges
    non_tree_edges = [i for i in range(num_edges) if i not in tree_edges]
    k = len(non_tree_edges)
    
    # Pre-compute base adjacency matrix structure
    # For each edge, store (u, v) and whether it's a tree edge
    edge_u = np.array([e[0] for e in edges], dtype=np.int32)
    edge_v = np.array([e[1] for e in edges], dtype=np.int32)
    is_tree_edge = np.array([1 if i in tree_edges else 0 for i in range(num_edges)], dtype=np.int8)
    
    def compute_radius(signs):
        """Compute spectral radius of signed adjacency matrix."""
        A = np.zeros((n, n))
        # Vectorized: set A[u,v] = sign and A[v,u] = sign for each edge
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))
    
    def is_valid_lift(signs):
        """Check if the lift is connected and 3-regular."""
        # Build the lift graph
        # Vertices: (v, 0) and (v, 1) for v in 0..n-1
        # Total vertices: 2n
        num_lift_vertices = 2 * n
        lift_adj = {i: set() for i in range(num_lift_vertices)}
        
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            if s == 1:
                # +1: joins equal sides
                lift_adj[u * 2].add(v * 2)
                lift_adj[v * 2].add(u * 2)
                lift_adj[u * 2 + 1].add(v * 2 + 1)
                lift_adj[v * 2 + 1].add(u * 2 + 1)
            else:
                # -1: joins opposite sides
                lift_adj[u * 2].add(v * 2 + 1)
                lift_adj[v * 2 + 1].add(u * 2)
                lift_adj[u * 2 + 1].add(v * 2)
                lift_adj[v * 2].add(u * 2 + 1)
        
        # Check 3-regular
        for i in range(num_lift_vertices):
            if len(lift_adj[i]) != 3:
                return False
        
        # Check connectivity using BFS
        visited = [False] * num_lift_vertices
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
        
        return count == num_lift_vertices
    
    best_radius = float('inf')
    best_signs = None
    
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Phase 1: Exhaustive search over non-tree edges
    for mask in range(1 << k):
        if time.time() > deadline:
            break
            
        # Skip the balanced case (all non-tree edges +1)
        if mask == 0:
            continue
            
        signs = np.ones(num_edges, dtype=np.int8)
        
        # Set non-tree edge signs based on mask
        for i, edge_idx in enumerate(non_tree_edges):
            if (mask >> i) & 1:
                signs[edge_idx] = -1
            else:
                signs[edge_idx] = 1
                
        # Check validity first (cheaper than eigenvalue computation)
        if not is_valid_lift(signs):
            continue
            
        # Compute spectral radius
        radius = compute_radius(signs)
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    # Phase 2: Local search to refine the best solution
    if best_signs is not None:
        current_signs = best_signs.copy()
        current_radius = best_radius
        
        # Local search: flip one edge at a time
        max_local_steps = 20000
        step = 0
        while step < max_local_steps:
            if time.time() > deadline:
                break
                
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                    
                # Try flipping edge i
                new_signs = current_signs.copy()
                new_signs[i] = -new_signs[i]
                
                # Check validity
                if not is_valid_lift(new_signs):
                    continue
                
                # Compute new radius
                new_radius = compute_radius(new_signs)
                
                if new_radius < current_radius - 1e-10:
                    current_signs = new_signs
                    current_radius = new_radius
                    improved = True
                    break  # Restart from this new state
            
            if not improved:
                break
            
            step += 1
        
        best_signs = current_signs
        best_radius = current_radius
    
    # Fallback if no valid lift found
    if best_signs is None:
        # Try a few simple patterns
        for test_mask in [1, 2, 4, 8, 16, 32]:
            if test_mask >= (1 << k):
                continue
            signs = np.ones(num_edges, dtype=np.int8)
            for i, edge_idx in enumerate(non_tree_edges):
                if (test_mask >> i) & 1:
                    signs[edge_idx] = -1
            
            if is_valid_lift(signs):
                best_signs = signs
                best_radius = compute_radius(signs)
                break
        
        if best_signs is None:
            # Last resort: all +1 except one edge
            signs = np.ones(num_edges, dtype=np.int8)
            signs[0] = -1
            best_signs = signs
    
    return best_signs
