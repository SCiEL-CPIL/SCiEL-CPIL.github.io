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
    
    best_radius = float('inf')
    best_signs = None
    
    # Pre-compute base adjacency matrix with tree edges set to +1
    base_A = np.zeros((n, n))
    for idx, (u, v) in enumerate(edges):
        if idx in tree_edges:
            base_A[u, v] = 1
            base_A[v, u] = 1
            
    # Pre-compute non-tree edge endpoints for fast update
    non_tree_u = [edges[idx][0] for idx in non_tree_edges]
    non_tree_v = [edges[idx][1] for idx in non_tree_edges]
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Iterate over all 2^k combinations of non-tree edge signs
    # k = m + 1
    for mask in range(1 << k):
        if time.time() > deadline:
            break
            
        # Skip the all-+1 case (mask=0) which is known to be disconnected
        # But we will verify connectivity with BFS to be safe
        if mask == 0:
             # Check if all-+1 is connected? It's not.
             # But let's just run the general check logic below
             pass

        # Build signed adjacency matrix
        A = base_A.copy()
        
        # Set non-tree edge signs based on mask
        for i, edge_idx in enumerate(non_tree_edges):
            if (mask >> i) & 1:
                s = -1
            else:
                s = 1
            u = non_tree_u[i]
            v = non_tree_v[i]
            A[u, v] = s
            A[v, u] = s
            
        # Check connectivity of the 2-lift
        # Build lift graph
        # Nodes: 0..2n-1. (v, 0) is v, (v, 1) is v + n
        lift_adj = [[] for _ in range(2 * n)]
        for idx, (u, v) in enumerate(edges):
            # Determine sign
            if idx in tree_edges:
                s = 1
            else:
                # Find position in non_tree_edges
                # This is inefficient. Better to precompute signs array.
                pass
        
        # Let's construct signs array first for this mask
        signs = np.ones(num_edges, dtype=int)
        for i, edge_idx in enumerate(non_tree_edges):
            if (mask >> i) & 1:
                signs[edge_idx] = -1
        
        # Now build lift adjacency using signs
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            # u0, u1, v0, v1
            u0 = u
            u1 = u + n
            v0 = v
            v1 = v + n
            
            if s == 1:
                lift_adj[u0].append(v0)
                lift_adj[v0].append(u0)
                lift_adj[u1].append(v1)
                lift_adj[v1].append(u1)
            else:
                lift_adj[u0].append(v1)
                lift_adj[v1].append(u0)
                lift_adj[u1].append(v0)
                lift_adj[v0].append(u1)
                
        # BFS on lift
        visited_lift = [False] * (2 * n)
        q = [0]
        visited_lift[0] = True
        count = 1
        while q:
            cur = q.pop(0)
            for nei in lift_adj[cur]:
                if not visited_lift[nei]:
                    visited_lift[nei] = True
                    count += 1
                    q.append(nei)
                    
        if count < 2 * n:
            # Disconnected
            continue
            
        # Compute spectral radius
        eigvals = np.linalg.eigvalsh(A)
        radius = np.max(np.abs(eigvals))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs
            
    # Fallback if no valid lift found (should not happen)
    if best_signs is None:
        # Try a random one or the first non-zero mask
        signs = np.ones(num_edges, dtype=int)
        if num_edges > 0:
            signs[0] = -1
        best_signs = signs
        
    return best_signs
