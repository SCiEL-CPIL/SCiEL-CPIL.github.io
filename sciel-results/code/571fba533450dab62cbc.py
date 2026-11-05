import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Generate Clebsch Graph
    vectors = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vectors.append(i)
            
    clebsch_edges = []
    adj_clebsch = [0] * 16
    for u_idx in range(16):
        for v_idx in range(u_idx + 1, 16):
            xor = vectors[u_idx] ^ vectors[v_idx]
            if bin(xor).count('1') == 4:
                clebsch_edges.append([u_idx, v_idx])
                adj_clebsch[u_idx] |= (1 << v_idx)
                adj_clebsch[v_idx] |= (1 << u_idx)
                
    best_edges = clebsch_edges
    best_count = len(clebsch_edges) # 40
    
    # 2. Fast 3-colorability check
    def is_3_colorable(adj_list):
        n = 16
        colors = [-1] * n
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            mask = adj_list[i]
            for j in range(n):
                if mask & (1 << j):
                    neighbors[i].append(j)
        
        order = sorted(range(n), key=lambda x: -len(neighbors[x]))
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            for c in range(3):
                valid = True
                for u in neighbors[v]:
                    if colors[u] == c:
                        valid = False
                        break
                if valid:
                    colors[v] = c
                    if backtrack(idx + 1):
                        return True
                    colors[v] = -1
            return False
            
        return backtrack(0)

    all_pairs = [(u, v) for u in range(16) for v in range(u+1, 16)]
    
    def can_add_edge(adj_mask, u, v):
        return (adj_mask[u] & adj_mask[v]) == 0

    # Spectral Greedy Construction
    def spectral_greedy_construction():
        adj = [0] * 16
        edges = []
        remaining = set(all_pairs)
        
        # Start with a random edge to break symmetry and ensure connectivity for Fiedler
        # Or just start empty. If empty, Fiedler is 0.
        
        while remaining:
            if time.time() > deadline:
                break
            
            # Find valid edges
            valid_edges = []
            for u, v in remaining:
                if can_add_edge(adj, u, v):
                    valid_edges.append((u, v))
            
            if not valid_edges:
                break
            
            if len(valid_edges) == 1:
                u, v = valid_edges[0]
            else:
                # Compute Fiedler vector
                # Build Laplacian
                L = np.zeros((16, 16))
                for i in range(16):
                    deg = bin(adj[i]).count('1')
                    L[i, i] = deg
                    mask = adj[i]
                    for j in range(16):
                        if mask & (1 << j):
                            L[i, j] = -1
                
                # Eigenvalues and eigenvectors
                # We want the 2nd smallest eigenvalue (Fiedler value)
                # If graph is disconnected, lambda_2 = 0.
                try:
                    eigvals, eigvecs = np.linalg.eigh(L)
                    # Sort by eigenvalue
                    idxs = np.argsort(eigvals)
                    fiedler_vec = eigvecs[:, idxs[1]]
                    
                    # Score each valid edge
                    best_score = -1
                    best_edge = None
                    for u, v in valid_edges:
                        diff = abs(fiedler_vec[u] - fiedler_vec[v])
                        if diff > best_score:
                            best_score = diff
                            best_edge = (u, v)
                    
                    if best_edge is None:
                        u, v = random.choice(valid_edges)
                    else:
                        u, v = best_edge
                        
                except Exception:
                    u, v = random.choice(valid_edges)
            
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
            remaining.discard((u, v))
            
        return adj, edges

    # Random Greedy Construction
    def random_greedy_construction():
        adj = [0] * 16
        edges = []
        pairs = list(all_pairs)
        random.shuffle(pairs)
        for u, v in pairs:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        return adj, edges

    # Main Loop
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        # Alternate between spectral and random
        if iterations % 2 == 0:
            adj, edges = spectral_greedy_construction()
        else:
            adj, edges = random_greedy_construction()
            
        if len(edges) > best_count:
            if not is_3_colorable(adj):
                best_edges = edges
                best_count = len(edges)
        
        # If we found a very good one, maybe try to polish?
        # But maximal triangle-free graphs can't be polished by adding edges.
        # So we just keep searching for new maximal graphs.

    return best_edges
