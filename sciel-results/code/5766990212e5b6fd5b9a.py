import time
import random
import numpy as np

def construct():
    n = 16
    # Initialize with seed
    # K_{8,8} + (0,1)
    adj = np.zeros((n, n), dtype=np.int8)
    for u in range(8):
        for v in range(8, 16):
            adj[u, v] = 1
            adj[v, u] = 1
    adj[0, 1] = 1
    adj[1, 0] = 1
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v]:
                edges.append([u, v])
                
    def get_score(adj):
        deg = adj.sum(axis=1)
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    # Find common neighbors
                    # common = np.logical_and(adj[u], adj[v])
                    # common_indices = np.where(common)[0]
                    # This is slow. Use bitsets?
                    # n=16 fits in int.
                    mask_u = 0
                    mask_v = 0
                    for i in range(n):
                        if adj[u, i]: mask_u |= (1 << i)
                        if adj[v, i]: mask_v |= (1 << i)
                    common_mask = mask_u & mask_v
                    while common_mask:
                        # Get lowest set bit
                        lsb = common_mask & -common_mask
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > max_sum:
                            max_sum = s
                        common_mask &= common_mask - 1
        return max_sum

    current_score = get_score(adj)
    best_adj = adj.copy()
    best_score = current_score
    
    start_time = time.time()
    time_limit = 110  # Safety margin
    
    # Local Search
    # Moves: 
    # 1. Remove edge (u,v), Add edge (x,y)
    # Ensure (x,y) not in graph, (u,v) in graph.
    
    all_possible_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    while time.time() - start_time < time_limit:
        # Pick a random edge to remove
        # Pick a random non-edge to add
        # Try a few random moves
        
        improved = False
        for _ in range(100): # Batch size
            if time.time() - start_time > time_limit:
                break
                
            # Remove a random edge
            u, v = random.choice(edges)
            # Add a random non-edge
            # Find a non-edge
            x, y = random.choice(all_possible_edges)
            if adj[x, y] == 1:
                continue
            if x == u and y == v:
                continue
            if x == v and y == u:
                continue
            # Also avoid creating duplicate edges if (x,y) is same as (u,v)
            
            # Apply move
            adj[u, v] = 0
            adj[v, u] = 0
            adj[x, y] = 1
            adj[y, x] = 1
            
            # Update edges list (simple: rebuild if needed, or track)
            # For simplicity, we can just check validity in score function or assume valid
            
            new_score = get_score(adj)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj.copy()
                improved = True
            else:
                # Revert
                adj[u, v] = 1
                adj[v, u] = 1
                adj[x, y] = 0
                adj[y, x] = 0
        
        if not improved:
            # Random restart or kick
            # Just continue for now
            break
            
    # Convert best_adj to edges list
    best_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                best_edges.append([u, v])
                
    return {'n': 16, 'edges': best_edges}
