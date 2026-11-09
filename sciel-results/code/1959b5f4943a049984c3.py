"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n = 20
    indices = np.arange(n)
    mask_w_not_u = (indices[None, None, :] != indices[:, None, None])
    mask_w_not_v = (indices[None, None, :] != indices[None, :, None])
    mask_valid_w = mask_w_not_u & mask_w_not_v

    def get_valid_edges(adj):
        D = adj @ adj
        # T1[u, v, w] = D[u, w] + adj[v, w]
        # Check max over w is <= 2
        T1 = D[:, None, :] + adj[None, :, :]
        # T2[u, v, w] = D[v, w] + adj[u, w]
        T2 = D[None, :, :] + adj[:, None, :]
        
        # We only care about w != u and w != v for the max?
        # Actually, if w == u, D[u, u] + adj[v, u] = 0 + 1 (if edge exists) or 0.
        # D[i, i] is 0.
        # If w == u, term is D[u, u] + adj[v, u] = adj[v, u].
        # If w == v, term is D[u, v] + adj[v, v] = D[u, v].
        # The constraint is on common neighbors.
        # Common neighbors of u and v are w != u, v.
        # But the formula C(u, w) + adj[v, w] <= 2 must hold for ALL w.
        # If w = u, C(u, u) is 0. adj[v, u] is 1 if edge exists.
        # We are checking if we CAN add (u, v). So adj[u, v] is 0 currently.
        # So adj[v, u] is 0. Term is 0.
        # If w = v, C(u, v) + adj[v, v] = C(u, v) + 0.
        # So we just need max over all w to be <= 2.
        
        Max1 = T1.max(axis=2)
        Max2 = T2.max(axis=2)
        
        valid_mask = (adj == 0) & (Max1 <= 2) & (Max2 <= 2)
        np.fill_diagonal(valid_mask, False)
        return valid_mask

    def greedy_fill(adj, seed=None):
        if seed is not None:
            random.seed(seed)
        while True:
            valid_mask = get_valid_edges(adj)
            valid_indices = np.argwhere(valid_mask)
            if len(valid_indices) == 0:
                break
            
            # Shuffle valid edges
            valid_list = valid_indices.tolist()
            random.shuffle(valid_list)
            valid_indices = valid_list
            
            # Greedily pick disjoint edges
            used = np.zeros(n, dtype=bool)
            added_any = False
            for u, v in valid_indices:
                if not used[u] and not used[v]:
                    adj[u, v] = 1
                    adj[v, u] = 1
                    used[u] = True
                    used[v] = True
                    added_any = True
            
            if not added_any:
                break
        return adj

    def local_search(adj, max_iterations=500):
        current_adj = adj.copy()
        current_count = np.sum(current_adj) // 2
        best_adj = current_adj.copy()
        best_count = current_count
        no_improve_count = 0
        
        for _ in range(max_iterations):
            # Pick a random edge to remove
            edges_in_graph = np.argwhere(current_adj == 1)
            # Filter to i < j to avoid duplicates
            edges_in_graph = edges_in_graph[edges_in_graph[:, 0] < edges_in_graph[:, 1]]
            if len(edges_in_graph) == 0:
                break
                
            idx = random.randint(0, len(edges_in_graph) - 1)
            u, v = edges_in_graph[idx]
            
            new_adj = current_adj.copy()
            new_adj[u, v] = 0
            new_adj[v, u] = 0
            
            # Greedily fill
            new_adj = greedy_fill(new_adj)
            new_count = np.sum(new_adj) // 2
            
            if new_count > best_count:
                best_adj = new_adj
                best_count = new_count
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Always update current to the new state to explore further? 
            # Or revert if worse? Hill climbing usually reverts if worse.
            # But here we might want to escape local minima.
            # Let's try: if new_count >= current_count, accept.
            if new_count >= current_count:
                current_adj = new_adj
                current_count = new_count
            else:
                # Revert
                pass
                
            if no_improve_count > 100:
                break
                
        return best_adj, best_count

    start_time = time.time()
    best_adj = None
    best_count = 0
    
    # Phase 1: Multiple greedy restarts
    # Since greedy is faster, we can do more restarts
    num_restarts = 500 
    for i in range(num_restarts):
        if time.time() - start_time > 90:
            break
        adj = np.zeros((n, n), dtype=np.int8)
        adj = greedy_fill(adj, seed=i)
        count = np.sum(adj) // 2
        if count > best_count:
            best_count = count
            best_adj = adj.copy()
            
    if best_adj is None:
        return []
        
    # Phase 2: Local search on the best solution
    # Use remaining time for local search
    time_left = 110 - (time.time() - start_time)
    if time_left > 5:
        # Scale iterations with time?
        iters = int(time_left * 20)
        best_adj, best_count = local_search(best_adj, max_iterations=iters)
        
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j]:
                edges.append([i, j])
                
    return edges
# EVOLVE-BLOCK-END
