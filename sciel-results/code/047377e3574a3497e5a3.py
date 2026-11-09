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
        T1 = D[:, None, :] + adj.T[None, :, :]
        T2 = adj[:, None, :] + D.T[None, :, :]
        T1_masked = np.where(mask_valid_w, T1, -1)
        T2_masked = np.where(mask_valid_w, T2, -1)
        Max1 = T1_masked.max(axis=2)
        Max2 = T2_masked.max(axis=2)
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
            idx = random.randint(0, len(valid_indices) - 1)
            u, v = valid_indices[idx]
            adj[u, v] = 1
            adj[v, u] = 1
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
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() - start_time > 100:
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
    if time.time() - start_time < 110:
        best_adj, best_count = local_search(best_adj, max_iterations=200)
        
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j]:
                edges.append([i, j])
                
    return edges
# EVOLVE-BLOCK-END
