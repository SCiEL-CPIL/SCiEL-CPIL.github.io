"""
K2,3-free graph construction on 20 vertices maximizing edges.

Strategy:
- Use NumPy vectorized adjacency matrix for fast common-neighbor checks.
- High-throughput greedy construction identifies maximum valid edge set quickly.
- Multi-restart local search with swap moves for further optimization.
- Return the best graph found within the time budget.
"""

import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    n = 20
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    # Precompute all pairs for vectorized operations
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    n_pairs = len(all_pairs)
    pair_u = np.array([p[0] for p in all_pairs], dtype=np.int8)
    pair_v = np.array([p[1] for p in all_pairs], dtype=np.int8)
    
    def count_edges(adj_mat):
        return int(np.sum(adj_mat)) // 2
    
    def valid_add_vectorized(adj_mat, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free using vectorized ops."""
        if adj_mat[u, v]:
            return False
        
        # 1. u and v must have at most 2 common neighbors
        common_uv = adj_mat[u] & adj_mat[v]
        if np.sum(common_uv) > 2:
            return False
        
        # 2. For each w in N(u), w and v must have at most 1 common neighbor
        #    Vectorized: check all w where adj_mat[w,u] == 1
        neighbors_u = np.where(adj_mat[:, u])[0]
        for w in neighbors_u:
            common_wv = adj_mat[w] & adj_mat[v]
            if np.sum(common_wv) > 1:
                return False
        
        # 3. For each w in N(v), w and u must have at most 1 common neighbor
        neighbors_v = np.where(adj_mat[:, v])[0]
        for w in neighbors_v:
            common_wu = adj_mat[w] & adj_mat[u]
            if np.sum(common_wu) > 1:
                return False
        
        return True
    
    def valid_add_batch(adj_mat, pairs):
        """Check validity of multiple edge additions simultaneously."""
        results = np.zeros(len(pairs), dtype=bool)
        for i, (u, v) in enumerate(pairs):
            if adj_mat[u, v]:
                continue
            common_uv = adj_mat[u] & adj_mat[v]
            if np.sum(common_uv) > 2:
                continue
            neighbors_u = np.where(adj_mat[:, u])[0]
            valid = True
            for w in neighbors_u:
                if np.sum(adj_mat[w] & adj_mat[v]) > 1:
                    valid = False
                    break
            if not valid:
                continue
            neighbors_v = np.where(adj_mat[:, v])[0]
            for w in neighbors_v:
                if np.sum(adj_mat[w] & adj_mat[u]) > 1:
                    valid = False
                    break
            results[i] = valid
        return results
    
    def add_edge(adj_mat, u, v):
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1
    
    def remove_edge(adj_mat, u, v):
        adj_mat[u, v] = 0
        adj_mat[v, u] = 0
    
    def get_edges(adj_mat):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_mat[u, v]:
                    edges.append((u, v))
        return edges
    
    def get_non_edges(adj_mat):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not adj_mat[u, v]:
                    edges.append((u, v))
        return edges
    
    def greedy_fill_vectorized(adj_mat):
        """Greedily add all possible edges using vectorized checks."""
        non_edges = get_non_edges(adj_mat)
        random.shuffle(non_edges)
        for u, v in non_edges:
            if valid_add_vectorized(adj_mat, u, v):
                add_edge(adj_mat, u, v)
    
    def make_k4_union():
        adj_mat = np.zeros((n, n), dtype=np.int8)
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    add_edge(adj_mat, base + a, base + b)
        return adj_mat
    
    best_edges = 0
    best_adj = None
    
    # First, establish baseline with vectorized greedy
    adj0 = make_k4_union()
    greedy_fill_vectorized(adj0)
    best_edges = count_edges(adj0)
    best_adj = adj0.copy()
    
    # Multi-restart simulated annealing
    num_restarts = 25
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Different starting points
        if r == 0:
            adj = make_k4_union()
        else:
            # Start from best found, then perturb
            adj = best_adj.copy()
            # Randomly remove a few edges
            edges = get_edges(adj)
            random.shuffle(edges)
            for u, v in edges[:3]:
                remove_edge(adj, u, v)
        
        greedy_fill_vectorized(adj)
        current_edges = count_edges(adj)
        
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj.copy()
        
        # Simulated annealing
        T = 0.5
        T_min = 0.05
        cooling = 0.995
        steps = 0
        
        while T > T_min and (time.time() - start_time < time_limit):
            steps += 1
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to add a random non-edge
                non_edges = get_non_edges(adj)
                if not non_edges:
                    T *= cooling
                    continue
                u, v = random.choice(non_edges)
                if valid_add_vectorized(adj, u, v):
                    add_edge(adj, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj.copy()
                
            elif move_type < 0.6:
                # Try to remove a random edge (accept with SA probability)
                edges = get_edges(adj)
                if not edges:
                    T *= cooling
                    continue
                u, v = random.choice(edges)
                if random.random() < math.exp(-1.0 / T):
                    remove_edge(adj, u, v)
                    current_edges -= 1
            
            else:
                # Swap: remove one edge, try to add another
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or not non_edges:
                    T *= cooling
                    continue
                for _ in range(5):
                    u1, v1 = random.choice(edges)
                    u2, v2 = random.choice(non_edges)
                    remove_edge(adj, u1, v1)
                    if valid_add_vectorized(adj, u2, v2):
                        add_edge(adj, u2, v2)
                        if random.random() < 0.8:
                            pass
                        else:
                            remove_edge(adj, u2, v2)
                            add_edge(adj, u1, v1)
                        break
                    else:
                        add_edge(adj, u1, v1)
            
            T *= cooling
    
    # Final polish: exhaustive greedy + systematic swap search
    if best_adj is not None:
        greedy_fill_vectorized(best_adj)
        final_edges = count_edges(best_adj)
        if final_edges > best_edges:
            best_edges = final_edges
        
        # Systematic polish: try all possible single edge additions
        changed = True
        while changed:
            changed = False
            non_edges = get_non_edges(best_adj)
            for u, v in non_edges:
                if time.time() - start_time > time_limit:
                    break
                if valid_add_vectorized(best_adj, u, v):
                    add_edge(best_adj, u, v)
                    final_edges += 1
                    changed = True
            if time.time() - start_time > time_limit:
                break
        
        # Systematic polish: try 2-edge swaps
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            improved = False
            edges = get_edges(best_adj)
            non_edges = get_non_edges(best_adj)
            for u1, v1 in edges:
                if time.time() - start_time > time_limit:
                    break
                for u2, v2 in non_edges:
                    if time.time() - start_time > time_limit:
                        break
                    remove_edge(best_adj, u1, v1)
                    if valid_add_vectorized(best_adj, u2, v2):
                        add_edge(best_adj, u2, v2)
                        save_adj = best_adj.copy()
                        greedy_fill_vectorized(save_adj)
                        save_edges = count_edges(save_adj)
                        if save_edges > final_edges:
                            best_adj = save_adj
                            final_edges = save_edges
                            improved = True
                            break
                        else:
                            remove_edge(best_adj, u2, v2)
                            add_edge(best_adj, u1, v1)
                    else:
                        add_edge(best_adj, u1, v1)
                if improved:
                    break
            if not improved:
                break
        
        if final_edges > best_edges:
            best_edges = final_edges
    
    # Convert to list of edges
    result = []
    for u, v in get_edges(best_adj):
        result.append([u, v])
    return result

# EVOLVE-BLOCK-END
