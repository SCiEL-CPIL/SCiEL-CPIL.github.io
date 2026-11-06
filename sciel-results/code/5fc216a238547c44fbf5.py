import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    n = 16
    target_edges = 65
    
    def make_initial_edges():
        edges = set()
        for u in range(8):
            for v in range(8, 16):
                edges.add((min(u,v), max(u,v)))
        edges.add((0, 1))
        return edges
    
    def edges_to_list(edge_set):
        return [[u, v] for u, v in sorted(edge_set)]
    
    # Fast scoring using bitmasks for adjacency
    # adj[i] is a bitmask of neighbors of i
    def compute_adj_bitmask(edge_set):
        adj = [0] * n
        deg = [0] * n
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg
    
    def max_triangle_sum_fast(adj, deg):
        """Use bitmask operations to find max triangle degree sum."""
        max_s = 0
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    # Common neighbors of i and j
                    common = adj[i] & adj[j]
                    # Only consider k > j to avoid duplicates
                    common &= ~((1 << (j+1)) - 1)  # mask out 0..j
                    while common:
                        k = common & (-common)
                        k_idx = k.bit_length() - 1
                        common ^= k
                        s = deg[i] + deg[j] + deg[k_idx]
                        if s > max_s:
                            max_s = s
        return max_s
    
    best_edges = make_initial_edges()
    best_score = max_triangle_sum_fast(*compute_adj_bitmask(best_edges))
    
    current_edges = set(best_edges)
    current_adj, current_deg = compute_adj_bitmask(current_edges)
    current_max = best_score
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Local search with full swap enumeration
    while (time.time() - start_time) < DEADLINE - 5:
        improved = False
        for remove_e in sorted(current_edges):
            if (time.time() - start_time) > DEADLINE - 5:
                break
            ru, rv = remove_e
            for add_e in all_edges:
                if add_e in current_edges:
                    continue
                au, av = add_e
                
                # Update degrees
                new_deg = list(current_deg)
                new_deg[ru] -= 1
                new_deg[rv] -= 1
                new_deg[au] += 1
                new_deg[av] += 1
                
                # Update adjacency
                new_adj = list(current_adj)
                new_adj[ru] &= ~(1 << rv)
                new_adj[rv] &= ~(1 << ru)
                new_adj[au] |= (1 << av)
                new_adj[av] |= (1 << au)
                
                new_max = max_triangle_sum_fast(new_adj, new_deg)
                
                if new_max < current_max:
                    current_edges = set(current_edges)
                    current_edges.remove(remove_e)
                    current_edges.add(add_e)
                    current_adj = new_adj
                    current_deg = new_deg
                    current_max = new_max
                    if current_max < best_score:
                        best_score = current_max
                        best_edges = set(current_edges)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return {'n': n, 'edges': edges_to_list(best_edges)}
