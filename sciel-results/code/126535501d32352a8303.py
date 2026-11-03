"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_best_swap(p):
        """
        Finds the best swap (i, j) and its delta using vectorized calculations.
        Returns (best_i, best_j, best_delta).
        """
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Precompute column of A and row of A? 
        # We will loop over i, and vectorize over j.
        
        for i in range(n):
            u = p[i]
            # V[v, k] = b[v, p[k]] - b[u, p[k]]
            # b[:, p] is shape (n, n). b[u, p] is shape (n,).
            V = b[:, p] - b[u, p][np.newaxis, :]
            
            # M[j, v] = sum_k a[j, k] * V[v, k]
            # M = a @ V.T
            M = a @ V.T
            
            # ProjA_i[v] = M[i, v]
            ProjA_i = M[i, :]
            
            # Term2_all[j] = sum_k a[j, k] * V[p[j], k] = M[j, p[j]]
            Term2_all = M[np.arange(n), p]
            
            # S_all[j] = ProjA_i[p[j]] - Term2_all[j]
            S_all = ProjA_i[p] - Term2_all
            
            # Term_i_all[j] = (a[i, i] - a[j, i]) * (b[p[j], u] - b[u, u])
            # a[:, i] is column i
            # b[p, u] is column u gathered by p
            Term_i_all = (a[i, i] - a[:, i]) * (b[p, u] - b[u, u])
            
            # Term_j_all[j] = (a[i, j] - a[j, j]) * (b[p[j], p[j]] - b[u, p[j]])
            # a[i, :] is row i
            # np.diag(a) is diagonal
            # b[p, p] is diagonal gathered by p
            # b[u, p] is row u gathered by p
            Term_j_all = (a[i, :] - np.diag(a)) * (b[p, p] - b[u, p])
            
            # Direct_all[j] = (a[i, j] + a[j, i]) * (b[p[j], u] - b[u, p[j]])
            Direct_all = (a[i, :] + a[:, i]) * (b[p, u] - b[u, p])
            
            # Delta_all[j]
            Delta_all = 2 * (S_all - Term_i_all - Term_j_all) + Direct_all
            
            # Mask out j = i
            Delta_all[i] = 0 # We only care about negative deltas, 0 is no change
            
            # Find min
            min_idx = np.argmin(Delta_all)
            min_val = Delta_all[min_idx]
            
            if min_val < best_delta:
                best_delta = min_val
                best_i = i
                best_j = min_idx
                
        return best_i, best_j, best_delta

    def local_search(p, current_cost):
        # Best Improvement Local Search
        improved = True
        while improved:
            improved = False
            best_i, best_j, best_delta = get_best_swap(p)
            
            if best_delta < 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_delta
                improved = True
                
        return p, current_cost

    start_time = time.time()
    timeout = 115.0
    
    # 1. Initial Solution: Rank Matching
    flow_sums = a.sum(axis=1)
    dist_sums = b.sum(axis=1)
    
    heavy_facilities = np.argsort(-flow_sums, kind="stable")
    central_locations = np.argsort(dist_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_facilities] = central_locations
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Local Search on initial solution
    p, current_cost = local_search(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # 3. ILS Loop
    p = best_p.copy()
    current_cost = best_cost
    
    while time.time() - start_time < timeout:
        # Perturbation
        num_swaps = 5
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            # Calculate delta for this specific swap
            # We can use the old get_delta or just calculate it
            # To be consistent and fast, let's use a simple inline calc or the vectorized one is overkill for 1 swap
            # But we don't have a single-swap delta function. 
            # Let's just swap and recalc? No, too slow.
            # Let's implement a quick single swap delta.
            u = p[i]
            v = p[j]
            # Reuse the logic from get_best_swap but for single pair?
            # Or just use the old get_delta if we kept it. 
            # Since we removed it, let's add a simple one.
            
            # Actually, we can just use the vectorized one if we want, but it's O(N^2).
            # A simple O(N) delta is better for kicks.
            
            # delta = 2 * sum_{k != i, j} (a[i, k] - a[j, k]) * (b[v, p[k]] - b[u, p[k]]) + direct
            # Let's just compute it.
            diff_a = a[i] - a[j]
            diff_b = b[v, p] - b[u, p]
            sum_all = np.sum(diff_a * diff_b)
            
            term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
            term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
            
            direct = (a[i, j] + a[j, i]) * (b[v, u] - b[u, v])
            
            delta = 2 * (sum_all - term_i - term_j) + direct
            
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
        # Local Search
        p, current_cost = local_search(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        if time.time() - start_time > 20.0:
             p = best_p.copy()
             current_cost = best_cost
             num_swaps = 5
             for _ in range(num_swaps):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j: continue
                u = p[i]
                v = p[j]
                diff_a = a[i] - a[j]
                diff_b = b[v, p] - b[u, p]
                sum_all = np.sum(diff_a * diff_b)
                term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                direct = (a[i, j] + a[j, i]) * (b[v, u] - b[u, v])
                delta = 2 * (sum_all - term_i - term_j) + direct
                p[i], p[j] = p[j], p[i]
                current_cost += delta
             
    return best_p.tolist()
# EVOLVE-BLOCK-END
