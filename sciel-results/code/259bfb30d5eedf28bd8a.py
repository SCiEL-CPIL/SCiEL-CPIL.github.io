"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    def calc_cost(p):
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_delta(p, i, j):
        u = p[i]
        v = p[j]
        
        b_v_p = b[v, p]
        b_u_p = b[u, p]
        diff_b = b_v_p - b_u_p
        
        diff_a = a[i] - a[j]
        
        sum_all = np.sum(diff_a * diff_b)
        
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        delta = 2 * (sum_all - term_i - term_j)
        return delta

    def local_search(p, current_cost, max_iter=None):
        # First Improvement Local Search with vectorized pair evaluation
        # We evaluate all pairs for a given i, find best improving move, apply it, repeat.
        # This is "Best Improvement" per i, but we restart after each move.
        
        if max_iter is None:
            max_iter = 1000
            
        iter_count = 0
        while iter_count < max_iter:
            iter_count += 1
            if time.time() - start_time > timeout:
                break
            
            moved = False
            for i in range(n):
                if time.time() - start_time > timeout:
                    break
                
                u = p[i]
                b_u_p = b[u, p]
                a_i = a[i]
                
                best_delta = 0
                best_j = -1
                
                for j in range(i + 1, n):
                    v = p[j]
                    diff_b = b[v, p] - b_u_p
                    diff_a = a_i - a[j]
                    sum_all = np.dot(diff_a, diff_b)
                    
                    term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
                    term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
                    
                    delta = 2 * (sum_all - term_i - term_j)
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_j = j
                
                if best_j != -1:
                    j = best_j
                    p[i], p[j] = p[j], p[i]
                    current_cost += best_delta
                    moved = True
                    break # Restart scan from i=0
            
            if not moved:
                break
        
        return p, current_cost

    def hungarian_init():
        # Use linear_sum_assignment on a matrix derived from a and b
        # A common heuristic is to use the sum of rows/columns
        # Or simply use a random permutation if Hungarian is too slow or not helpful
        # Let's try a simple rank-based initialization first, then maybe Hungarian on a surrogate
        
        # Surrogate: minimize sum a[i,j] * b[p[i], p[j]]
        # This is hard to directly use Hungarian on.
        # However, we can try to match facilities with high flow sums to locations with low distance sums.
        
        flow_sums = a.sum(axis=1)
        dist_sums = b.sum(axis=1)
        
        # Sort facilities by flow (heaviest first)
        heavy_facilities = np.argsort(-flow_sums, kind="stable")
        # Sort locations by distance sum (most central first - lowest sum is usually "central" in QAP context? 
        # Actually, we want to minimize sum of products. 
        # High flow should go to low distance? 
        # Let's check correlation. 
        # If a[i,j] is large, we want b[p[i], p[j]] to be small.
        # So heavy facilities (high sum of flows) should be placed at locations that have low sum of distances to others?
        # Yes.
        central_locations = np.argsort(dist_sums, kind="stable")
        
        p = np.empty(n, dtype=np.int64)
        p[heavy_facilities] = central_locations
        
        return p

    start_time = time.time()
    timeout = 115.0
    
    # 1. Initial Solution: Rank Matching
    p = hungarian_init()
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Local Search on initial solution
    p, current_cost = local_search(p, current_cost)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # 3. ILS Loop with adaptive perturbation
    p = best_p.copy()
    current_cost = best_cost
    no_improve_count = 0
    
    while time.time() - start_time < timeout:
        # Perturbation: vary the number of swaps based on stagnation
        if no_improve_count < 5:
            num_swaps = 3
        elif no_improve_count < 15:
            num_swaps = 7
        else:
            num_swaps = 12
        
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = get_delta(p, i, j)
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
        # Local Search
        p, current_cost = local_search(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Reset p to best_p to stay in the neighborhood of the best solution
            p = best_p.copy()
            current_cost = best_cost
            
        # Check timeout before next iteration
        if time.time() - start_time > timeout - 5.0:
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
