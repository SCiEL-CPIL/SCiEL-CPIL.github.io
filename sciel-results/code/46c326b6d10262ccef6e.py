"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve(a, b):
    n = len(a)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # Pre-compute some properties if needed, but a and b are fixed.
    
    def calc_cost(p):
        # p is permutation of 0..n-1
        # Cost = sum a[i,j] * b[p[i], p[j]]
        # We can do this as np.sum(a * b[np.ix_(p, p)])
        b_perm = b[np.ix_(p, p)]
        return int(np.sum(a * b_perm))

    def get_delta(p, i, j):
        """
        Calculate delta cost for swapping facilities i and j in permutation p.
        p[i] -> u, p[j] -> v.
        """
        u = p[i]
        v = p[j]
        
        # We need: 2 * sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]])
        # Let diff_a = a[i] - a[j]
        # Let diff_b_k = b[v, p[k]] - b[u, p[k]]
        
        # Vectorized calculation
        # p is array of locations.
        # b_v = b[v, p] gives array of length n: b[v, p[0]], b[v, p[1]], ...
        # b_u = b[u, p]
        
        b_v_p = b[v, p]
        b_u_p = b[u, p]
        diff_b = b_v_p - b_u_p
        
        diff_a = a[i] - a[j]
        
        # Total sum over all k
        sum_all = np.sum(diff_a * diff_b)
        
        # Subtract terms for k=i and k=j
        # Term k=i: diff_a[i] * diff_b[i]
        # diff_b[i] = b[v, p[i]] - b[u, p[i]] = b[v, u] - b[u, u]
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        
        # Term k=j: diff_a[j] * diff_b[j]
        # diff_b[j] = b[v, p[j]] - b[u, p[j]] = b[v, v] - b[u, v]
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        delta = 2 * (sum_all - term_i - term_j)
        return delta

    def local_search(p, current_cost, first_pass=True):
        # First pass: Best Improvement
        # Subsequent passes: First Improvement (much faster, allows more passes)
        improved = True
        pass_count = 0
        while improved:
            improved = False
            best_delta = 0
            best_i = -1
            best_j = -1
            found_improvement = False
            
            if first_pass and pass_count == 0:
                # Best improvement for first pass
                for i in range(n):
                    for j in range(i + 1, n):
                        delta = get_delta(p, i, j)
                        if delta < best_delta:
                            best_delta = delta
                            best_i = i
                            best_j = j
                
                if best_delta < 0:
                    p[best_i], p[best_j] = p[best_j], p[best_i]
                    current_cost += best_delta
                    improved = True
                    pass_count += 1
            else:
                # First improvement for subsequent passes
                for i in range(n):
                    for j in range(i + 1, n):
                        delta = get_delta(p, i, j)
                        if delta < 0:
                            p[i], p[j] = p[j], p[i]
                            current_cost += delta
                            improved = True
                            found_improvement = True
                            pass_count += 1
                            break
                    if found_improvement:
                        break
                if not found_improvement:
                    improved = False
                    
        return p, current_cost

    start_time = time.time()
    timeout = 115.0
    
    # 1. Initial Solution: Rank Matching
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
    
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # 2. Local Search on initial solution
    p, current_cost = local_search(p, current_cost, first_pass=True)
    if current_cost < best_cost:
        best_cost = current_cost
        best_p = p.copy()
    
    # 3. ILS / SA Loop
    # We will use a simple ILS with random kicks.
    # To diversify, we also try restarting from random permutations occasionally.
    
    p = best_p.copy()
    current_cost = best_cost
    
    while time.time() - start_time < timeout:
        # Perturbation
        # Number of swaps: usually 5-10
        num_swaps = 5
        for _ in range(num_swaps):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            delta = get_delta(p, i, j)
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
        # Local Search
        p, current_cost = local_search(p, current_cost, first_pass=False)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        # Occasionally, restart from a new random solution to escape deep local optima
        # This is risky if we run out of time, but ILS can benefit from it.
        # Let's check time. If we have > 20s left, do a random restart.
        if time.time() - start_time > 20.0:
             # Don't restart too frequently, but if stuck for a while?
             # Simple strategy: if no improvement for last 10 iterations, restart?
             # For now, stick to ILS on current best, but maybe perturb best_p directly?
             # The above loop perturbs p, which is currently the result of the last LS.
             # If it didn't improve best, p is worse than best_p.
             # It's better to keep p close to best_p.
             # Let's reset p to best_p and perturb that.
             p = best_p.copy()
             current_cost = best_cost
             # And then perturb p
             num_swaps = 5
             for _ in range(num_swaps):
                i = np.random.randint(n)
                j = np.random.randint(n)
                if i == j: continue
                delta = get_delta(p, i, j)
                p[i], p[j] = p[j], p[i]
                current_cost += delta
             # And loop continues to LS.
             # This ensures we always start LS from a perturbed version of the global best.
             # This is standard ILS.
             
    return best_p.tolist()
# EVOLVE-BLOCK-END
