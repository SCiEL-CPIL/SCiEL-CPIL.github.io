"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def solve(a, b):
    n = len(a)
    # Deadline guard: 115 seconds to be safe
    deadline = time.time() + 115.0
    
    # 1. Initialization: Rank-matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost
    current_cost = np.sum(a * b[np.ix_(p, p)])
    best_cost = current_cost
    best_p = p.copy()
    
    # Precompute for faster delta computation
    # row_a[i] = sum of a[i, :] (not directly needed but useful)
    # We'll use the O(n) delta formula with precomputed row differences
    
    def delta_swap_fast(i, j, p, a, b):
        """Compute delta cost for swapping positions i and j in permutation p."""
        u = p[i]
        v = p[j]
        
        # D_k = B[v, p_k] - B[u, p_k] for all k
        D = b[v, p] - b[u, p]
        
        # diff_a[k] = A[i, k] - A[j, k]
        diff_a = a[i] - a[j]
        
        # Full dot product
        full_sum = np.dot(diff_a, D)
        
        # Subtract terms where k=i and k=j
        # When k=i: p[k] = u, so D[i] = B[v, u] - B[u, u]
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        # When k=j: p[k] = v, so D[j] = B[v, v] - B[u, v]  
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        sum_k_neq = full_sum - term_i - term_j
        
        # Diagonal and cross terms
        term_ii = a[i, i] * (b[v, v] - b[u, u])
        term_jj = a[j, j] * (b[u, u] - b[v, v])
        term_ij = a[i, j] * (b[v, u] - b[u, v])
        term_ji = a[j, i] * (b[u, v] - b[v, u])
        
        return sum_k_neq + term_ii + term_jj + term_ij + term_ji
    
    def local_search_2opt(p, a, b, current_cost, deadline, max_iters=200):
        """Steepest-descent 2-opt local search."""
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            best_dc = 0
            best_i = -1
            best_j = -1
            # Sample a subset of pairs to speed up
            num_samples = min(500, n * (n-1) // 2)
            for _ in range(num_samples):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                dc = delta_swap_fast(i, j, p, a, b)
                if dc < best_dc:
                    best_dc = dc
                    best_i = i
                    best_j = j
            if best_i >= 0:
                p[best_i], p[best_j] = p[best_j], p[best_i]
                current_cost += best_dc
                improved = True
        return p, current_cost
    
    # Simulated Annealing with restarts
    T_start = 500.0
    cooling = 0.999
    min_T = 5.0
    restart_interval = 5000  # restart every 5000 iterations
    max_restarts = 10
    
    T = T_start
    iters = 0
    restarts = 0
    
    while time.time() < deadline - 10.0:  # Leave 10s for final polish
        if T < min_T:
            # Reheat and restart
            restarts += 1
            if restarts > max_restarts:
                break
            # Reset to best solution and reheat
            p = best_p.copy()
            current_cost = best_cost
            T = T_start * 0.5
            iters = 0
            continue
        
        # Generate random swap
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        
        if i == j:
            continue
            
        dc = delta_swap_fast(i, j, p, a, b)
        
        # Acceptance criterion
        if dc < 0 or random.random() < math.exp(-dc / T):
            # Perform swap
            p[i], p[j] = p[j], p[i]
            current_cost += dc
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        # Cool down
        T *= cooling
        iters += 1
        
        # Periodic local search to escape plateaus
        if iters % 2000 == 0:
            p, current_cost = local_search_2opt(p, a, b, current_cost, deadline, max_iters=50)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    # Final high-precision polish: multiple passes of steepest-descent 2-opt
    p = best_p.copy()
    current_cost = best_cost
    for _ in range(5):
        if time.time() > deadline - 2.0:
            break
        p, current_cost = local_search_2opt(p, a, b, current_cost, deadline, max_iters=100)
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
