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
    # Cost = sum(a[i, j] * b[p[i], p[j]])
    # Efficient calculation:
    # C = trace(A * B_perm) where B_perm[i, j] = B[p[i], p[j]]
    # Or simply np.sum(a * b[np.ix_(p, p)])
    current_cost = np.sum(a * b[np.ix_(p, p)])
    best_cost = current_cost
    best_p = p.copy()
    
    # Simulated Annealing parameters
    # n=150, typical QAP instances
    # Start temp high enough to accept most bad moves, cool down
    T = 1000.0 
    cooling = 0.9995 # Slow cooling to allow more exploration
    min_T = 1.0
    
    # Precompute row sums? No, we use the O(n) delta formula.
    
    # Helper to compute delta cost for swapping i and j
    def delta_swap(i, j, p, a, b):
        u = p[i]
        v = p[j]
        if u == v:
            return 0
        
        # D_k = B[v, p_k] - B[u, p_k]
        D = b[v, p] - b[u, p]
        
        # Sum over k != i, j of (A[i, k] - A[j, k]) * D[k]
        # First compute full sum
        diff_a = a[i] - a[j]
        full_sum = np.dot(diff_a, D)
        
        # Subtract k=i term
        # (A[i, i] - A[j, i]) * D[i] = (A[i, i] - A[j, i]) * (B[v, p_i] - B[u, p_i])
        # p_i = u
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        
        # Subtract k=j term
        # (A[i, j] - A[j, j]) * D[j] = (A[i, j] - A[j, j]) * (B[v, p_j] - B[u, p_j])
        # p_j = v
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        
        sum_k_neq = full_sum - term_i - term_j
        
        # Terms involving i, i and j, j
        term_ii = a[i, i] * (b[v, v] - b[u, u])
        term_jj = a[j, j] * (b[u, u] - b[v, v])
        
        # Terms involving i, j and j, i
        term_ij = a[i, j] * (b[v, u] - b[u, v])
        term_ji = a[j, i] * (b[u, v] - b[v, u])
        
        return sum_k_neq + term_ii + term_jj + term_ij + term_ji

    # Main SA Loop
    while time.time() < deadline:
        if T < min_T:
            break
            
        # Generate random swap
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        
        if i == j:
            continue
            
        dc = delta_swap(i, j, p, a, b)
        
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
        
        # Optional: Periodic reheat or restart if stuck?
        # For simplicity, keep cooling.
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
