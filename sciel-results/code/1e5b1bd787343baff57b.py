"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _build_cost_matrix(a, b, p):
    """Build cost matrix C[i,l] = sum_j a[i,j] * b[l, p[j]]."""
    V = b[:, p]  # V[l,j] = b[l, p[j]]
    C = a @ V.T
    return C


def _swap_delta_fast(C, a, b, p, i, j):
    """Compute swap delta using precomputed cost matrix C."""
    pi = p[i]
    pj = p[j]
    
    # Main part from cost matrix
    delta_approx = C[i, pj] - C[i, pi] + C[j, pi] - C[j, pj]
    
    # Correction for diagonal and cross terms
    correction = (a[i, j] * (b[pj, pi] - b[pj, pj]) +
                  a[j, i] * (b[pi, pj] - b[pi, pi]) -
                  a[i, i] * (b[pj, pi] - b[pi, pi]) -
                  a[j, j] * (b[pi, pj] - b[pj, pj]))
    
    return int(delta_approx + correction)


def _update_cost_matrix(C, a, b, p, i, j):
    """Update cost matrix C after swapping p[i] and p[j]."""
    pi = p[i]
    pj = p[j]
    
    # C[k, l] += (a[k,i] - a[k,j]) * (b[l,pj] - b[l,pi])
    C += np.outer(a[:, i] - a[:, j], b[:, pj] - b[:, pi])
    
    # Swap in p
    p[i], p[j] = p[j], p[i]


def _local_search_steepest(C, a, b, p, current_cost, deadline, max_passes=100):
    """Steepest descent local search using precomputed cost matrix."""
    n = len(p)
    for _ in range(max_passes):
        if time.time() >= deadline - 1:
            break
        best_d = 0
        best_i, best_j = -1, -1
        
        # Find the best swap
        for i in range(n):
            pi = p[i]
            for j in range(i + 1, n):
                pj = p[j]
                d = _swap_delta_fast(C, a, b, p, i, j)
                if d < best_d:
                    best_d = d
                    best_i, best_j = i, j
        
        if best_i == -1:
            break
            
        _update_cost_matrix(C, a, b, p, best_i, best_j)
        current_cost += best_d
        
    return p, current_cost, C


def _perturb(p, k, rng):
    """Apply k random swaps to p."""
    p_new = p.copy()
    n = len(p_new)
    for _ in range(k):
        i, j = rng.integers(0, n, size=2)
        if i != j:
            p_new[i], p_new[j] = p_new[j], p_new[i]
    return p_new


def _lap_initial(a, b, n):
    """Get initial assignment using LAP on linearized cost."""
    C = a @ b
    row_ind, col_ind = linear_sum_assignment(C)
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p


def solve(a, b):
    n = len(a)
    deadline = time.time() + 115
    rng = np.random.default_rng(42)
    
    # Initial assignment
    p0 = _lap_initial(a, b, n)
    best_p = p0.copy()
    best_cost = _qap_cost(a, b, best_p)
    
    # Also try rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    cost_rank = _qap_cost(a, b, p_rank)
    if cost_rank < best_cost:
        best_cost = cost_rank
        best_p = p_rank.copy()
        
    # Build initial cost matrix
    C = _build_cost_matrix(a, b, best_p)
    
    # ILS parameters
    T = 1000.0
    cooling = 0.999
    min_T = 1.0
    
    current_p = best_p.copy()
    current_cost = best_cost
    current_C = C.copy()
    
    while time.time() < deadline - 5:
        # 1. Local Search
        current_p, current_cost, current_C = _local_search_steepest(
            current_C, a, b, current_p, current_cost, deadline, max_passes=50
        )
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = current_p.copy()
            
        # 2. Perturbation (The "Quantum" Kick)
        # Vary kick size slightly to explore different basin scales
        kick_size = rng.integers(3, 10)
        perturbed_p = _perturb(current_p, kick_size, rng)
        
        # Calculate cost of perturbed solution
        perturbed_cost = _qap_cost(a, b, perturbed_p)
        
        # 3. Acceptance
        delta = perturbed_cost - current_cost
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_p = perturbed_p
            current_cost = perturbed_cost
            current_C = _build_cost_matrix(a, b, current_p)
        
        T *= cooling
        if T < min_T:
            T = min_T

    # Final polish
    final_p, final_cost, _ = _local_search_steepest(
        _build_cost_matrix(a, b, best_p), a, b, best_p, best_cost, deadline, max_passes=50
    )
    if final_cost < best_cost:
        best_cost = final_cost
        best_p = final_p
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
