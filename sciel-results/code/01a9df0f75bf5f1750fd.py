import numpy as np
import time

def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))

def _swap_delta(a, b, p, i, j):
    pi = p[i]
    pj = p[j]
    # Vectorized delta calculation
    bpj = b[:, pj][p]
    bpi = b[:, pi][p]
    brj = b[pj, :][p]
    bri = b[pi, :][p]
    
    delta = (a[:, i] * (bpj - bpi) +
             a[:, j] * (bpi - bpj) +
             a[i, :] * (brj - bri) +
             a[j, :] * (bri - brj))
    delta[i] = 0
    delta[j] = 0
    direct = a[i, j] * (b[pj, pi] - b[pi, pj]) + a[j, i] * (b[pi, pj] - b[pj, pi])
    return int(np.sum(delta) + direct)

def _local_search_2opt(a, b, p, current_cost, deadline, max_passes=100):
    n = len(p)
    for _ in range(max_passes):
        if time.time() >= deadline - 1:
            break
        best_d = 0
        best_i, best_j = -1, -1
        for i in range(n):
            if time.time() >= deadline - 1:
                break
            for j in range(i + 1, n):
                d = _swap_delta(a, b, p, i, j)
                if d < best_d:
                    best_d = d
                    best_i, best_j = i, j
        if best_i == -1:
            break
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_d
    return p, current_cost

def _spectral_init(a, b, k=10, rng=None):
    """
    Generate initial permutation using PCA on flow and distance matrices.
    Projects both matrices onto their top-k principal components and
    matches facilities to locations based on proximity in the reduced space.
    """
    n = a.shape[0]
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Center the matrices
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    
    # Compute PCA components
    try:
        # Use SVD for stability
        _, _, Vt_a = np.linalg.svd(a_centered, full_matrices=False)
        _, _, Vt_b = np.linalg.svd(b_centered, full_matrices=False)
        
        # Project onto top-k components
        U_a = Vt_a[:k, :].T  # n x k
        U_b = Vt_b[:k, :].T  # n x k
        
        # Normalize projections
        U_a_norm = np.linalg.norm(U_a, axis=1, keepdims=True)
        U_b_norm = np.linalg.norm(U_b, axis=1, keepdims=True)
        U_a = U_a / (U_a_norm + 1e-10)
        U_b = U_b / (U_b_norm + 1e-10)
        
        # Compute distance matrix in reduced space
        # dist[i,j] = ||U_a[i] - U_b[j]||^2
        # = ||U_a[i]||^2 + ||U_b[j]||^2 - 2*U_a[i]@U_b[j]
        Ua_sq = np.sum(U_a**2, axis=1)
        Ub_sq = np.sum(U_b**2, axis=1)
        cross_term = U_a @ U_b.T
        dist_matrix = Ua_sq[:, None] + Ub_sq[None, :] - 2 * cross_term
        
        # Find optimal assignment using greedy or Hungarian
        # For efficiency, use a simple greedy approach
        p = np.empty(n, dtype=np.int64)
        used = np.zeros(n, dtype=bool)
        
        # Sort facilities by their "importance" (row sum of a)
        facility_order = np.argsort(-a.sum(axis=1), kind='stable')
        
        for fac in facility_order:
            # Find best available location
            dist_row = dist_matrix[fac].copy()
            dist_row[used] = np.inf
            best_loc = np.argmin(dist_row)
            p[fac] = best_loc
            used[best_loc] = True
        
        return p
        
    except Exception:
        # Fallback to rank matching
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p

def solve(a, b):
    n = len(a)
    deadline = time.time() + 115
    rng = np.random.default_rng(42)
    
    # 1. Multi-Start Initialization with Spectral Projection
    # Generate diverse initial solutions using spectral methods
    num_starts = 3
    best_p = None
    best_cost = np.inf
    
    for start in range(num_starts):
        if time.time() >= deadline - 20:
            break
        
        # Generate initial permutation using spectral method
        if start == 0:
            p = _spectral_init(a, b, k=10, rng=rng)
        elif start == 1:
            p = _spectral_init(a, b, k=5, rng=rng)
        else:
            # Fallback to rank matching for diversity
            heavy = np.argsort(-a.sum(axis=1), kind="stable")
            central = np.argsort(b.sum(axis=1), kind="stable")
            p = np.empty(n, dtype=np.int64)
            p[heavy] = central
        
        # Initial Local Search
        p, cost = _local_search_2opt(a, b, p, _qap_cost(a, b, p), deadline)
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    # 2. Intensive Local Search (ILS) with Perturbation
    stagnation = 0
    perturbation_strength = 5
    
    while time.time() < deadline - 5:
        # Perturb the current best solution
        p_curr = best_p.copy()
        cost_curr = best_cost
        
        for _ in range(perturbation_strength):
            i, j = rng.integers(0, n, size=2)
            if i != j:
                d = _swap_delta(a, b, p_curr, i, j)
                p_curr[i], p_curr[j] = p_curr[j], p_curr[i]
                cost_curr += d
        
        # Local Search on perturbed solution
        p_curr, cost_curr = _local_search_2opt(a, b, p_curr, cost_curr, deadline)
        
        if cost_curr < best_cost:
            best_cost = cost_curr
            best_p = p_curr.copy()
            stagnation = 0
            # Optionally reduce perturbation strength if improving
            if perturbation_strength > 2:
                perturbation_strength = max(2, perturbation_strength - 1)
        else:
            stagnation += 1
            # Increase perturbation strength if stuck
            if stagnation % 10 == 0:
                perturbation_strength = min(50, perturbation_strength + 5)
    
    # 3. Final Polish
    p_final = best_p.copy()
    cost_final = best_cost
    p_final, cost_final = _local_search_2opt(a, b, p_final, cost_final, deadline)
    
    if cost_final < best_cost:
        best_p = p_final
        best_cost = cost_final

    return best_p.tolist()
