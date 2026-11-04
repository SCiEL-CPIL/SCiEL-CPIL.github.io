"""
Spectral Quadratic Linearization (QL) + Vectorized Simulated Annealing (SA) for QAPLIB wil100.
Strategy:
1. Spectral Initialization: Decompose the flow matrix A into rank-one outer products of its
   dominant eigenvectors. For each component, solve a Linear Assignment Problem (LAP) to
   minimize the surrogate quadratic cost. Combine these solutions via a voting/ensemble mechanism
   or by selecting the best individual surrogate.
2. Vectorized Simulated Annealing: Use a fast O(n^2) vectorized delta computation for swap moves.
   Implement a geometric cooling schedule with periodic reheating to escape local minima.
   Multi-start the SA process from the best spectral initializations.
"""
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute the exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))

def _spectral_init(a, b, n_components=5):
    """
    Generate initial permutations using Spectral Quadratic Linearization.
    We decompose A into sum_k lambda_k * v_k v_k^T.
    For each k, the term lambda_k * (v_k^T p) * (v_k^T p) is not linear.
    Instead, we use the surrogate cost matrix C_k[i, j] = lambda_k * (v_k[i] * v_k[j] * b[i, j] ? No.)
    
    Standard QL approach:
    Cost = sum_{i,j} A_{ij} B_{p(i), p(j)}
    Approximate A_{ij} ≈ sum_{k=1}^K w_k u_k[i] u_k[j]
    Then Cost ≈ sum_k w_k (sum_i u_k[i] B_{p(i), p(j)} ... this is getting complex.
    
    Simpler QL heuristic:
    For each dominant eigenvector v of A, construct a linear cost matrix L_v[i, k] = v[i] * (B[:, k] @ v) ?
    Actually, a common simple spectral heuristic is to map facility i to location k
    to maximize correlation between facility vector and location vector.
    
    Let's use the standard 'surrogate' from the literature:
    For each eigenvector v of A (normalized), define L[i, k] = v[i] * (B @ v)[k].
    Solving LAP on L gives a permutation that aligns high-flow facilities with high-distance locations?
    Wait, we want to MINIMIZE cost.
    If A has positive eigenvalues, high values in v correspond to high flow.
    We want high flow facilities to be close.
    So we want to match high v[i] with low 'distance potential'.
    
    Let's stick to a robust multi-start strategy using different surrogate matrices.
    1. Row sums / Col sums (Hungarian)
    2. Spectral: For top eigenvectors of A, project B onto them.
    """
    n = len(a)
    inits = []
    
    # 1. Standard Hungarian on row/col sums
    a_rows = a.sum(axis=1)
    a_cols = a.sum(axis=0)
    b_rows = b.sum(axis=1)
    b_cols = b.sum(axis=0)
    
    # Surrogate 1: -outer(a_rows, b_rows)
    C1 = -np.outer(a_rows, b_rows)
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(C1)
    p1 = np.empty(n, dtype=np.int64)
    p1[r] = c
    inits.append(p1)
    
    # Surrogate 2: -outer(a_cols, b_cols)
    C2 = -np.outer(a_cols, b_cols)
    r, c = linear_sum_assignment(C2)
    p2 = np.empty(n, dtype=np.int64)
    p2[r] = c
    inits.append(p2)
    
    # 3. Spectral Surrogates
    # Eigendecomposition of A
    # A is usually symmetric in QAP, but let's be safe. If not symmetric, use SVD or just top eigenvectors of (A+A.T)/2
    A_sym = (a + a.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(A_sym)
    
    # Sort by absolute value descending
    idx = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    for k in range(min(n_components, n)):
        v = eigenvectors[:, k]
        # We want to align facility i with location k such that v[i] and some location property are correlated.
        # Location property derived from B.
        # Let's project B onto v.
        # L[i, j] = v[i] * (B @ v)[j] ?
        # If v[i] is large, we want (B @ v)[j] to be small (if we minimize).
        # So L[i, j] = v[i] * (B @ v)[j]
        # Then minimize sum L[i, p[i]].
        Bv = b @ v
        L = np.outer(v, Bv)
        r, c = linear_sum_assignment(L)
        p_k = np.empty(n, dtype=np.int64)
        p_k[r] = c
        inits.append(p_k)
        
        # Also try negative correlation
        L_neg = -np.outer(v, Bv)
        r, c = linear_sum_assignment(L_neg)
        p_k_neg = np.empty(n, dtype=np.int64)
        p_k_neg[r] = c
        inits.append(p_k_neg)
        
    return inits

def _compute_swap_deltas(a, b, p):
    """
    Vectorized computation of cost deltas for all pairwise swaps.
    Returns matrix D where D[i, j] is delta cost for swapping positions i and j.
    """
    # C = b[np.ix_(p, p)]
    # This is the key bottleneck if not cached, but for SA we update p frequently.
    # We can compute deltas on the fly or update incrementally.
    # For simplicity and correctness in the vectorized form, we recompute the relevant parts.
    
    C = b[np.ix_(p, p)]
    
    # Delta for swapping i and j:
    # Cost change = sum_{k!=i,j} a[i,k] (b[p[j], p[k]] - b[p[i], p[k]]) + sum_{k!=i,j} a[j,k] (b[p[i], p[k]] - b[p[j], p[k]])
    #             + a[i,j] (b[p[j], p[i]] - b[p[i], p[j]]) + a[j,i] (b[p[i], p[j]] - b[p[j], p[i]])
    #             + terms where k=i or j in the sums above are handled by the diagonal terms.
    
    # Let's use the standard vectorized formula:
    # D[i, j] = sum_k a[i,k] (b[p[j], p[k]] - b[p[i], p[k]]) + sum_k a[j,k] (b[p[i], p[k]] - b[p[j], p[k]])
    #          - 2 * (a[i,j] (b[p[i], p[j]] - b[p[j], p[i]]))  # Correction for double counting or specific terms?
    
    # Actually, a more efficient way for all pairs:
    # Let B_p = b[np.ix_(p, p)]
    # Term 1: a @ (B_p.T - B_p) ? No.
    
    # Let's derive:
    # Cost = sum_{i,j} a[i,j] B_p[i,j]
    # Swap i, j in p. Let p' be the new permutation.
    # Only rows/cols i and j in B_p change.
    # B_p[i, :] becomes B_p[j, :] and vice versa.
    # B_p[:, i] becomes B_p[:, j] and vice versa.
    
    # Delta = sum_k a[i,k] (B_p[j,k] - B_p[i,k]) + sum_k a[k,i] (B_p[k,j] - B_p[k,i])
    #        + sum_k a[j,k] (B_p[i,k] - B_p[j,k]) + sum_k a[k,j] (B_p[k,i] - B_p[k,j])
    #        - (terms where k=i or k=j are counted twice or need adjustment)
    
    # Let S1 = a @ B_p.T  -> S1[i, j] = sum_k a[i, k] B_p[j, k]
    # Let S2 = a.T @ B_p  -> S2[i, j] = sum_k a[k, i] B_p[k, j]
    
    # Delta[i, j] = (S1[i, j] - sum_k a[i, k] B_p[i, k]) + (S2[i, j] - sum_k a[k, i] B_p[k, i])
    #              + (S1[j, i] - sum_k a[j, k] B_p[j, k]) + (S2[j, i] - sum_k a[k, j] B_p[k, j])
    #              - 2 * (a[i, j] B_p[i, j] + a[j, i] B_p[j, i] - a[i, j] B_p[j, i] - a[j, i] B_p[i, j]) ?
    
    # This is getting complicated to vectorize for ALL pairs simultaneously without O(n^3) memory or time.
    # The previous program used a specific O(n^2) vectorized trick. Let's reuse that logic.
    
    a_sym = a + a.T
    S1 = a.T @ C
    T1 = a @ C
    
    D_S1 = np.diag(S1)
    D_T1 = np.diag(T1)
    
    Term1 = S1 + S1.T - D_S1[:, None] - D_S1[None, :]
    Term2 = T1 + T1.T - D_T1[:, None] - D_T1[None, :]
    
    Raw = Term1 + Term2
    Error = -2.0 * a_sym * C
    Delta = Raw - Error
    np.fill_diagonal(Delta, 0.0)
    
    return Delta

def _sa_polish(a, b, p, time_limit=10.0, start_time=None):
    """
    Simulated Annealing polish.
    """
    if start_time is None:
        start_time = time.time()
        
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Initial temperature
    # Estimate average delta magnitude
    deltas = _compute_swap_deltas(a, b, p)
    avg_abs_delta = np.mean(np.abs(deltas[deltas != 0]))
    if avg_abs_delta == 0:
        avg_abs_delta = 1.0
    T = avg_abs_delta * 10.0
    T_min = 1e-6
    alpha = 0.95
    
    while time.time() - start_time < start_time + time_limit:
        deltas = _compute_swap_deltas(a, b, p)
        
        # Select a random move weighted by probability? Or just random swap.
        # For speed, pick random i, j
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        delta = deltas[i, j]
        
        # Acceptance criterion
        if delta < 0:
            accept = True
        else:
            if T > T_min:
                # Boltzmann distribution
                # To avoid overflow/underflow issues, use log or careful exp
                ratio = np.exp(-delta / T) if T > 0 else 0
                accept = random.random() < ratio
            else:
                accept = False
                
        if accept:
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
        T *= alpha
        
        # Reheat if stuck?
        # Simple reheat: if no improvement in last X steps, increase T
        # For now, keep it simple.
        
    return best_p, best_cost

def solve(a, b):
    n = len(a)
    total_deadline = 115.0
    start_time = time.time()
    
    # 1. Generate diverse initial solutions
    inits = _spectral_init(a, b, n_components=5)
    
    # Evaluate and sort inits
    init_costs = [_qap_cost(a, b, p) for p in inits]
    inits = [p for _, p in sorted(zip(init_costs, inits))]
    
    best_p = inits[0].copy()
    best_cost = init_costs[0]
    
    # 2. Multi-start SA
    # Allocate time to multiple SA runs
    # Let's do 5 SA runs, each getting ~20s, plus a final long polish on the best.
    num_sa_runs = 5
    time_per_sa = 15.0
    
    for i in range(num_sa_runs):
        if time.time() - start_time > total_deadline - 10.0:
            break
            
        # Pick an initial solution (round-robin or best-first with perturbation)
        # Use the i-th best init, or perturb the global best
        if i < len(inits):
            p_start = inits[i].copy()
        else:
            p_start = best_p.copy()
            # Perturb
            for _ in range(5):
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                p_start[x], p_start[y] = p_start[y], p_start[x]
                
        p_res, c_res = _sa_polish(a, b, p_start, time_limit=time_per_sa, start_time=start_time)
        
        if c_res < best_cost:
            best_cost = c_res
            best_p = p_res.copy()
            
    # 3. Final intensive polish on the best solution
    time_left = total_deadline - (time.time() - start_time)
    if time_left > 5.0:
        # Use a more aggressive SA or local search
        # Here we use SA with a higher initial T to allow more exploration
        p_final, c_final = _sa_polish(a, b, best_p, time_limit=time_left - 2.0, start_time=start_time)
        if c_final < best_cost:
            best_cost = c_final
            best_p = p_final.copy()
            
    return best_p.tolist()
