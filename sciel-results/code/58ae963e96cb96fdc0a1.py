"""
Second autocorrelation inequality: maximize ratio of squared L2 norm to L1 norm 
times supremum norm of autoconvolution.

Combined strategy:
1. Generate diverse structured initializations (Sidon-like, alternating, smooth)
2. Global epigraph SLSQP optimization (allows support to adapt/merge)
3. Coordinate descent refinement on the best solutions
4. Final polish with tight tolerances
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]
NUM_CONV = 2 * N - 1

def _autoconv(h):
    """Linear autoconvolution of h with itself."""
    return np.convolve(h, h, mode='full')

def _score_raw(h):
    """Raw score: L2^2 / (L1 * DELTA * max(autoconv))."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    c = _autoconv(h)
    max_c = np.max(c)
    if max_c <= 1e-15:
        return 0.0
    return l2_sq / (l1 * DELTA * max_c)

def _find_sidon_support(target_size):
    """
    Greedy construction of a set with minimal pairwise sum collisions.
    """
    if target_size <= 0:
        return []
    
    support = [0]
    used_sums = {0: 1}
    
    for _ in range(target_size - 1):
        best_candidate = None
        best_score = -1
        
        for candidate in range(N):
            if candidate in support:
                continue
            
            new_collisions = 0
            for s in support:
                pair_sum = candidate + s
                if pair_sum in used_sums:
                    new_collisions += 1
            
            score = -new_collisions
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        if best_candidate is not None:
            support.append(best_candidate)
            for s in support:
                pair_sum = best_candidate + s
                used_sums[pair_sum] = used_sums.get(pair_sum, 0) + 1
        else:
            break
            
    return support

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)

    def update_best(h):
        nonlocal best_h, best_score
        if h is None:
            return
        h = np.asarray(h, dtype=np.float64)
        h = np.maximum(0, h)
        s = np.sum(h)
        if s <= 0:
            return
        h = h / s
        sc = _score_raw(h)
        if sc > best_score:
            best_score = sc
            best_h = h.copy()

    # --- Phase 1: Generate diverse structured initializations ---
    inits = []

    # 1. Uniform
    inits.append(np.ones(N) / N)

    # 2. Sidon-like sparse supports
    np.random.seed(42)
    for K in range(3, 15):
        support = _find_sidon_support(K)
        if len(support) >= 3:
            h_s = np.zeros(N)
            h_s[support] = 1.0
            h_s /= np.sum(h_s)
            inits.append(h_s)

    # 3. Random sparse
    for _ in range(20):
        k = np.random.randint(3, 12)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)

    # 4. Alternating patterns (helps reduce auto-correlation peaks)
    for start in range(N):
        for gap in [2, 3, 4, 5]:
            h = np.zeros(N)
            for i in range(start, N, gap):
                h[i] = 1.0
            if np.sum(h) > 0:
                h /= np.sum(h)
                inits.append(h)

    # 5. Concentrated Gaussians
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(0.5, N/3, 5):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)

    # 6. Triangular
    for c in np.linspace(3, N-3, 6):
        for w in np.linspace(2, N, 4):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                h_t /= np.sum(h_t)
                inits.append(h_t)

    # 7. Random smooth
    for _ in range(15):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 1e-10:
            h_r /= np.sum(h_r)
            inits.append(h_r)

    # --- Phase 2: Global Epigraph SLSQP Optimization ---
    # Variables: x = [h_0, ..., h_{N-1}, M]
    # Objective: Minimize M (x[N])
    # Constraints:
    # 1. sum(h) - 1 == 0
    # 2. h_i >= 0 (via bounds)
    # 3. M - conv_k >= 0 for all k

    def objective(x):
        return x[N]

    def objective_grad(x):
        grad = np.zeros(N + 1)
        grad[N] = 1.0
        return grad

    def constraint_eq_fun(x):
        return np.sum(x[:N]) - 1.0

    def constraint_eq_jac(x):
        jac = np.zeros(N + 1)
        jac[:N] = 1.0
        return jac

    def constraint_ineq_fun(x):
        h = x[:N]
        M = x[N]
        conv = _autoconv(h)
        return M - conv

    def constraint_ineq_jac(x):
        h = x[:N]
        jac = np.zeros((NUM_CONV, N + 1))
        jac[:, N] = 1.0
        for j in range(N):
            # Part 1: k-j is in [0, N-1] => k in [j, j+N-1]
            k_start = j
            k_end = min(j + N, NUM_CONV)
            if k_start < NUM_CONV and k_end > 0:
                indices_k = np.arange(k_start, k_end)
                indices_h = indices_k - j
                jac[indices_k, j] -= h[indices_h]

            # Part 2: j-k is in [0, N-1] => k in [j-N+1, j]
            k_start2 = max(0, j - N + 1)
            k_end2 = min(j, NUM_CONV)
            if k_start2 < NUM_CONV and k_end2 >= 0:
                indices_k = np.arange(k_start2, k_end2 + 1)
                indices_h = j - indices_k
                jac[indices_k, j] -= h[indices_h]

        return jac

    constraints = [
        {
            'type': 'eq',
            'fun': constraint_eq_fun,
            'jac': constraint_eq_jac
        },
        {
            'type': 'ineq',
            'fun': constraint_ineq_fun,
            'jac': constraint_ineq_jac
        }
    ]

    bounds = [(0.0, 1.0)] * N + [(0.0, None)]

    def run_epigraph(h_init, maxiter=300, ftol=1e-12):
        if np.sum(h_init) < 1e-15:
            return None
        h_init = h_init / np.sum(h_init)
        conv_init = _autoconv(h_init)
        M_init = np.max(conv_init) + 1e-6
        x0 = np.concatenate((h_init, [M_init]))

        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                jac=objective_grad,
                options={'maxiter': maxiter, 'ftol': ftol}
            )

            if res.success:
                h_candidate = res.x[:N]
                h_candidate = np.clip(h_candidate, 0, None)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate /= np.sum(h_candidate)
                    return h_candidate
        except Exception:
            pass
        return None

    # Sort inits by initial score (descending) to try best ones first
    scored_inits = []
    for h_init in inits:
        s = _score_raw(h_init)
        scored_inits.append((s, h_init))
    scored_inits.sort(key=lambda x: -x[0])

    # Run epigraph from top candidates
    max_epigraph_runs = min(len(scored_inits), 25)
    for i in range(max_epigraph_runs):
        if time.time() > deadline - 50:
            break
        s_init, h_init = scored_inits[i]
        h_candidate = run_epigraph(h_init, maxiter=200, ftol=1e-12)
        if h_candidate is not None:
            update_best(h_candidate)

    # --- Phase 3: Coordinate Descent Refinement ---
    def refine_local(h_start, max_iter=30):
        h = h_start.copy()
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            return h
        h /= np.sum(h)

        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time.time() > deadline - 15:
                break
            improved = False
            iter_count += 1

            conv = _autoconv(h)
            M_val = np.max(conv)
            l1 = np.sum(h)
            l2_sq = np.sum(h**2)
            current_score = l2_sq / (l1 * DELTA * M_val)

            # Try moving mass from high-impact positions to low-impact ones
            for i in range(N):
                if time.time() > deadline - 15:
                    break
                h_old = h[i]
                for delta_val in [0.01, -0.01, 0.001, -0.001, 0.05, -0.05]:
                    h_new = h.copy()
                    h_new[i] = max(0, h_old + delta_val)
                    if np.sum(h_new) < 1e-15:
                        continue
                    h_new /= np.sum(h_new)
                    s_new = _score_raw(h_new)
                    if s_new > current_score:
                        h = h_new
                        current_score = s_new
                        improved = True

        return h

    # Apply local refinement to the best solution
    if time.time() < deadline - 15:
        h_refined = refine_local(best_h, max_iter=30)
        update_best(h_refined)

    # --- Phase 4: Final Polish with Tight Tolerances ---
    if time.time() < deadline - 5:
        h_polished = run_epigraph(best_h, maxiter=500, ftol=1e-14)
        if h_polished is not None:
            update_best(h_polished)

    # Ensure valid output
    best_h = np.maximum(0, best_h)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0
        
    return list(best_h)
