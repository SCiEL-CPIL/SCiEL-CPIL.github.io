import numpy as np
import time
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]
NUM_CONV = 2 * N - 1

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max <= 0:
        return 0.0
    return l2_sq / (l1 * autoconv_max)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)

    def update_best(h):
        nonlocal best_h, best_score
        s = _score_raw(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()

    def time_left():
        return deadline - time.time()

    # === Phase 1: Generate diverse initial supports ===
    # Strategy: For sparse supports, the autoconvolution peak is determined by
    # the maximum number of pairs (i,j) with i+j=k. We want to minimize this.
    # A Sidon set has all pairwise sums distinct, giving peak = 1 (or 2 for symmetric).
    # We generate supports using different heuristics.

    def generate_supports():
        supports = []
        
        # 1. Greedy Sidon-like construction with multiple seeds
        for seed in range(30):
            if time_left() < 60:
                break
            rng = np.random.RandomState(seed)
            for max_size in [4, 6, 8, 10, 12, 14]:
                if time_left() < 55:
                    break
                h = np.zeros(N)
                support = []
                # Start with random anchor
                anchor = rng.randint(N)
                h[anchor] = 1.0
                support.append(anchor)
                
                for _ in range(max_size - 1):
                    if time_left() < 50:
                        break
                    best_idx = -1
                    best_metric = np.inf
                    
                    for idx in range(N):
                        if idx in support:
                            continue
                        h_try = h.copy()
                        h_try[idx] = 1.0
                        conv = np.convolve(h_try, h_try, mode='full')
                        max_conv = np.max(conv)
                        # Also penalize the number of pairs achieving the max
                        num_max = np.sum(conv == max_conv)
                        metric = max_conv + 0.1 * num_max
                        if metric < best_metric:
                            best_metric = metric
                            best_idx = idx
                    
                    if best_idx != -1:
                        h[best_idx] = 1.0
                        support.append(best_idx)
                    else:
                        break
                
                if len(support) >= 3:
                    supports.append(support)
        
        # 2. Random sparse supports
        rng = np.random.RandomState(123)
        for _ in range(40):
            if time_left() < 50:
                break
            k = rng.randint(3, 15)
            idx = rng.choice(N, k, replace=False)
            supports.append(list(idx))
        
        # 3. Structured supports: arithmetic progressions, geometric-like
        for start in range(0, N, 2):
            for step in [2, 3, 4, 5, 7]:
                support = list(range(start, N, step))
                if len(support) >= 3:
                    supports.append(support)
        
        # 4. Complements of structured sets
        for start in range(0, N, 3):
            for step in [3, 4, 5]:
                excluded = set(range(start, N, step))
                support = [i for i in range(N) if i not in excluded]
                if 3 <= len(support) <= 20:
                    supports.append(support)
        
        return supports

    all_supports = generate_supports()
    
    # Deduplicate
    unique_supports = []
    seen = set()
    for s in all_supports:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_supports.append(s)
    
    # Score each support with uniform weights
    def score_support(support):
        h = np.zeros(N)
        h[support] = 1.0
        return _score_raw(h)
    
    scored = [(score_support(s), s) for s in unique_supports]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # === Phase 2: Continuous optimization on top supports ===
    # Use epigraph formulation: minimize M subject to sum(h)=1, h>=0, M >= conv_k
    
    def run_epigraph(support, h_init=None, maxiter=150, ftol=1e-10):
        """Run epigraph optimization for a fixed support."""
        if len(support) < 2:
            return None
        
        n_vars = len(support) + 1
        
        def objective(x):
            return x[-1]
        
        def objective_grad(x):
            grad = np.zeros(n_vars)
            grad[-1] = 1.0
            return grad
        
        def constraint_eq_fun(x):
            return np.sum(x[:-1]) - 1.0
        
        def constraint_eq_jac(x):
            jac = np.zeros(n_vars)
            jac[:-1] = 1.0
            return jac
        
        def constraint_ineq_fun(x):
            w = x[:-1]
            M = x[-1]
            h_full = np.zeros(N)
            h_full[support] = w
            conv = np.convolve(h_full, h_full, mode='full')
            return M - conv
        
        def constraint_ineq_jac(x):
            w = x[:-1]
            h_full = np.zeros(N)
            h_full[support] = w
            jac = np.zeros((NUM_CONV, n_vars))
            jac[:, -1] = 1.0
            for i, s_idx in enumerate(support):
                for k in range(NUM_CONV):
                    j = k - s_idx
                    if 0 <= j < N:
                        # d(conv_k)/dh_i = 2 * h_{k-i} if valid
                        jac[k, i] = -2.0 * h_full[j]
            return jac
        
        constraints = [
            {'type': 'eq', 'fun': constraint_eq_fun, 'jac': constraint_eq_jac},
            {'type': 'ineq', 'fun': constraint_ineq_fun, 'jac': constraint_ineq_jac}
        ]
        
        # Initial guess
        if h_init is not None:
            w_init = h_init[support].copy()
            if np.sum(w_init) < 1e-15:
                w_init = np.ones(len(support))
        else:
            w_init = np.ones(len(support))
        w_init = w_init / np.sum(w_init)
        h_full_init = np.zeros(N)
        h_full_init[support] = w_init
        conv_init = np.convolve(h_full_init, h_full_init, mode='full')
        M_init = np.max(conv_init) + 1e-6
        
        x0 = np.concatenate((w_init, [M_init]))
        bounds = [(0.0, 1.0)] * len(support) + [(0.0, None)]
        
        try:
            res = opt.minimize(
                objective, x0, method='SLSQP', bounds=bounds,
                constraints=constraints, jac=objective_grad,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            if res.success or res.fun < M_init:
                w_opt = res.x[:-1]
                w_opt = np.clip(w_opt, 0, None)
                if np.sum(w_opt) > 1e-15:
                    w_opt /= np.sum(w_opt)
                    h_out = np.zeros(N)
                    h_out[support] = w_opt
                    return h_out
        except Exception:
            pass
        return None

    # Run epigraph on top supports
    max_runs = min(len(scored), 40)
    for i, (s_init, support) in enumerate(scored[:max_runs]):
        if time_left() < 30:
            break
        h_candidate = run_epigraph(support, maxiter=100, ftol=1e-9)
        if h_candidate is not None:
            update_best(h_candidate)
    
    # === Phase 3: Block-coordinate descent refinement ===
    def block_coordinate_refine(h_start, max_iter=15):
        h = h_start.copy()
        h = np.clip(h, 0, None)
        if np.sum(h) < 1e-15:
            return h
        h /= np.sum(h)
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            if time_left() < 10:
                break
            improved = False
            iter_count += 1
            
            conv = np.convolve(h, h, mode='full')
            k_star = np.argmax(conv)
            
            # For each pair (i, k_star - i), try to shift weight
            for i in range(N):
                j = k_star - i
                if j < 0 or j >= N or j == i:
                    continue
                
                hi = h[i]
                hj = h[j]
                if hi + hj < 1e-15:
                    continue
                
                # Try shifting t from i to j
                best_t = 0.0
                best_local = _score_raw(h)
                
                for t in np.linspace(-hi * 0.9, hj * 0.9, 15):
                    h_test = h.copy()
                    h_test[i] += t
                    h_test[j] -= t
                    if np.any(h_test < -1e-12):
                        continue
                    h_test = np.clip(h_test, 0, None)
                    if np.sum(h_test) < 1e-15:
                        continue
                    h_test /= np.sum(h_test)
                    s_test = _score_raw(h_test)
                    if s_test > best_local:
                        best_local = s_test
                        best_t = t
                
                if best_t != 0.0:
                    h[i] += best_t
                    h[j] -= best_t
                    h = np.clip(h, 0, None)
                    h /= np.sum(h)
                    improved = True
        
        return h

    # Refine the best solution
    if best_score > 0:
        h_refined = block_coordinate_refine(best_h, max_iter=10)
        update_best(h_refined)
    
    # === Phase 4: Local search - add/remove/move peaks ===
    if best_score > 0:
        h_curr = best_h.copy()
        support_curr = np.where(h_curr > 1e-6)[0]
        
        for _ in range(25):
            if time_left() < 5:
                break
            
            improved = False
            
            # Try adding a new peak
            zero_idx = np.where(h_curr < 1e-6)[0]
            if len(zero_idx) > 0:
                best_add_idx = -1
                best_add_s = best_score
                for idx in zero_idx[:20]:  # Limit to first 20 for speed
                    h_try = h_curr.copy()
                    h_try[idx] = 0.05
                    s = _score_raw(h_try)
                    if s > best_add_s:
                        best_add_s = s
                        best_add_idx = idx
                
                if best_add_idx != -1:
                    h_curr[best_add_idx] = 0.05
                    h_refined = block_coordinate_refine(h_curr, max_iter=8)
                    if _score_raw(h_refined) > best_score:
                        best_score = _score_raw(h_refined)
                        best_h = h_refined.copy()
                        h_curr = h_refined.copy()
                        improved = True
            
            # Try removing a peak
            support_curr = np.where(h_curr > 1e-6)[0]
            if len(support_curr) > 3:
                best_remove_idx = -1
                best_remove_s = best_score
                for idx in support_curr[:15]:
                    h_try = h_curr.copy()
                    h_try[idx] = 0.0
                    s = _score_raw(h_try)
                    if s > best_remove_s:
                        best_remove_s = s
                        best_remove_idx = idx
                
                if best_remove_idx != -1:
                    h_curr[best_remove_idx] = 0.0
                    h_curr = h_curr / np.sum(h_curr)
                    h_refined = block_coordinate_refine(h_curr, max_iter=8)
                    if _score_raw(h_refined) > best_score:
                        best_score = _score_raw(h_refined)
                        best_h = h_refined.copy()
                        h_curr = h_refined.copy()
                        improved = True
            
            # Try moving a peak
            support_curr = np.where(h_curr > 1e-6)[0]
            for idx in support_curr[:10]:
                for offset in [-1, 1, -2, 2]:
                    new_idx = idx + offset
                    if 0 <= new_idx < N and h_curr[new_idx] < 1e-6:
                        h_try = h_curr.copy()
                        h_try[new_idx] = h_try[idx]
                        h_try[idx] = 0.0
                        s = _score_raw(h_try)
                        if s > best_score:
                            best_score = s
                            best_h = h_try.copy()
                            h_curr = h_try.copy()
                            improved = True
                            break
            
            if not improved:
                break
    
    return list(best_h)
