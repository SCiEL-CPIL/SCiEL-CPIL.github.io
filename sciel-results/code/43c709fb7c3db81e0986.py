"""
Second autocorrelation inequality: choose 50 nonnegative heights on equal intervals of [-1/4, 1/4]
to maximize (sum h_i^2) / ((sum h_i) * sup_k C[k]) where C = h autoconvolution.

Combined strategy:
1. Parametric Gaussian mixture model (2-3 components) to reduce search space
2. Epigraph formulation via SLSQP for structured optimization
3. Block-coordinate descent on symmetric pairs around dominant peak
4. Sparse initialization with diverse patterns
5. Local refinement with peak moving/scaling
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Generate diverse initializations ---
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Sparse structured
    np.random.seed(42)
    for trial in range(40):
        k = np.random.randint(2, 12)
        idx = np.random.choice(N, k, replace=False)
        h_s = np.zeros(N)
        h_s[idx] = 1.0
        h_s /= np.sum(h_s)
        inits.append(h_s)
        
    # 3. Concentrated Gaussians
    for c in np.linspace(3, N-3, 10):
        for w in np.linspace(0.5, N/3, 6):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            if np.sum(h_g) > 1e-10:
                h_g /= np.sum(h_g)
                inits.append(h_g)
                
    # 4. Triangular
    for c in np.linspace(3, N-3, 8):
        for w in np.linspace(2, N, 6):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 1e-10:
                h_t /= np.sum(h_t)
                inits.append(h_t)
                
    # 5. Random smooth
    for _ in range(25):
        h_r = np.random.rand(N)
        kernel = np.ones(5)/5
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 1e-10:
            h_r /= np.sum(h_r)
            inits.append(h_r)
            
    # 6. Structured sparse: pairs symmetric around center
    for k in range(2, 10):
        for offset in range(1, 8):
            h_pair = np.zeros(N)
            center = N // 2
            for i in range(k):
                pos1 = center - offset - i
                pos2 = center + offset + i
                if 0 <= pos1 < N:
                    h_pair[pos1] = 1.0
                if 0 <= pos2 < N:
                    h_pair[pos2] = 1.0
            if np.sum(h_pair) > 0:
                h_pair /= np.sum(h_pair)
                inits.append(h_pair)
    
    # 7. Alternating patterns
    for start in range(N):
        for gap in [2, 3, 4, 5, 7, 10]:
            h = np.zeros(N)
            for i in range(start, N, gap):
                h[i] = 1.0
            if np.sum(h) > 0:
                h /= np.sum(h)
                inits.append(h)
    
    # --- Phase 2: Parametric Gaussian Mixture Optimization ---
    # Represent h as sum of 2-3 shifted, scaled Gaussian basis functions
    # Parameters: [c1, w1, a1, c2, w2, a2, (c3, w3, a3)]
    # This reduces 50-dim search to 6-9 dim continuous search
    
    def gmm_to_h(params, n_components):
        """Convert GMM parameters to height vector"""
        h = np.zeros(N)
        for i in range(n_components):
            c = params[3*i]
            w = params[3*i+1]
            a = params[3*i+2]
            if w > 0.01:
                h += a * np.exp(-0.5 * ((x_idx - c) / w)**2)
        h = np.maximum(0, h)
        total = np.sum(h)
        if total > 1e-15:
            h /= total
        else:
            h = np.ones(N) / N
        return h
    
    def gmm_score(params, n_components):
        h = gmm_to_h(params, n_components)
        return _score_raw(h)
    
    # Optimize GMM parameters using Nelder-Mead
    gmm_inits = []
    
    # Try 2-component GMMs
    for c1 in np.linspace(5, N-5, 5):
        for w1 in [1, 2, 3, 5]:
            for c2 in np.linspace(5, N-5, 5):
                if abs(c2 - c1) < 3:
                    continue
                for w2 in [1, 2, 3, 5]:
                    params = np.array([c1, w1, 1.0, c2, w2, 1.0])
                    gmm_inits.append((params, 2))
    
    # Try 3-component GMMs
    for c1 in np.linspace(5, N-5, 3):
        for w1 in [1, 2, 4]:
            for c2 in np.linspace(5, N-5, 3):
                if abs(c2 - c1) < 3:
                    continue
                for w2 in [1, 2, 4]:
                    for c3 in np.linspace(5, N-5, 3):
                        if abs(c3 - c1) < 3 or abs(c3 - c2) < 3:
                            continue
                        for w3 in [1, 2, 4]:
                            params = np.array([c1, w1, 1.0, c2, w2, 1.0, c3, w3, 1.0])
                            gmm_inits.append((params, 3))
    
    # Optimize top GMM candidates
    gmm_candidates = []
    for params, n_comp in gmm_inits[:50]:
        s = gmm_score(params, n_comp)
        gmm_candidates.append((s, params, n_comp))
    
    gmm_candidates.sort(key=lambda x: -x[0])
    
    for s_init, params_init, n_comp in gmm_candidates[:15]:
        if time.time() > deadline - 50:
            break
        
        def neg_score(params):
            return -gmm_score(params, n_comp)
        
        bounds = []
        for i in range(n_comp):
            bounds.append((1, N-1))  # center
            bounds.append((0.5, N))  # width
            bounds.append((0.1, 10))  # amplitude
        
        try:
            res = opt.minimize(
                neg_score,
                params_init,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            h_candidate = gmm_to_h(res.x, n_comp)
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
        except Exception:
            pass
    
    # --- Phase 3: Epigraph Formulation Optimization (vectorized, fewer constraints) ---
    num_conv = 2 * N - 1

    # Precompute the convolution structure: for each output index k,
    # which pairs (i, j) contribute to conv[k] = sum_i h[i]*h[k-i]
    # We only need constraints at the argmax location for a tighter formulation,
    # but for safety we use a log-sum-exp soft-max surrogate.

    def _convolve(h):
        return np.convolve(h, h, mode='full')

    def run_epigraph(h_init, maxiter=300, ftol=1e-10, soft_beta=50.0):
        """Epigraph with soft-max surrogate for the sup norm.
        
        Minimize M s.t. sum(h)=1, h>=0, M >= logsumexp(beta * (conv - max_conv))
        where max_conv is tracked. Actually simpler: use the constraint
        M >= conv[k] for all k, but with a smoothed version.
        
        We use: minimize M + (1/beta)*log(sum(exp(beta*conv))) 
        s.t. sum(h)=1, h>=0
        This is a smooth objective in (h, M) where M is auxiliary.
        Actually, let's just do the standard epigraph but with vectorized jac
        and fewer iterations since we have better inits.
        """
        if np.sum(h_init) < 1e-15:
            return None
        h_init = np.asarray(h_init, dtype=np.float64) / np.sum(h_init)
        
        conv_init = _convolve(h_init)
        M_init = np.max(conv_init) + 1e-6
        x0 = np.concatenate((h_init, [M_init]))
        
        def obj(x):
            return x[N]
        
        def obj_grad(x):
            g = np.zeros(N + 1)
            g[N] = 1.0
            return g
        
        def eq_fun(x):
            return np.sum(x[:N]) - 1.0
        
        def eq_jac(x):
            j = np.zeros(N + 1)
            j[:N] = 1.0
            return j
        
        def ineq_fun(x):
            h = x[:N]
            M = x[N]
            conv = _convolve(h)
            # Only enforce the most active constraint (argmax) for speed,
            # plus a few neighbors
            k_max = np.argmax(conv)
            # Use top few peaks to be robust
            top_k = np.argsort(conv)[::-1][:5]
            return M - conv[top_k]
        
        def ineq_jac(x):
            h = x[:N]
            conv = _convolve(h)
            k_max = np.argmax(conv)
            top_k = np.argsort(conv)[::-1][:5]
            num_c = len(top_k)
            jac = np.zeros((num_c, N + 1))
            jac[:, N] = 1.0
            for ci, k in enumerate(top_k):
                # d(conv[k])/d(h[i]) = 2*h[k-i] if 0<=k-i<N else 0
                # but conv[k] = sum_i h[i]*h[k-i]
                # d/dh[j] conv[k] = h[k-j] + h[j-k] (if valid)
                for j in range(N):
                    a = k - j
                    if 0 <= a < N:
                        jac[ci, j] += -h[a]
                    b = j - k
                    if 0 <= b < N:
                        jac[ci, j] += -h[b]
            return jac
        
        constraints = [
            {'type': 'eq', 'fun': eq_fun, 'jac': eq_jac},
            {'type': 'ineq', 'fun': ineq_fun, 'jac': ineq_jac}
        ]
        
        bounds = [(0.0, 1.0)] * N + [(0.0, None)]
        
        try:
            res = opt.minimize(
                obj,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                jac=obj_grad,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            
            if res.success:
                h_candidate = np.clip(res.x[:N], 0, None)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate /= np.sum(h_candidate)
                    return h_candidate
        except Exception:
            pass
        
        # Fallback: try full epigraph with all constraints if soft version fails
        try:
            def ineq_fun_full(x):
                h = x[:N]
                M = x[N]
                conv = _convolve(h)
                return M - conv
            
            def ineq_jac_full(x):
                h = x[:N]
                jac = np.zeros((num_conv, N + 1))
                jac[:, N] = 1.0
                for j in range(N):
                    for k in range(j, min(j + N, num_conv)):
                        a = k - j
                        if 0 <= a < N:
                            jac[k, j] += -2.0 * h[a]
                return jac
            
            constraints_full = [
                {'type': 'eq', 'fun': eq_fun, 'jac': eq_jac},
                {'type': 'ineq', 'fun': ineq_fun_full, 'jac': ineq_jac_full}
            ]
            
            res = opt.minimize(
                obj,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_full,
                jac=obj_grad,
                options={'maxiter': maxiter, 'ftol': ftol}
            )
            
            if res.success:
                h_candidate = np.clip(res.x[:N], 0, None)
                if np.sum(h_candidate) > 1e-15:
                    h_candidate /= np.sum(h_candidate)
                    return h_candidate
        except Exception:
            pass
        
        return None

    # Run epigraph from diverse inits
    # Sort inits by initial score to prioritize good starting points
    init_scores = [(_score_raw(h), h) for h in inits]
    init_scores.sort(key=lambda x: -x[0])
    
    max_epigraph_runs = min(len(init_scores), 30)
    for i, (s_init, h_init) in enumerate(init_scores[:max_epigraph_runs]):
        if time.time() > deadline - 45:
            break
        h_candidate = run_epigraph(h_init, maxiter=250, ftol=1e-10)
        if h_candidate is not None:
            s_candidate = _score_raw(h_candidate)
            if s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    # --- Phase 4: Enhanced Block-Coordinate Descent with Single-Variable Moves ---
    def block_coordinate_refine(h_start, max_iter=40):
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
            
            conv = np.convolve(h, h, mode='full')
            k_star = np.argmax(conv)
            M_val = conv[k_star]
            l1 = np.sum(h)
            l2_sq = np.sum(h**2)
            
            # Single-variable moves: for each i, optimize h[i] with others fixed
            # After rescaling, the score depends on the ratio
            for i in range(N):
                if h[i] < 1e-15:
                    # Try increasing from zero
                    for delta in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
                        h_test = h.copy()
                        h_test[i] += delta
                        h_test = np.clip(h_test, 0, None)
                        h_test /= np.sum(h_test)
                        s_test = _score_raw(h_test)
                        if s_test > best_score and s_test > _score_raw(h):
                            h = h_test
                            improved = True
                            break
                else:
                    # Try adjusting h[i]
                    base_score = _score_raw(h)
                    for frac in [0.5, 0.7, 0.85, 1.15, 1.3, 1.5, 2.0, 0.1, 0.01]:
                        h_test = h.copy()
                        h_test[i] *= frac
                        h_test = np.clip(h_test, 0, None)
                        if np.sum(h_test) < 1e-15:
                            continue
                        h_test /= np.sum(h_test)
                        s_test = _score_raw(h_test)
                        if s_test > base_score:
                            h = h_test
                            base_score = s_test
                            improved = True
            
            # Pair moves: transfer mass between symmetric positions around k_star
            for i in range(N // 2):
                j = k_star - i
                if j < 0 or j >= N or j == i:
                    continue
                
                base_score = _score_raw(h)
                for frac in [0.3, 0.5, 0.7, 1.3, 1.5, 2.0]:
                    h_test = h.copy()
                    transfer = h_test[i] * (frac - 1.0)
                    h_test[i] -= transfer
                    h_test[j] += transfer
                    h_test = np.clip(h_test, 0, None)
                    if np.sum(h_test) < 1e-15:
                        continue
                    h_test /= np.sum(h_test)
                    s_test = _score_raw(h_test)
                    if s_test > base_score:
                        h = h_test
                        base_score = s_test
                        improved = True
        
        return h
    
    # Apply block coordinate refinement to best solution
    if time.time() < deadline - 10:
        h_refined = block_coordinate_refine(best_h, max_iter=40)
        s_refined = _score_raw(h_refined)
        if s_refined > best_score:
            best_score = s_refined
            best_h = h_refined
    
    # Additional refinement: try to zero out small components (sparsification)
    if time.time() < deadline - 8:
        h_sparse = best_h.copy()
        # Try zeroing the smallest components one by one
        for _ in range(10):
            if np.sum(h_sparse < 0.01 * np.max(h_sparse)) == 0:
                break
            # Find the smallest nonzero component
            mask = h_sparse > 1e-10
            if not np.any(mask):
                break
            smallest_idx = np.argmin(np.where(mask, h_sparse, np.inf))
            h_trial = h_sparse.copy()
            h_trial[smallest_idx] = 0
            h_trial /= np.sum(h_trial)
            s_trial = _score_raw(h_trial)
            if s_trial >= _score_raw(h_sparse) - 1e-6:
                h_sparse = h_trial
                if s_trial > best_score:
                    best_score = s_trial
                    best_h = h_sparse.copy()
            else:
                break
    
    # --- Phase 5: Final Polish with Tighter Tolerances and Nelder-Mead on top variables ---
    if time.time() < deadline - 3:
        h_polished = run_epigraph(best_h, maxiter=400, ftol=1e-12)
        if h_polished is not None:
            s_polished = _score_raw(h_polished)
            if s_polished > best_score:
                best_score = s_polished
                best_h = h_polished
        
        # Nelder-Mead polish on the support of the best solution
        if time.time() < deadline - 2:
            support = np.where(best_h > 0.005 * np.max(best_h))[0]
            if 2 <= len(support) <= 20:
                def nm_objective(params):
                    h_nm = np.zeros(N)
                    h_nm[support] = np.maximum(0, params)
                    if np.sum(h_nm) < 1e-15:
                        return 0.0
                    h_nm /= np.sum(h_nm)
                    return -_score_raw(h_nm)
                
                x0_nm = best_h[support].copy()
                try:
                    res_nm = opt.minimize(
                        nm_objective,
                        x0_nm,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    h_nm = np.zeros(N)
                    h_nm[support] = np.maximum(0, res_nm.x)
                    if np.sum(h_nm) > 1e-15:
                        h_nm /= np.sum(h_nm)
                        s_nm = _score_raw(h_nm)
                        if s_nm > best_score:
                            best_score = s_nm
                            best_h = h_nm
                except Exception:
                    pass
    
    # Ensure valid output
    best_h = np.maximum(0, best_h)
    s = np.sum(best_h)
    if s > 0:
        best_h = best_h / s
    else:
        best_h = np.zeros(N)
        best_h[0] = 1.0
        
    return list(best_h)
