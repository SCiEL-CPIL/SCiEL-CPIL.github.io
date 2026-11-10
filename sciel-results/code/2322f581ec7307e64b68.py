import numpy as np
import time
import scipy.optimize as so

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    delta = 1.0 / N
    
    # Deadline guard
    deadline = time.time() + 610.0
    
    def _union_len_1d(pos, c):
        """Compute 1D union length of intervals [pos[i] + c*i/N, pos[i] + c*i/N + delta] for i=0..N-1"""
        lefts = pos + c * np.arange(N) / N
        rights = lefts + delta
        events = np.zeros(2 * N, dtype=np.int8)
        vals = np.concatenate([lefts, rights])
        types = np.concatenate([np.zeros(N, dtype=np.int8), np.ones(N, dtype=np.int8)])
        order = np.lexsort((types, vals))
        vals_s = vals[order]
        types_s = types[order]
        
        total = 0.0
        count = 0
        start = 0.0
        for v, t in zip(vals_s, types_s):
            if t == 0:  # left
                if count == 0:
                    start = v
                count += 1
            else:  # right
                count -= 1
                if count == 0:
                    total += v - start
        return total
    
    def _find_critical(pos):
        """Find all c in (0,1) where ordering of endpoints changes"""
        crit = set()
        for i in range(N):
            for j in range(i + 1, N):
                denom = i - j  # negative
                c1 = N * (pos[j] - pos[i]) / (i - j)
                c2 = N * (pos[j] + delta - pos[i]) / (i - j)
                c3 = N * (pos[j] - pos[i] - delta) / (i - j)
                for c_val in [c1, c2, c3]:
                    if 0.0 < c_val < 1.0:
                        crit.add(round(c_val, 14))
        return sorted(crit)
    
    def _exact_volume(X, Y):
        """Compute exact union volume"""
        crit_x = _find_critical(X)
        crit_y = _find_critical(Y)
        all_crit = sorted(set([0.0] + crit_x + crit_y + [1.0]))
        
        vol = 0.0
        for k in range(len(all_crit) - 1):
            c0 = all_crit[k]
            c1 = all_crit[k + 1]
            h = c1 - c0
            if h < 1e-15:
                continue
            
            Lx0 = _union_len_1d(X, c0)
            Lx1 = _union_len_1d(X, c1)
            Ly0 = _union_len_1d(Y, c0)
            Ly1 = _union_len_1d(Y, c1)
            
            dx = Lx1 - Lx0
            dy = Ly1 - Ly0
            term = Lx0 * Ly0 + 0.5 * (Lx0 * dy + Ly0 * dx) + (1.0 / 3.0) * dx * dy
            vol += h * term
        
        return vol
    
    def _objective(params):
        X = params[:N]
        Y = params[N:]
        return _exact_volume(X, Y)
    
    def _soft_min(a, b, eps=1e-6):
        """Smooth minimum via log-sum-exp trick: min(a,b) ≈ -log(exp(a/eps)+exp(b/eps))*eps for small eps,
        but we use a stable formulation."""
        m = np.minimum(a, b)
        return m + eps * np.log(1.0 + np.exp(-(a - b) / eps)) if a > b else m + eps * np.log(1.0 + np.exp(-(b - a) / eps))
    
    def _soft_max(a, b, eps=1e-6):
        """Smooth maximum."""
        m = np.maximum(a, b)
        return m + eps * np.log(1.0 + np.exp(-(a - b) / eps)) if a < b else m + eps * np.log(1.0 + np.exp(-(b - a) / eps))
    
    def _smooth_volume(params, eps=1e-5):
        """Differentiable approximation of the union volume using inclusion-exclusion
        with soft-min/soft-max for pairwise intersection volumes.
        
        For tubes (i,j) and (i',j'), their 3D intersection at height c is the product of
        1D interval overlaps in x and y. The x-overlap of intervals [x_i+c*i/N, x_i+c*i/N+d]
        and [x_i'+c*i'/N, x_i'+c*i'/N+d] has length:
          max(0, min(right_i, right_i') - max(left_i, left_i'))
        We approximate max(0, ...) and min/max with soft versions for differentiability.
        
        Then V ≈ sum_k vol_k - sum_{k<k'} vol_{k∩k'} + sum_{k<k'<k''} vol_{k∩k'∩k''} - ...
        For 64 tubes this is intractable, so we use a surrogate:
        V_smooth ≈ ∫_0^1 [Lx_smooth(c) * Ly_smooth(c)] dc
        where Lx_smooth is a smooth approximation of the 1D union length.
        
        Actually, let's use a different approach: parameterize the 1D union length
        as a smooth function and integrate.
        """
        X = params[:N]
        Y = params[N:]
        
        # For the 1D union of intervals [pos[i] + c*i/N, pos[i] + c*i/N + d] for i=0..N-1,
        # the union length can be written as:
        # L(c) = max_i(right_i) - min_i(left_i) - sum of gaps
        # where gaps are the uncovered regions between consecutive intervals when sorted by left endpoint.
        #
        # A smoother approach: L(c) = ∫ 1_{∃i: x ∈ [left_i(c), right_i(c)]} dx
        # ≈ ∫ soft_max(0, sum_i 1_{x∈[left_i,right_i]}) dx  -- but this is still hard.
        #
        # Alternative: use the fact that for N=8, we can compute the exact 1D union length
        # piecewise-linearly in c, and the volume is a piecewise-quadratic function of the
        # parameters. The nonsmoothness comes from the ordering of endpoints.
        #
        # For a differentiable surrogate, use:
        # L(c) ≈ soft_max over all pairs of (right_j - left_i) for j >= i in some ordering
        # This is getting complex. Let's just use the exact volume but with a smoothed
        # version of the critical-point detection.
        #
        # Simplest differentiable approach: Monte Carlo with many samples, but that's slow.
        #
        # Let me try: the 1D union length is a convex function of the positions? No.
        #
        # Practical approach: use the exact volume but add a tiny smooth penalty for
        # near-degeneracies to help gradient methods.
        
        # For now, return exact volume (same as _objective) -- the gradient-based method
        # will use finite differences or we accept it's nonsmooth.
        # Better: implement a truly differentiable version.
        
        # Differentiable 1D union length via sorted endpoints with soft sorting:
        # The union length = sum over merged intervals of (right - left).
        # With N intervals, after merging, the union length is:
        # (max of all rights) - (min of all lefts) - sum of gaps between consecutive merged intervals.
        #
        # For N=8, there are at most 8 merged intervals. The union length is:
        # L = max(rights) - min(lefts) - sum_{gaps} gap_length
        #
        # Each gap is between the right endpoint of one merged group and the left endpoint
        # of the next. This is still nonsmooth.
        #
        # Let's use a different formulation. The union of intervals [a_i, b_i] has length:
        # L = ∫_R 1_{∪[a_i,b_i]}(x) dx
        #
        # A smooth upper bound (not equal to L) is:
        # L_smooth = ∫ soft_max(0, Σ_i sigmoid((x-a_i)/eps) * sigmoid((b_i-x)/eps)) dx
        # This is an overestimate that approaches L as eps→0.
        #
        # This is too complex to integrate analytically. Let me just use the exact volume
        # and rely on the fact that L-BFGS-B with finite differences can handle mild
        # nonsmoothness.
        
        return _exact_volume(X, Y)
    
    def _smooth_objective(params, eps=1e-5):
        """A differentiable surrogate: use the exact volume but add a small quadratic
        regularization to smooth out the kinks at critical points."""
        X = params[:N]
        Y = params[N:]
        vol = _exact_volume(X, Y)
        
        # Add a tiny smooth penalty for configurations where endpoints are very close
        # (near critical points), which helps gradient methods navigate the nonsmooth landscape.
        # The penalty is: eps * sum over pairs of |c_ij - c_ij'|^2 where c_ij are critical values
        # This is overkill; just return vol.
        return vol
    
    def _build_positions(X, Y):
        i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        i_flat = i_grid.flatten()
        j_flat = j_grid.flatten()
        base_x = X[i_flat]
        base_y = Y[j_flat]
        return np.stack([base_x, base_y], axis=1)
    
    # Keich-like initial guess
    pos_bits = int(np.log2(N))
    j_terms = np.arange(1, pos_bits + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps_bits = np.array(list(bin(idx)[2:]), dtype=int)
        eps_bits = np.pad(eps_bits, (pos_bits - len(eps_bits), 0), 'constant')
        keich[idx] = np.sum((1 - j_terms) * eps_bits * (2.0 ** -j_terms) / pos_bits)
    
    # Generate multiple initial candidates
    rng_init = np.random.default_rng(42)
    candidates = []
    
    # Keich
    candidates.append(np.concatenate([keich, keich]))
    
    # Uniform
    uniform = np.linspace(0, 0.5, N)
    candidates.append(np.concatenate([uniform, uniform]))
    
    # Random starts
    for _ in range(30):
        rx = rng_init.uniform(0, 0.5, N)
        ry = rng_init.uniform(0, 0.5, N)
        candidates.append(np.concatenate([rx, ry]))
    
    # Evaluate all candidates
    best_v = float('inf')
    best_p = None
    for p0 in candidates:
        try:
            v = _objective(p0)
            if v < best_v:
                best_v = v
                best_p = p0.copy()
        except Exception:
            pass
    
    # Nelder-Mead optimization with multiple restarts
    t_start = time.time()
    for restart in range(20):
        if time.time() > deadline - 90:
            break
        
        if restart == 0:
            p_current = best_p.copy()
        else:
            # Perturb with decreasing scale
            scale = 0.1 / (1 + restart * 0.3)
            perturbation = rng_init.normal(0, scale, 2 * N)
            p_current = best_p + perturbation
        
        try:
            res = so.minimize(_objective, p_current, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9,
                                       'adaptive': True})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
        
        # Powell refinement
        if time.time() > deadline - 90:
            break
        try:
            res2 = so.minimize(_objective, best_p, method='Powell',
                               options={'maxiter': 1000, 'xtol': 1e-10, 'ftol': 1e-10})
            if res2.fun < best_v:
                best_v = res2.fun
                best_p = res2.x.copy()
        except Exception:
            pass
    
    # L-BFGS-B polish using finite-difference gradients
    if time.time() < deadline - 60:
        try:
            res_lbfgs = so.minimize(_objective, best_p, method='L-BFGS-B',
                                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-9,
                                             'maxls': 50, 'maxfun': 50000})
            if res_lbfgs.fun < best_v:
                best_v = res_lbfgs.fun
                best_p = res_lbfgs.x.copy()
        except Exception:
            pass
    
    # Powell after L-BFGS-B for robustness
    if time.time() < deadline - 40:
        try:
            res_powell2 = so.minimize(_objective, best_p, method='Powell',
                                      options={'maxiter': 2000, 'xtol': 1e-11, 'ftol': 1e-11})
            if res_powell2.fun < best_v:
                best_v = res_powell2.fun
                best_p = res_powell2.x.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 20:
        try:
            res3 = so.minimize(_objective, best_p, method='Nelder-Mead',
                               options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12,
                                        'adaptive': True})
            if res3.fun < best_v:
                best_v = res3.fun
                best_p = res3.x.copy()
        except Exception:
            pass
    
    # Final L-BFGS-B with very tight tolerance
    if time.time() < deadline - 5:
        try:
            res_final = so.minimize(_objective, best_p, method='L-BFGS-B',
                                    options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10,
                                             'maxls': 100, 'maxfun': 100000})
            if res_final.fun < best_v:
                best_v = res_final.fun
                best_p = res_final.x.copy()
        except Exception:
            pass
    
    X_opt = best_p[:N]
    Y_opt = best_p[N:]
    
    positions = _build_positions(X_opt, Y_opt)
    
    # Validate
    if not np.all(np.isfinite(positions)):
        positions = _build_positions(keich, keich)
    
    # Ensure box area > 1e-9
    X0 = np.min(positions[:, 0])
    X1 = np.max(positions[:, 0] + (np.arange(64) // 8 + 1) / 8)
    Y0 = np.min(positions[:, 1])
    Y1 = np.max(positions[:, 1] + (np.arange(64) % 8 + 1) / 8)
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        positions = _build_positions(keich, keich)
    
    return positions
