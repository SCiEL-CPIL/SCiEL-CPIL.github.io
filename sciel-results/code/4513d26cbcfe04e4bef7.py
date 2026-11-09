# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy import optimize


def _compute_P_fast(s, w):
    """Fast exact P[X1+X2+X3 < 2*X4] for support s, normalized weights w (strict <)."""
    n = s.size
    if n <= 1:
        return 0.0
    pair_sums = s[:, None] + s[None, :]
    pair_w = w[:, None] * w[None, :]
    
    flat_sums = pair_sums.ravel()
    flat_w = pair_w.ravel()
    order = np.argsort(flat_sums)
    sorted_sums = flat_sums[order]
    sorted_w = flat_w[order]
    cumsum = np.cumsum(sorted_w)
    
    thresholds = 2.0 * s[None, :] - s[:, None]
    
    idx = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    H = np.zeros(n * n)
    mask = idx > 0
    H[mask] = cumsum[idx[mask] - 1]
    H = H.reshape(n, n)
    
    total = np.sum(w[:, None] * w[None, :] * H)
    return float(total)


def _P_geometric(log_r, log_alpha, n):
    """Compute P for a geometric measure: s_k = r^k, w_k ∝ α^k, k=0..n-1."""
    r = np.exp(log_r)
    alpha = np.exp(log_alpha)
    if r <= 0 or alpha <= 0 or r >= 1 or alpha >= 1:
        return -1.0
    k = np.arange(n)
    s = r ** k
    w = alpha ** k
    w = w / w.sum()
    if s[-1] < 1e-12:
        return -1.0
    return _compute_P_fast(s, w)


def _P_2point(a, b, p):
    """Compute P for a 2-point measure: {a, b} with weights {p, 1-p}."""
    if a < 0 or b < 0 or a >= b:
        return 0.0
    s = np.array([a, b], dtype=np.float64)
    w = np.array([p, 1.0 - p], dtype=np.float64)
    return _compute_P_fast(s, w)


def _P_3point(a, b, c, p, q):
    """Compute P for a 3-point measure: {a, b, c} with weights {p, q, 1-p-q}."""
    if a < 0 or b < 0 or c < 0 or a >= b or b >= c:
        return 0.0
    if p < 0 or q < 0 or p + q > 1:
        return 0.0
    s = np.array([a, b, c], dtype=np.float64)
    w = np.array([p, q, 1.0 - p - q], dtype=np.float64)
    return _compute_P_fast(s, w)


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    t0 = time.time()
    deadline = 620.0
    rng = np.random.default_rng(12345)
    
    best_P = -1.0
    best_s = None
    best_w = None
    
    def consider(s, w):
        nonlocal best_P, best_s, best_w
        if s.size < 1:
            return
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
            return
        if np.any(s < -1e-12) or np.any(w < -1e-12):
            return
        if w.sum() <= 1e-12:
            return
        w = w / w.sum()
        P = _compute_P_fast(s, w)
        if P > best_P:
            best_P = P
            best_s = s.copy()
            best_w = w.copy()
    
    # === Phase 1: Analytical / near-analytical 2-point solutions ===
    # For 2-point {a, b} with a < b/2, P depends only on p (weight on a)
    # Optimal p* = (15 - sqrt(33))/16 ≈ 0.3752
    p_star = (15.0 - np.sqrt(33.0)) / 16.0
    
    # For 2-point {a, b} with a >= b/2, optimal p = 3/4
    # Try a grid of (a, b) pairs
    for a in [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]:
        for b in [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
            if a < b / 2.0:
                consider(np.array([a, b]), np.array([p_star, 1 - p_star]))
            else:
                consider(np.array([a, b]), np.array([0.75, 0.25]))
    
    # === Phase 2: 2-parameter geometric family search ===
    # s_k = r^k for k=0..n-1, w_k ∝ α^k
    # Search over (log_r, log_alpha) with r in (0,1), α in (0,1)
    
    def neg_P_geo(params, n):
        val = _P_geometric(params[0], params[1], n)
        return -val if val > 0 else 1.0
    
    # Grid scan over geometric family
    for n in [3, 4, 5, 6, 8, 10]:
        if time.time() - t0 > deadline - 120:
            break
        log_r_vals = np.linspace(-10.0, -0.05, 30)  # r from ~4.5e-5 to ~0.95
        log_alpha_vals = np.linspace(-10.0, -0.05, 30)  # α from ~4.5e-5 to ~0.95
        
        # Coarse grid
        for lr in log_r_vals[::3]:
            for la in log_alpha_vals[::3]:
                P = _P_geometric(lr, la, n)
                if P > best_P:
                    s = np.exp(lr) ** np.arange(n)
                    w = np.exp(la) ** np.arange(n)
                    consider(s, w)
    
    # Fine grid around best geometric parameters found
    if best_s is not None and best_s.size >= 3:
        # Try to infer r, alpha from best solution
        pass
    
    # Local optimization in (log_r, log_alpha) space for each n
    for n in [4, 5, 6, 8, 10, 12]:
        if time.time() - t0 > deadline - 60:
            break
        
        # Multiple starting points
        starts = [
            (-2.0, -2.0), (-3.0, -1.0), (-1.0, -3.0),
            (-4.0, -2.0), (-2.0, -4.0), (-5.0, -1.0),
            (-1.0, -5.0), (-3.0, -3.0), (-1.5, -2.5),
            (-2.5, -1.5), (-4.0, -4.0), (-0.5, -2.0),
        ]
        
        for (lr0, la0) in starts:
            if time.time() - t0 > deadline - 30:
                break
            try:
                res = optimize.minimize(
                    lambda p: neg_P_geo(p, n),
                    x0=np.array([lr0, la0]),
                    method='Nelder-Mead',
                    options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                if res.x is not None and np.all(np.isfinite(res.x)):
                    lr, la = res.x
                    P = _P_geometric(lr, la, n)
                    if P > 0:
                        s = np.exp(lr) ** np.arange(n)
                        w = np.exp(la) ** np.arange(n)
                        consider(s, w)
            except Exception:
                pass
    
    # === Phase 3: 3-point analytical search ===
    # For 3-point {a, b, c} with a < b/2 < c, optimize (a, b, c, p, q)
    # Use coordinate descent on the 5 parameters
    
    def coordinate_descent_3pt(a0, b0, c0, p0, q0, max_rounds=200):
        """Coordinate descent for 3-point measure."""
        a, b, c, p, q = a0, b0, c0, p0, q0
        best_local_P = _P_3point(a, b, c, p, q)
        
        for rnd in range(max_rounds):
            if time.time() - t0 > deadline - 20:
                break
            improved = False
            
            # Optimize a (keep b, c, p, q fixed)
            if b > 0:
                a_lo, a_hi = max(0.0, b * 0.001), b * 0.49
                if a_lo < a_hi:
                    res_a = optimize.minimize_scalar(
                        lambda x: -_P_3point(x, b, c, p, q),
                        bounds=(a_lo, a_hi), method='bounded',
                        options={'xatol': 1e-10}
                    )
                    if res_a.success and -res_a.fun > best_local_P:
                        a = res_a.x
                        best_local_P = -res_a.fun
                        improved = True
            
            # Optimize b (keep a, c, p, q fixed)
            if c > 0:
                b_lo, b_hi = a * 1.01, c * 0.99
                if b_lo < b_hi:
                    res_b = optimize.minimize_scalar(
                        lambda x: -_P_3point(a, x, c, p, q),
                        bounds=(b_lo, b_hi), method='bounded',
                        options={'xatol': 1e-10}
                    )
                    if res_b.success and -res_b.fun > best_local_P:
                        b = res_b.x
                        best_local_P = -res_b.fun
                        improved = True
            
            # Optimize c (keep a, b, p, q fixed)
            c_lo = b * 1.01
            c_hi = max(c_lo * 100, b * 10)
            res_c = optimize.minimize_scalar(
                lambda x: -_P_3point(a, b, x, p, q),
                bounds=(c_lo, c_hi), method='bounded',
                options={'xatol': 1e-10}
            )
            if res_c.success and -res_c.fun > best_local_P:
                c = res_c.x
                best_local_P = -res_c.fun
                improved = True
            
            # Optimize p (keep a, b, c, q fixed)
            p_lo, p_hi = 1e-6, 1.0 - q - 1e-6
            if p_lo < p_hi:
                res_p = optimize.minimize_scalar(
                    lambda x: -_P_3point(a, b, c, x, q),
                    bounds=(p_lo, p_hi), method='bounded',
                    options={'xatol': 1e-10}
                )
                if res_p.success and -res_p.fun > best_local_P:
                    p = res_p.x
                    best_local_P = -res_p.fun
                    improved = True
            
            # Optimize q (keep a, b, c, p fixed)
            q_lo, q_hi = 1e-6, 1.0 - p - 1e-6
            if q_lo < q_hi:
                res_q = optimize.minimize_scalar(
                    lambda x: -_P_3point(a, b, c, p, x),
                    bounds=(q_lo, q_hi), method='bounded',
                    options={'xatol': 1e-10}
                )
                if res_q.success and -res_q.fun > best_local_P:
                    q = res_q.x
                    best_local_P = -res_q.fun
                    improved = True
            
            if not improved:
                break
        
        return a, b, c, p, q, best_local_P
    
    # Try multiple starting points for 3-point CD
    for trial in range(15):
        if time.time() - t0 > deadline - 30:
            break
        a0 = rng.uniform(1e-6, 0.3)
        b0 = rng.uniform(a0 * 2, 2.0)
        c0 = rng.uniform(b0 * 2, 50.0)
        p0 = rng.uniform(0.1, 0.7)
        q0 = rng.uniform(0.05, 0.3)
        
        a, b, c, p, q, P_cd = coordinate_descent_3pt(a0, b0, c0, p0, q0)
        if P_cd > 0:
            consider(np.array([a, b, c]), np.array([p, q, 1 - p - q]))
    
    # === Phase 4: General N-point coordinate descent (N=4,5) ===
    def coordinate_descent_Npt(s0, w0, max_rounds=100):
        """Coordinate descent for general N-point measure."""
        n = s0.size
        s = s0.copy().astype(np.float64)
        w = w0.copy().astype(np.float64)
        w = w / w.sum()
        best_local_P = _compute_P_fast(s, w)
        
        for rnd in range(max_rounds):
            if time.time() - t0 > deadline - 15:
                break
            improved = False
            
            # Optimize each support point
            for i in range(n):
                if i == 0:
                    lo, hi = 0.0, s[1] * 0.99 if n > 1 else 1.0
                elif i == n - 1:
                    lo, hi = s[i-1] * 1.01, s[i] * 10 + 1.0
                else:
                    lo, hi = s[i-1] * 1.001, s[i+1] * 0.999
                
                if lo < hi:
                    def obj_si(xi):
                        s_new = s.copy()
                        s_new[i] = xi
                        return -_compute_P_fast(s_new, w)
                    
                    res = optimize.minimize_scalar(obj_si, bounds=(lo, hi),
                                                   method='bounded', options={'xatol': 1e-9})
                    if res.success and -res.fun > best_local_P:
                        s[i] = res.x
                        best_local_P = -res.fun
                        improved = True
            
            # Optimize each weight (with sum constraint)
            for i in range(n):
                if n == 1:
                    continue
                # Optimize w[i] with others fixed: w[i] in [0, 1-sum(others)]
                others_sum = 1.0 - w[i]
                lo, hi = 1e-8, 1.0 - others_sum - 1e-8
                if lo < hi:
                    def obj_wi(wi):
                        w_new = w.copy()
                        w_new[i] = wi
                        # Renormalize: actually we need w[i] to be the actual weight
                        # Let's use a different parameterization
                        # w[i] = wi, others scale to (1-wi)
                        w_new[i] = wi
                        if wi < 1.0:
                            w_new[~(np.arange(n) == i)] = (1.0 - wi) * w[~(np.arange(n) == i)] / others_sum if others_sum > 0 else 0
                        else:
                            w_new[~(np.arange(n) == i)] = 0
                        return -_compute_P_fast(s, w_new)
                    
                    res = optimize.minimize_scalar(obj_wi, bounds=(lo, hi),
                                                   method='bounded', options={'xatol': 1e-9})
                    if res.success and -res.fun > best_local_P:
                        w_new = w.copy()
                        w_new[i] = res.x
                        if res.x < 1.0:
                            w_new[~(np.arange(n) == i)] = (1.0 - res.x) * w[~(np.arange(n) == i)] / others_sum if others_sum > 0 else 0
                        else:
                            w_new[~(np.arange(n) == i)] = 0
                        w = w_new / w_new.sum()
                        best_local_P = -res.fun
                        improved = True
            
            if not improved:
                break
        
        return s, w, best_local_P
    
    # Multi-start for N=4
    for trial in range(8):
        if time.time() - t0 > deadline - 20:
            break
        s0 = np.sort(rng.uniform(1e-6, 20.0, 4))
        w0 = rng.dirichlet(np.ones(4) * 0.5)
        s_cd, w_cd, P_cd = coordinate_descent_Npt(s0, w0, max_rounds=80)
        if P_cd > 0:
            consider(s_cd, w_cd)
    
    # Multi-start for N=5
    for trial in range(5):
        if time.time() - t0 > deadline - 15:
            break
        s0 = np.sort(rng.uniform(1e-6, 50.0, 5))
        w0 = rng.dirichlet(np.ones(5) * 0.5)
        s_cd, w_cd, P_cd = coordinate_descent_Npt(s0, w0, max_rounds=60)
        if P_cd > 0:
            consider(s_cd, w_cd)
    
    # === Phase 5: Final polish on best solution ===
    if best_s is not None:
        n_best = best_s.size
        if time.time() - t0 < deadline - 5:
            # Nelder-Mead polish in raw space
            def neg_obj_nm(params):
                s = np.maximum(params[:n_best], 0.0)
                w = np.maximum(params[n_best:], 0.0)
                if w.sum() < 1e-15:
                    return 1.0
                w = w / w.sum()
                return -_compute_P_fast(s, w)
            
            x0 = np.concatenate([best_s, best_w])
            try:
                res_nm = optimize.minimize(neg_obj_nm, x0, method='Nelder-Mead',
                                           options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-16})
                if res_nm.x is not None and np.all(np.isfinite(res_nm.x)):
                    s = np.maximum(res_nm.x[:n_best], 0.0)
                    w = np.maximum(res_nm.x[n_best:], 0.0)
                    if w.sum() > 1e-15:
                        w = w / w.sum()
                        consider(s, w)
            except Exception:
                pass
    
    # Prepare final answer
    if best_s is None:
        best_s = np.array([1e-8, 1.0], dtype=np.float64)
        best_w = np.array([p_star, 1.0 - p_star], dtype=np.float64)
    
    s = np.ascontiguousarray(best_s, dtype=np.float64).ravel()
    w = np.ascontiguousarray(best_w, dtype=np.float64).ravel()
    s = np.maximum(s, 0.0)
    w = np.maximum(w, 0.0)
    w = w / w.sum()
    
    # Final sanity check
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(w))):
        s = np.array([1e-8, 1.0], dtype=np.float64)
        w = np.array([p_star, 1.0 - p_star], dtype=np.float64)
    
    assert s.size == w.size >= 1
    assert np.all(np.isfinite(s)) and np.all(np.isfinite(w))
    assert np.all(s >= -1e-9) and np.all(w >= -1e-9) and w.sum() > 1e-9
    return s, w

# EVOLVE-BLOCK-END
