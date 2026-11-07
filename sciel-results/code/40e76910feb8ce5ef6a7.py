"""Second autocorrelation inequality - piecewise-linear tent basis optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _h_from_tent_params(params):
    """
    Convert tent-basis parameters to normalized nonnegative heights.
    params: array of length 2*K where K is number of knots.
            First K values are knot positions in [0, N-1], last K are amplitudes.
    """
    K = len(params) // 2
    knots = np.sort(params[:K])
    amps = params[K:]
    
    # Ensure knots are within [0, N-1]
    knots = np.clip(knots, 0.0, N - 1)
    
    # Build piecewise linear function
    x = np.arange(N, dtype=float)
    h = np.zeros(N)
    
    for k in range(K):
        center = knots[k]
        amp = amps[k]
        if amp <= 0:
            continue
        # Find neighboring knots for this tent
        left = 0.0
        right = N - 1.0
        for j in range(K):
            if j != k and knots[j] < center:
                if knots[j] > left:
                    left = knots[j]
            if j != k and knots[j] > center:
                if knots[j] < right:
                    right = knots[j]
        
        # Build tent: linear rise from left to center, linear fall from center to right
        for i in range(N):
            xi = x[i]
            if xi < left or xi > right:
                continue
            if xi <= center:
                if center - left < 1e-10:
                    h[i] += amp
                else:
                    h[i] += amp * (xi - left) / (center - left)
            else:
                if right - center < 1e-10:
                    h[i] += amp
                else:
                    h[i] += amp * (right - xi) / (right - center)
    
    # Enforce non-negativity and normalize
    h = np.maximum(h, 1e-10)
    s = h.sum()
    if s < 1e-300:
        return np.ones(N) / N
    return h / s

def _neg_value_tent(params):
    h = _h_from_tent_params(params)
    return -_compute_value(h)

def _coordinate_descent_tent(params, max_sweeps=30, deadline=None):
    """Coordinate descent with golden section search in tent parameter space."""
    params = np.array(params, dtype=float)
    best_val = -_neg_value_tent(params)
    K = len(params) // 2
    
    for sweep in range(max_sweeps):
        if deadline is not None and time.time() > deadline - 3:
            break
        improved = False
        for i in range(len(params)):
            if deadline is not None and time.time() > deadline - 3:
                break
            old_val = -_neg_value_tent(params)
            
            # Determine search bounds
            if i < K:
                # Knot position: search in [0, N-1]
                lo = 0.0
                hi = N - 1.0
            else:
                # Amplitude: search in [0, 10]
                lo = 0.0
                hi = 10.0
            
            # Golden section search
            gr = (np.sqrt(5) - 1) / 2
            a = lo + (1 - gr) * (hi - lo)
            b = lo + gr * (hi - lo)
            
            params_a = params.copy()
            params_b = params.copy()
            params_a[i] = a
            params_b[i] = b
            fa = _neg_value_tent(params_a)
            fb = _neg_value_tent(params_b)
            
            for _ in range(40):
                if deadline is not None and time.time() > deadline - 3:
                    break
                if fa < fb:
                    hi = b
                    b = a
                    fb = fa
                    a = lo + (1 - gr) * (hi - lo)
                    params_a[i] = a
                    fa = _neg_value_tent(params_a)
                else:
                    lo = a
                    a = b
                    fa = fb
                    b = lo + gr * (hi - lo)
                    params_b[i] = b
                    fb = _neg_value_tent(params_b)
            
            best_i = (a + b) / 2
            params_new = params.copy()
            params_new[i] = best_i
            val_new = -_neg_value_tent(params_new)
            
            if val_new > old_val + 1e-12:
                params = params_new
                improved = True
                best_val = val_new
        
        if not improved:
            break
    
    return params, best_val

def _generate_tent_candidates(rng, K):
    """Generate diverse tent-basis initializations."""
    candidates = []
    
    # Uniform-like: single wide tent
    p = np.zeros(2 * K)
    p[0] = N / 2
    p[K] = 1.0
    candidates.append(p.copy())
    
    # Gaussian-like: multiple tents approximating Gaussian
    for n_knots in [2, 3, 4, 5]:
        if n_knots > K:
            break
        for width in [0.2, 0.3, 0.5, 0.7, 1.0]:
            p = np.zeros(2 * K)
            centers = np.linspace(N/2 - width*N/2, N/2 + width*N/2, n_knots)
            for k in range(n_knots):
                p[k] = centers[k]
                p[K + k] = np.exp(-((centers[k] - N/2) / (width * N / 2))**2)
            candidates.append(p.copy())
    
    # Bimodal
    for sep in [0.2, 0.3, 0.4, 0.5]:
        for width in [0.1, 0.2, 0.3]:
            p = np.zeros(2 * K)
            c1 = N/2 - sep * N / 2
            c2 = N/2 + sep * N / 2
            n1 = max(1, K // 2)
            n2 = max(1, K - n1)
            for k in range(n1):
                p[k] = c1 + (k - (n1-1)/2) * width * N / 4
                p[K + k] = np.exp(-((p[k] - c1) / (width * N / 4))**2)
            for k in range(n2):
                p[n1 + k] = c2 + (k - (n2-1)/2) * width * N / 4
                p[K + n1 + k] = np.exp(-((p[n1 + k] - c2) / (width * N / 4))**2)
            candidates.append(p.copy())
    
    # Random smooth
    for _ in range(50):
        p = np.zeros(2 * K)
        for k in range(K):
            p[k] = rng.uniform(0, N - 1)
            p[K + k] = rng.uniform(0.1, 2.0) * np.exp(-k * 0.2)
        candidates.append(p)
    
    # From known good shapes
    xh = np.linspace(-0.5, 0.5, N)
    idx = np.arange(N)
    center = N // 2
    
    shapes = [
        np.exp(-xh**2 * 4),
        np.exp(-xh**2 * 16),
        np.exp(-xh**2 * 64),
        np.maximum(0, 1 - 2*np.abs(xh)),
        np.maximum(0, np.cos(np.pi * xh)),
        np.exp(-((idx - center)**2) / 10),
        np.exp(-((idx - center)**2) / 40),
        1.0 / (1 + (4*xh)**2),
        1.0 / np.cosh(4*xh),
    ]
    
    for shape in shapes:
        shape = np.maximum(shape, 1e-8)
        shape = shape / shape.sum()
        # Fit tent basis: use peaks as knot locations
        p = np.zeros(2 * K)
        # Find top K peaks
        order = np.argsort(shape)[::-1]
        for k in range(min(K, len(order))):
            pos = order[k]
            p[k] = float(pos)
            p[K + k] = shape[pos] * N / 10  # Scale amplitude
        candidates.append(p)
    
    # Asymmetric
    for asym in [0.3, 0.7, 1.0, 1.5, 2.0]:
        shape = np.exp(-xh**2 * (4 + asym * xh * 4))
        shape = np.maximum(shape, 1e-8)
        shape = shape / shape.sum()
        p = np.zeros(2 * K)
        order = np.argsort(shape)[::-1]
        for k in range(min(K, len(order))):
            pos = order[k]
            p[k] = float(pos)
            p[K + k] = shape[pos] * N / 10
        candidates.append(p)
    
    # Multi-lobe
    for nlobes in [2, 3, 4]:
        for width in [0.1, 0.2, 0.3]:
            h_w = np.zeros(N)
            for k in range(nlobes):
                c = -1 + 2*k/(nlobes-1)
                h_w += np.exp(-(xh-c)**2/(2*width**2))
            h_w = np.maximum(h_w, 1e-8)
            h_w = h_w / h_w.sum()
            p = np.zeros(2 * K)
            order = np.argsort(h_w)[::-1]
            for k in range(min(K, len(order))):
                pos = order[k]
                p[k] = float(pos)
                p[K + k] = h_w[pos] * N / 10
            candidates.append(p)
    
    return candidates

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    K = 6  # Number of tent knots
    best_params = np.zeros(2 * K)
    best_val = _compute_value(_h_from_tent_params(best_params))
    
    def try_optimize(params0, method='L-BFGS-B', maxiter=500, **kwargs):
        nonlocal best_params, best_val
        if time.time() > deadline - 5:
            return
        try:
            if method == 'L-BFGS-B':
                opts = {'maxiter': maxiter, 'gtol': 1e-8, 'maxfun': 3000}
            elif method == 'Nelder-Mead':
                opts = {'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            elif method == 'Powell':
                opts = {'maxiter': maxiter, 'xtol': 1e-10, 'ftol': 1e-13}
            elif method == 'CG':
                opts = {'maxiter': maxiter, 'gtol': 1e-8}
            else:
                opts = {}
            res = minimize(_neg_value_tent, params0, method=method, options=opts)
            if res.fun < -best_val + 1e-15:
                best_val = -res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    def try_coordinate(params0, max_sweeps=30):
        nonlocal best_params, best_val
        if time.time() > deadline - 3:
            return
        try:
            p_new, val_new = _coordinate_descent_tent(params0, max_sweeps=max_sweeps, deadline=deadline)
            if val_new > best_val + 1e-15:
                best_val = val_new
                best_params = p_new
        except Exception:
            pass
    
    rng = np.random.default_rng(50)
    
    # Generate candidates
    candidates = _generate_tent_candidates(rng, K)
    
    # Phase 1: L-BFGS-B from diverse candidates
    phase1_deadline = t0 + 50
    for p0 in candidates:
        if time.time() > phase1_deadline:
            break
        try_optimize(p0, 'L-BFGS-B', maxiter=400)
    
    # Phase 2: Coordinate descent on best
    try_coordinate(best_params, max_sweeps=30)
    
    # Phase 3: Powell refinement
    if time.time() < deadline - 30:
        try_optimize(best_params, 'Powell', maxiter=400)
    
    # Phase 4: More random restarts
    phase4_deadline = t0 + 90
    for seed in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]:
        if time.time() > phase4_deadline:
            break
        rng2 = np.random.default_rng(seed)
        for _ in range(3):
            if time.time() > phase4_deadline:
                break
            p0 = np.zeros(2 * K)
            for k in range(K):
                p0[k] = rng2.uniform(0, N - 1)
                p0[K + k] = rng2.uniform(0.1, 3.0) * np.exp(-k * 0.3)
            try_optimize(p0, 'L-BFGS-B', maxiter=300)
    
    # Phase 5: Coordinate descent again
    try_coordinate(best_params, max_sweeps=30)
    
    # Phase 6: Nelder-Mead polish
    if time.time() < deadline - 20:
        try_optimize(best_params, 'Nelder-Mead', maxiter=500)
    
    # Phase 7: Structured restarts with different K
    phase7_deadline = t0 + 125
    for K2 in [4, 5, 7, 8]:
        if time.time() > phase7_deadline:
            break
        rng3 = np.random.default_rng(K2 * 100)
        p0 = np.zeros(2 * K2)
        for k in range(K2):
            p0[k] = rng3.uniform(0, N - 1)
            p0[K2 + k] = rng3.uniform(0.1, 2.5) * np.exp(-k * 0.25)
        # Optimize with different K
        local_best_params = p0.copy()
        local_best_val = _compute_value(_h_from_tent_params(p0))
        
        def _neg_value_tent_k2(params):
            h = _h_from_tent_params(params)
            return -_compute_value(h)
        
        try:
            res = minimize(_neg_value_tent_k2, p0, method='L-BFGS-B', 
                          options={'maxiter': 300, 'gtol': 1e-8, 'maxfun': 2000})
            if res.fun < -local_best_val + 1e-15:
                local_best_val = -res.fun
                local_best_params = res.x.copy()
        except Exception:
            pass
        
        # Convert to 2*K params if better
        if local_best_val > best_val + 1e-15:
            # Map K2 knots to K knots (take top K by amplitude)
            amps = local_best_params[K2:]
            order = np.argsort(amps)[::-1]
            p_mapped = np.zeros(2 * K)
            for k in range(min(K, K2)):
                p_mapped[k] = local_best_params[order[k]]
                p_mapped[K + k] = local_best_params[K2 + order[k]]
            val_mapped = _compute_value(_h_from_tent_params(p_mapped))
            if val_mapped > best_val + 1e-15:
                best_val = val_mapped
                best_params = p_mapped
    
    # Phase 8: Final coordinate descent + Powell
    try_coordinate(best_params, max_sweeps=30)
    if time.time() < deadline - 10:
        try_optimize(best_params, 'Powell', maxiter=300)
    
    # Phase 9: Final NM
    if time.time() < deadline - 5:
        try_optimize(best_params, 'Nelder-Mead', maxiter=400)
    
    # Final coordinate descent
    try_coordinate(best_params, max_sweeps=20)
    
    # Convert to heights
    best_h = _h_from_tent_params(best_params)
    best_h = np.maximum(best_h, 0)
    s = best_h.sum()
    if s < 1e-300:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / s
    return list(best_h)
