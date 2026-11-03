import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Compute the ratio: L2^2 / (L1 * Sup(AutoConv))
    where AutoConv is the continuous autoconvolution of the step function.
    For a step function f(t) = sum h_i * 1_{[x_i, x_{i+1})}(t),
    the autoconvolution (f*f)(u) = integral f(t)f(u-t) dt.
    The peak value is DELTA * max(conv(h,h)).
    """
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

def _is_sidon_set(indices, n):
    """Check if a set of indices forms a Sidon set (all pairwise sums distinct)."""
    sums = set()
    m = len(indices)
    for i in range(m):
        for j in range(i, m):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _find_sidon_set(n, size, rng):
    """Find a Sidon set of given size in [0, n)."""
    for attempt in range(1000):
        candidates = rng.choice(n, size, replace=False)
        if _is_sidon_set(candidates, n):
            return candidates
    # Fallback: greedy construction
    result = []
    used_sums = set()
    for i in range(n):
        valid = True
        for j in result:
            s = i + j
            if s in used_sums:
                valid = False
                break
        if valid:
            for j in result:
                used_sums.add(i + j)
            used_sums.add(2 * i)
            result.append(i)
            if len(result) == size:
                break
    if len(result) == size:
        return np.array(result)
    return np.sort(rng.choice(n, size, replace=False))

def _autoconv_peak(h):
    """Compute the peak of the autoconvolution (without DELTA factor)."""
    h_conv = np.convolve(h, h, mode='full')
    return np.max(h_conv)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    def elapsed():
        return time.time() - start_time
    
    best_h = np.ones(N)
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    rng = np.random.RandomState(42)
    
    # --- Phase 1: Generate diverse initializations ---
    inits = []
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Gaussians
    for c in np.linspace(5, N-5, 8):
        for w in np.linspace(1.5, N/2, 8):
            h_g = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_g /= np.sum(h_g)
            inits.append(h_g.copy())
            
    # 3. Triangles
    for c in np.linspace(5, N-5, 8):
        for w in np.linspace(2, N, 8):
            h_t = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_t) > 0:
                h_t /= np.sum(h_t)
                inits.append(h_t.copy())
                
    # 4. Sidon-like structured sets
    for size in range(5, 18, 2):
        for _ in range(5):
            sidon_idx = _find_sidon_set(N, size, rng)
            h_s = np.zeros(N)
            h_s[sidon_idx] = 1.0
            h_s /= np.sum(h_s)
            inits.append(h_s.copy())
    
    # 5. Random sparse
    for _ in range(30):
        h_r = np.zeros(N)
        k = rng.randint(4, 16)
        idx = rng.choice(N, k, replace=False)
        h_r[idx] = rng.uniform(0.5, 1.5, k)
        h_r /= np.sum(h_r)
        inits.append(h_r.copy())
        
    # 6. Random smooth
    for _ in range(20):
        h_rs = rng.rand(N)
        kernel = np.ones(5)/5
        h_rs = np.convolve(h_rs, kernel, mode='same')
        h_rs /= np.sum(h_rs)
        inits.append(h_rs.copy())

    # --- Phase 2: Epigraph-based optimization ---
    # For fixed tau, minimize peak autoconvolution subject to:
    #   sum(h) = 1, h >= 0, sum(h^2) >= tau
    # This is convex for fixed tau. We sweep tau to find optimal shape.
    
    def _epigraph_solve(h_init, tau, maxiter=300):
        """Minimize peak autoconvolution with L1=1, L2^2 >= tau, h >= 0."""
        n = len(h_init)
        
        def obj(v):
            # v = [h (n vars), t (1 var)]
            h = v[:n]
            t = v[n]
            # We want to minimize t, but SLSQP minimizes obj
            # The peak autoconv is max of a set of quadratic forms
            # Use smooth approximation: t + penalty for violating t >= peak
            h_conv = np.convolve(h, h, mode='full')
            peak = np.max(h_conv)
            # Smooth max via softplus-like
            # Actually, let's just use the direct approach with constraint
            return t
        
        def gradient(v):
            h = v[:n]
            t = v[n]
            h_conv = np.convolve(h, h, mode='full')
            peak = np.max(h_conv)
            # Gradient of peak w.r.t. h is complex; use numerical
            return None
        
        constraints = [
            {'type': 'eq', 'fun': lambda v: np.sum(v[:n]) - 1.0},
            {'type': 'ineq', 'fun': lambda v: np.sum(v[:n]**2) - tau},
        ]
        # Add constraint: t >= peak_autoconv(h)
        # This is non-smooth. Use a different approach.
        
        # Alternative: directly minimize peak autoconv with SLSQP
        # using the fact that peak = max over shifts of sum_i h_i * h_{i+k}
        
        # Let's use a different formulation:
        # Minimize t subject to:
        #   sum(h) = 1
        #   h >= 0
        #   sum(h^2) >= tau
        #   t >= h*h[k] for all k (the autoconv at lag k)
        # The autoconv at lag k is sum_i h_i * h_{i+k}
        # This gives 2N-1 linear-ish constraints (quadratic in h)
        
        # For SLSQP, we can use:
        def neg_obj(v):
            h = v[:n]
            t = v[n]
            return -t  # minimize t -> maximize -t
        
        # Actually, let's just use the direct ratio optimization with better init
        # and rely on the epigraph idea for the LP-based approach
        
        # Use SLSQP to minimize peak autoconv directly
        def peak_obj(h):
            h_conv = np.convolve(h, h, mode='full')
            return np.max(h_conv)
        
        def neg_peak_obj(h):
            return -peak_obj(h)
        
        cons = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
            {'type': 'ineq', 'fun': lambda h: np.sum(h**2) - tau},
            {'type': 'ineq', 'fun': lambda h: h}
        ]
        bounds = [(0.0, 1.0)] * n
        
        try:
            res = minimize(
                neg_peak_obj, h_init, method='SLSQP',
                bounds=bounds, constraints=cons,
                options={'maxiter': maxiter, 'ftol': 1e-10}
            )
            if res.success:
                h_res = np.maximum(0, res.x)
                if np.sum(h_res) > 0:
                    h_res /= np.sum(h_res)
                    return h_res
        except Exception:
            pass
        return None
    
    # --- Phase 3: Optimize from each initialization ---
    for h_init in inits:
        if elapsed() > deadline - 30:
            break
        
        h_norm = h_init / np.sum(h_init)
        
        # Direct ratio optimization
        def neg_obj_h(h):
            l2_sq = np.sum(h**2)
            peak = _autoconv_peak(h)
            if peak == 0:
                return 0
            return -(l2_sq / peak)
        
        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
            {'type': 'ineq', 'fun': lambda h: h}
        ]
        bounds = [(0.0, 1.0)] * N
        
        try:
            res = minimize(
                neg_obj_h, h_norm, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-11}
            )
            if res.success:
                h_candidate = np.maximum(0, res.x)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate.copy()
        except Exception:
            pass
        
        # Epigraph: sweep tau values
        if elapsed() > deadline - 20:
            break
        
        l2_sq_init = np.sum(h_norm**2)
        tau_range = np.linspace(max(0.01, l2_sq_init * 0.5), min(1.0, l2_sq_init * 1.5), 5)
        for tau in tau_range:
            if elapsed() > deadline - 15:
                break
            h_ep = _epigraph_solve(h_norm, tau, maxiter=200)
            if h_ep is not None:
                s_ep = _score_raw(h_ep)
                if s_ep > best_score:
                    best_score = s_ep
                    best_h = h_ep.copy()

    # --- Phase 4: Final polish from best ---
    if elapsed() < deadline - 10:
        h_start = best_h / np.sum(best_h)
        
        def neg_obj_h(h):
            l2_sq = np.sum(h**2)
            peak = _autoconv_peak(h)
            if peak == 0:
                return 0
            return -(l2_sq / peak)
        
        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
            {'type': 'ineq', 'fun': lambda h: h}
        ]
        bounds = [(0.0, 1.0)] * N
        
        try:
            res = minimize(
                neg_obj_h, h_start, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 2000, 'ftol': 1e-14}
            )
            if res.success:
                h_candidate = np.maximum(0, res.x)
                if np.sum(h_candidate) > 0:
                    h_candidate /= np.sum(h_candidate)
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate.copy()
        except Exception:
            pass

    # Ensure nonnegative
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) == 0:
        best_h = np.ones(N)
    
    return list(best_h)
