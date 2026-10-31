"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines the epigraph optimization structure with a piecewise-constant + soft interface
parameterization to exploit the structure of the optimal solution.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill

# EVOLVE-BLOCK-START

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    
    # Gradient computation
    # dC_k/dh_j = -(2/n) * (h_{j-k} [j>=k] + h_{j+k} [j+k<n])
    # grad_j = sum_k w_k * dC_k/dh_j
    n_fft = 2 * n
    W = np.fft.rfft(w, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    conv_wh = np.fft.irfft(W * H, n=n_fft)[:n]
    
    T = np.zeros(n)
    for k in range(n):
        if w[k] != 0.0:
            T[:n-k] += w[k] * h[k:]
            
    S = conv_wh + T
    grad = -(2.0 / n) * S
    
    return val, grad

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Analytical Start: Cosine Wave
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # 2. Epigraph optimization
    def _run_epigraph(h0, t0_val, maxiter, ftol):
        nonlocal best_h, best_score
        
        x0 = np.concatenate([h0, [t0_val]])
        
        def obj(x):
            return x[N]
        
        def obj_grad(x):
            g = np.zeros(N + 1)
            g[N] = 1.0
            return g
        
        def cons_ineq(x):
            h = x[:N]
            t = x[N]
            cs = _linear_overlap_values(h)
            return t - cs
        
        def cons_ineq_jac(x):
            h = x[:N]
            t = x[N]
            n = N
            J = np.zeros((n, n + 1))
            for k in range(n):
                for j in range(n):
                    g = 0.0
                    if j >= k:
                        g += h[j-k]
                    if j + k < n:
                        g += h[j+k]
                    J[k, j] = -(2.0 / n) * g
                J[k, n] = -1.0
            return J
        
        def cons_eq(x):
            return np.sum(x[:N]) - N/2.0
        
        def cons_eq_jac(x):
            g = np.zeros(N + 1)
            g[:N] = 1.0
            return g
        
        constraints = [
            {'type': 'ineq', 'fun': cons_ineq, 'jac': cons_ineq_jac},
            {'type': 'eq', 'fun': cons_eq, 'jac': cons_eq_jac}
        ]
        bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)]
        
        try:
            res = minimize(obj, x0, jac=obj_grad, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.x is not None:
                h_new = res.x[:N]
                _update_best(h_new)
        except Exception:
            pass

    # 3. Piecewise-constant + soft interface parameterization
    # Hypothesis: The optimal h is mostly 0 and 1, with a small transition region.
    # We parameterize h by:
    #   - A binary base pattern (0 or 1)
    #   - A transition window position and width
    #   - A smooth profile within the window (e.g., sigmoid or cosine)
    
    def _build_piecewise_h(params):
        """
        params: [pos, width, slope, phase, ...]
        pos: center of transition (0 to N)
        width: width of transition
        slope: steepness of transition
        phase: offset for the binary pattern
        
        We construct h such that:
        - h is 0 or 1 in the bulk
        - h transitions smoothly in a window of size 'width' around 'pos'
        - The sum constraint is handled by _fix_sum later, but we try to
          make the base pattern have the right mean.
        """
        pos = params[0]
        width = params[1]
        slope = params[2]
        
        h = np.zeros(N)
        x = np.arange(N)
        
        # Create a smooth step function
        # sigmoid-like: 1 / (1 + exp(-slope * (x - pos)))
        # But we want it to go from 1 to 0 or 0 to 1
        # Let's say h is 1 for x < pos - width/2 and 0 for x > pos + width/2
        # Smooth transition in between
        
        # Actually, let's use a simpler approach:
        # h_i = 1 if i < pos, 0 if i >= pos, with a smooth transition
        # But we need to ensure the sum is approximately N/2.
        
        # Let's define a smooth step:
        # s(x) = 0.5 * (1 + tanh(slope * (x - pos)))
        # This goes from 1 to 0 as x increases past pos.
        
        # To get the right mean, we can shift pos.
        # If we want mean 0.5, we need the area under s(x) to be N/2.
        # For a symmetric tanh, this happens when pos = N/2.
        
        # Let's try a different structure:
        # h is a block of 1s followed by a block of 0s, with a smooth transition.
        # The position of the transition determines the sum.
        
        # We'll optimize pos and width.
        # h(x) = 1 / (1 + exp(slope * (x - pos)))
        # This is a smooth step from 1 to 0.
        
        h = 1.0 / (1.0 + np.exp(slope * (x - pos)))
        
        # Clip to [0, 1]
        h = np.clip(h, 0.0, 1.0)
        
        return h
    
    def _run_piecewise_optimization():
        nonlocal best_h, best_score
        
        # Optimize pos, width, slope
        # pos: 0 to N
        # width: 1 to N/2
        # slope: 1 to 20
        
        bounds_pw = [(0.0, float(N)), (1.0, N/2.0), (1.0, 20.0)]
        
        def obj_pw(params):
            h = _build_piecewise_h(params)
            h = _fix_sum(h)
            return _score(h)
        
        try:
            # Use differential evolution for global search
            res = differential_evolution(obj_pw, bounds_pw, seed=42, maxiter=50, tol=1e-6)
            if res.fun < best_score:
                h_new = _build_piecewise_h(res.x)
                _update_best(h_new)
        except Exception:
            pass
        
        # Then refine with SLSQP
        if best_h is not None:
            # Use the best piecewise result as start
            # But we need to convert back to params... which is tricky.
            # Instead, let's just run a few random piecewise starts and refine.
            pass

    # Phase 1: Epigraph optimization from cosine start
    if time.time() - t0 < _DEADLINE - 40:
        t_init = _score(h_cos)
        _run_epigraph(h_cos, t_init, maxiter=300, ftol=1e-10)
        
    # Phase 2: Piecewise-constant + soft interface search
    if time.time() - t0 < _DEADLINE - 20:
        _run_piecewise_optimization()
        
    # Phase 3: Try a few random piecewise starts and refine with SLSQP
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        for _ in range(5):
            if time.time() - t0 > _DEADLINE - 5:
                break
            
            # Random piecewise start
            pos = np.random.uniform(0, N)
            width = np.random.uniform(1, N/2)
            slope = np.random.uniform(1, 20)
            
            h_pw = _build_piecewise_h([pos, width, slope])
            h_pw = _fix_sum(h_pw)
            _update_best(h_pw)
            
            # Refine with smooth max SLSQP
            def _run_smooth_max(h0, beta, maxiter, ftol):
                nonlocal best_h, best_score
                cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                         'jac': lambda x: np.ones(N)}]
                bounds = [(0.0, 1.0)] * N
                
                def obj_func(x):
                    val, _ = _smooth_max_grad(x, beta)
                    return val
                    
                def jac_func(x):
                    _, grad = _smooth_max_grad(x, beta)
                    return grad
                    
                try:
                    res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                                   bounds=bounds, constraints=cons,
                                   options={'maxiter': maxiter, 'ftol': ftol})
                    if res.success or (res.fun < best_score and np.all(res.x >= -1e-6)):
                        _update_best(res.x)
                except Exception:
                    pass
            
            _run_smooth_max(h_pw, beta=50.0, maxiter=100, ftol=1e-8)
        
    # Phase 4: Final polish with epigraph
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        t_init = _score(best_h)
        _run_epigraph(best_h, t_init, maxiter=200, ftol=1e-12)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h

# EVOLVE-BLOCK-END
