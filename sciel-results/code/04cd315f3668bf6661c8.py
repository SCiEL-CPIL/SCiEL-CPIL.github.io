"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines the epigraph optimization structure with a piecewise-constant + soft interface
parameterization to exploit the structure of the optimal solution.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
_DEADLINE = time.time() + 115  # Safety margin before 120s kill

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
    
    C_k = (2/n) * (n/2 - sum_{i=k}^{n-1} h[i] * h[i-k])
    dC_k/dh_j = -(2/n) * (h[j-k] if j>=k else 0 + h[j+k] if j+k<n else 0)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    
    # Compute gradient: grad_j = sum_k w_k * dC_k/dh_j
    # dC_k/dh_j = -(2/n) * (h[j-k] * [j>=k] + h[j+k] * [j+k<n])
    # So grad_j = -(2/n) * sum_k w_k * (h[j-k]*[j>=k] + h[j+k]*[j+k<n])
    #            = -(2/n) * (sum_{k<=j} w_k * h[j-k] + sum_{k<n-j} w_k * h[j+k])
    
    # First term: for each j, sum over k=0..j of w[k] * h[j-k] = (w * h)[j] (linear correlation)
    # Second term: for each j, sum over k=0..n-1-j of w[k] * h[j+k] = (w * flip(h))[n-1-j]
    
    n_fft = 2 * n
    W = np.fft.rfft(w, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    
    # Linear correlation: sum_{k=0}^{j} w[k] * h[j-k]
    corr1 = np.fft.irfft(W * np.conj(H), n=n_fft)[:n]
    
    # Second term: sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # This is like a cross-correlation with reversed indices
    # Let's compute it directly using FFT
    # sum_k w[k] * h[j+k] = sum_k w[k] * h[j+k]
    # Define g[k] = h[n-1-k], then h[j+k] = g[n-1-j-k]
    # sum_k w[k] * g[n-1-j-k] = (w * g)[n-1-j] (linear correlation)
    g = h[::-1]
    G = np.fft.rfft(g, n=n_fft)
    corr2_full = np.fft.irfft(W * np.conj(G), n=n_fft)[:n]
    # corr2_full[m] = sum_{k=0}^{m} w[k] * g[m-k] = sum_{k=0}^{m} w[k] * h[n-1-m+k]
    # We want for index j: sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # Let m = n-1-j, then we want sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # corr2_full[m] = sum_{k=0}^{m} w[k] * h[n-1-m+k] = sum_{k=0}^{n-1-j} w[k] * h[j+k]  ✓
    # So corr2[j] = corr2_full[n-1-j]
    corr2 = corr2_full[::-1]
    
    S = corr1 + corr2
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
            # Vectorized: J[k, j] = -(2/n) * (h[j-k] if j>=k else 0 + h[j+k] if j+k<n else 0)
            # For each k, this is a shifted version of h
            for k in range(n):
                # h[j-k] for j>=k: h[0:n-k] placed at positions k:n
                J[k, k:] = -(2.0 / n) * h[:n-k]
                # h[j+k] for j+k<n: h[k:n] placed at positions 0:n-k
                J[k, :n-k] += -(2.0 / n) * h[k:]
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

    # 3. Multi-scale structure search
    # Try various structured initializations that exploit the expected
    # piecewise-constant nature of the optimal solution
    
    def _build_structured_h(params):
        """
        Build h from structured parameters:
        params: [freq, phase, amplitude, bias, n_blocks, block_width]
        
        Creates a combination of:
        - A low-frequency cosine component
        - Block structure with smooth transitions
        """
        freq = params[0]      # 1 to 5
        phase = params[1]     # 0 to 2*pi
        amp = params[2]       # 0 to 1
        bias = params[3]      # 0 to 1
        n_blocks = int(round(params[4]))  # 1 to 10
        block_width = params[5]  # 1 to N/2
        
        i = np.arange(N)
        h = bias + amp * np.cos(2.0 * np.pi * freq * i / N + phase)
        
        # Add block structure: smooth transitions between 0 and 1 regions
        if n_blocks > 1 and block_width > 0:
            # Create a periodic block pattern
            period = N // n_blocks
            for b in range(n_blocks):
                start = b * period
                end = min(start + period, N)
                # Smooth transition at boundaries
                half_w = min(block_width, period / 4)
                if half_w > 0:
                    # Transition at start
                    trans_start = max(0, start - half_w)
                    trans_end = min(end, start + half_w)
                    if trans_end > trans_start:
                        t = (i[trans_start:trans_end] - start) / (2 * half_w) if half_w > 0 else 0
                        t = np.clip(t, -1, 1)
                        smooth = 0.5 * (1 + np.tanh(5 * t))
                        h[trans_start:trans_end] = h[trans_start:trans_end] * (1 - 0.3 * smooth)
                    
                    # Transition at end
                    trans_start2 = max(0, end - half_w)
                    trans_end2 = min(N, end + half_w)
                    if trans_end2 > trans_start2:
                        t2 = (i[trans_start2:trans_end2] - end) / (2 * half_w) if half_w > 0 else 0
                        t2 = np.clip(t2, -1, 1)
                        smooth2 = 0.5 * (1 + np.tanh(5 * t2))
                        h[trans_start2:trans_end2] = h[trans_start2:trans_end2] * (1 - 0.3 * smooth2)
        
        h = np.clip(h, 0.0, 1.0)
        return h
    
    def _run_structured_optimization():
        nonlocal best_h, best_score
        
        bounds_struct = [
            (1.0, 5.0),       # freq
            (0.0, 2 * np.pi), # phase
            (0.0, 1.0),       # amplitude
            (0.3, 0.7),       # bias
            (1.0, 10.0),      # n_blocks
            (1.0, N / 4.0)    # block_width
        ]
        
        def obj_struct(params):
            h = _build_structured_h(params)
            h = _fix_sum(h)
            return _score(h)
        
        try:
            res = differential_evolution(obj_struct, bounds_struct, seed=42, 
                                         maxiter=30, tol=1e-5, popsize=15)
            if res.fun < best_score:
                h_new = _build_structured_h(res.x)
                _update_best(h_new)
        except Exception:
            pass

    # Phase 1: Epigraph optimization from cosine start
    if time.time() - t0 < _DEADLINE - 45:
        t_init = _score(h_cos)
        _run_epigraph(h_cos, t_init, maxiter=300, ftol=1e-10)
        
    # Phase 2: Structured global search
    if time.time() - t0 < _DEADLINE - 25:
        _run_structured_optimization()
        
    # Phase 3: Smooth-max refinement from multiple starts
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        starts = [best_h.copy(), h_cos.copy()]
        
        # Add some perturbed versions
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 10:
                break
            noise = np.random.normal(0, 0.1, N)
            h_pert = np.clip(best_h + noise, 0, 1)
            h_pert = _fix_sum(h_pert)
            starts.append(h_pert)
        
        for h0 in starts:
            if time.time() - t0 > _DEADLINE - 10:
                break
            
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
            
            _run_smooth_max(h0, beta=50.0, maxiter=150, ftol=1e-9)
        
    # Phase 4: Final polish with epigraph
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        t_init = _score(best_h)
        _run_epigraph(best_h, t_init, maxiter=200, ftol=1e-12)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h

# EVOLVE-BLOCK-END
