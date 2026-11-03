"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses smooth-max (log-sum-exp) with epigraph-style SLSQP polishing and
coordinate-descent refinement to equalize top active C_k constraints.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


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
    Uses iterative clipping and redistribution with exact final adjustment.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    for _ in range(20):
        diff = target - np.sum(h)
        if abs(diff) < 1e-14:
            break
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
    
    # Final exact adjustment: distribute residual to elements with slack
    diff = target - np.sum(h)
    if abs(diff) > 1e-13:
        if diff > 0:
            slack = 1.0 - h
            # Sort by slack descending, add to top elements
            order = np.argsort(-slack)
            for idx in order:
                if abs(diff) < 1e-14:
                    break
                add = min(diff, slack[idx])
                h[idx] += add
                diff -= add
        else:
            order = np.argsort(h)
            for idx in order:
                if abs(diff) < 1e-14:
                    break
                sub = min(-diff, h[idx])
                h[idx] -= sub
                diff += sub
    
    return h


def _smooth_max_value_grad(h, beta):
    """
    Returns value and gradient of the smooth max of C_k.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights
    
    val = m + np.log(sum_exp) / beta

    # Gradient calculation using FFT
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h_pad, n=n_fft)

    # term1_j = sum_{k=0}^{j} w[k] * h[j-k]
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]

    # term2_j = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    h_flip = h[::-1]
    h_flip_pad = np.zeros(n_fft)
    h_flip_pad[:n] = h_flip
    HF = np.fft.rfft(h_flip_pad, n=n_fft)
    conv2_full = np.fft.irfft(W * HF, n=n_fft)
    term2 = conv2_full[n - 1 - np.arange(n)]

    grad = -(2.0 / n) * (conv1 + term2)

    return val, grad


def _epigraph_obj(x, beta):
    """
    Epigraph-style objective: minimize t + penalty for C_k > t.
    x = [h_0, ..., h_{N-1}, t]
    """
    h = x[:N]
    t = x[N]
    cs = _linear_overlap_values(h)
    # Smooth penalty for violations: sum of softplus(beta*(cs - t))
    violations = cs - t
    softplus = np.log1p(np.exp(beta * violations)) / beta
    obj = t + 0.1 * np.sum(softplus)
    return obj


def _epigraph_grad(x, beta):
    """
    Gradient of epigraph objective.
    """
    h = x[:N]
    t = x[N]
    cs = _linear_overlap_values(h)
    violations = cs - t
    # d/dt = 1 - sum(sigma(beta*(cs-t)))
    sig = 1.0 / (1.0 + np.exp(-beta * violations))
    dt = 1.0 - np.sum(sig)
    
    # d/dh_j = -(2/N) * sum_k sigma_k * (h_{j-k} + h_{j+k})  [circular]
    n = N
    n_fft = 2 * n
    sig_pad = np.zeros(n_fft)
    sig_pad[:n] = sig
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    S = np.fft.rfft(sig_pad, n=n_fft)
    H = np.fft.rfft(h_pad, n=n_fft)
    conv1 = np.fft.irfft(S * H, n=n_fft)[:n]
    h_flip = h[::-1]
    h_flip_pad = np.zeros(n_fft)
    h_flip_pad[:n] = h_flip
    HF = np.fft.rfft(h_flip_pad, n=n_fft)
    conv2_full = np.fft.irfft(S * HF, n=n_fft)
    term2 = conv2_full[n - 1 - np.arange(n)]
    dh = -(2.0 / n) * (conv1 + term2)
    
    grad = np.concatenate([dh, [dt]])
    return grad


def _projected_gradient_descent(h0, beta, max_iter=2000, lr=None, tol=1e-9):
    """
    Projected Gradient Descent on the smooth max objective.
    Projection handles sum constraint and box constraints.
    """
    h = np.array(h0, dtype=np.float64)
    h = _fix_sum(h)
    
    if lr is None:
        val0, grad0 = _smooth_max_value_grad(h, beta)
        lr = 1.0 / (np.max(np.abs(grad0)) + 1e-6)
        lr = min(lr, 0.5)
    
    prev_val = float('inf')
    
    for i in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
            
        val, grad = _smooth_max_value_grad(h, beta)
        
        if abs(prev_val - val) < tol:
            break
        prev_val = val
        
        # Gradient step with backtracking line search
        step = lr
        h_new = h - step * grad
        h_new = _fix_sum(h_new)
        val_new = _score(h_new)
        
        # Backtrack if worse
        max_backtrack = 5
        for _ in range(max_backtrack):
            if val_new < val:
                break
            step *= 0.5
            h_new = h - step * grad
            h_new = _fix_sum(h_new)
            val_new = _score(h_new)
        
        h = h_new
        
    return _fix_sum(h)


def _coordinate_descent_refine(h, max_iter=200, top_k=5):
    """
    Coordinate descent that equalizes the top active C_k constraints.
    For each pair (i,j) of h indices, do a 2D line search to reduce max C_k.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    best_s = _score(h)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        cs = _linear_overlap_values(h)
        # Get top_k indices with highest C_k
        top_idx = np.argsort(-cs)[:top_k]
        current_max = cs[top_idx[0]]
        
        improved = False
        # Try adjusting pairs of h values
        n_try = 0
        for i in range(N):
            if n_try >= 50:
                break
            for j in range(i + 1, N):
                if n_try >= 50:
                    break
                n_try += 1
                
                # 2D line search: adjust h[i] and h[j] while preserving sum
                # h[i] += d, h[j] -= d
                best_d = 0.0
                best_val = current_max
                for d in np.linspace(-0.05, 0.05, 11):
                    if h[i] + d < 0 or h[i] + d > 1:
                        continue
                    if h[j] - d < 0 or h[j] - d > 1:
                        continue
                    h_test = h.copy()
                    h_test[i] += d
                    h_test[j] -= d
                    h_test = _fix_sum(h_test)
                    cs_test = _linear_overlap_values(h_test)
                    max_test = np.max(cs_test)
                    if max_test < best_val - 1e-12:
                        best_val = max_test
                        best_d = d
                
                if best_d != 0.0:
                    h[i] += best_d
                    h[j] -= best_d
                    h = _fix_sum(h)
                    s = _score(h)
                    if s < best_s - 1e-12:
                        best_s = s
                        current_max = s
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            break
    
    return h, best_s


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
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

    i = np.arange(N)
    
    # Phase 0: Parametric Search (Structural Change)
    # Model h_i = clip(c + A * sin(2*pi*f*i/N + phi), 0, 1)
    # We search over (A, f, phi, c)
    # A in [0, 1], f in [1, 10] (integer frequencies are good for periodicity), phi in [0, 2pi], c in [0, 1]
    
    param_starts = []
    
    # Grid search over frequencies and amplitudes
    # We'll do a coarse grid first
    freqs = np.arange(1, 11)
    amps = np.linspace(0.1, 0.9, 5)
    phases = np.linspace(0, 2*np.pi, 5, endpoint=False)
    
    # Add some random parametric starts
    for _ in range(20):
        if time.time() - t0 > _DEADLINE - 100:
            break
        f = np.random.uniform(1, 15)
        a = np.random.uniform(0.2, 0.9)
        p = np.random.uniform(0, 2*np.pi)
        c = np.random.uniform(0.2, 0.8)
        h_raw = c + a * np.sin(2*np.pi*f*i/N + p)
        h_raw = np.clip(h_raw, 0, 1)
        # Normalize to sum N/2
        h_norm = _fix_sum(h_raw)
        param_starts.append(h_norm)
        
    # Coarse Grid
    for f in freqs:
        if time.time() - t0 > _DEADLINE - 100:
            break
        for a in amps:
            for p in phases:
                # Optimize c to satisfy sum constraint? 
                # _fix_sum handles it, but we can try to center it.
                # Ideally sum(h) = N/2.
                # If h = c + A sin(...), sum(h) = N*c + A * sum(sin(...))
                # For integer f, sum(sin) is 0. So sum(h) = N*c.
                # To get sum N/2, c should be 0.5.
                c = 0.5
                h_raw = c + a * np.sin(2*np.pi*f*i/N + p)
                h_raw = np.clip(h_raw, 0, 1)
                h_norm = _fix_sum(h_raw)
                param_starts.append(h_norm)
                
                # Also try c slightly offset to see if clipping helps
                for c_offset in [-0.1, 0.1]:
                     c_test = 0.5 + c_offset
                     h_raw2 = c_test + a * np.sin(2*np.pi*f*i/N + p)
                     h_raw2 = np.clip(h_raw2, 0, 1)
                     h_norm2 = _fix_sum(h_raw2)
                     param_starts.append(h_norm2)

    # Update best with parametric starts
    for h_start in param_starts:
        _update_best(h_start)
        
    # If we found a very good parametric solution, we might not need as many random starts
    # But we still need to refine.
    
    # Phase 1: Gradient Descent from best parametric start and a few randoms
    starts = [best_h] if best_h is not None else []
    
    # Add a few random starts to ensure coverage
    for _ in range(5):
        if time.time() - t0 > _DEADLINE - 80:
            break
        h_rand = np.random.uniform(0, 1, N)
        h_norm = _fix_sum(h_rand)
        starts.append(h_norm)
        _update_best(h_norm)

    for h_start in starts:
        if time.time() - t0 > _DEADLINE - 40:
            break
        _update_best(h_start)
        
        # Run PGT with moderate beta
        h_pgd = _projected_gradient_descent(h_start, beta=30.0, max_iter=400)
        _update_best(h_pgd)
        
        # If time permits, try higher beta for more precise max approximation
        if time.time() - t0 < _DEADLINE - 20:
            h_pgd2 = _projected_gradient_descent(h_pgd, beta=80.0, max_iter=200)
            _update_best(h_pgd2)

    # Phase 2: SLSQP polishing with increasing beta
    if best_h is not None:
        for beta, maxiter, ftol in [(50.0, 150, 1e-8), (100.0, 150, 1e-9), (200.0, 100, 1e-10)]:
            if time.time() - t0 > _DEADLINE - 15:
                break
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N

            def obj_func(x, b=beta):
                val, _ = _smooth_max_value_grad(x, b)
                return val

            def jac_func(x, b=beta):
                _, grad = _smooth_max_value_grad(x, b)
                return grad

            try:
                res = minimize(obj_func, best_h.copy(), jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass

    # Phase 3: Coordinate descent refinement to equalize top C_k
    if best_h is not None and time.time() - t0 < _DEADLINE - 3:
        h_cd, s_cd = _coordinate_descent_refine(best_h, max_iter=300, top_k=5)
        if s_cd < best_score:
            best_h = h_cd
            best_score = s_cd
    
    # Phase 4: Final SLSQP polish after coordinate descent
    if best_h is not None and time.time() - t0 < _DEADLINE - 1:
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N

        def obj_func_final(x):
            val, _ = _smooth_max_value_grad(x, 300.0)
            return val

        def jac_func_final(x):
            _, grad = _smooth_max_value_grad(x, 300.0)
            return grad

        try:
            res = minimize(obj_func_final, best_h.copy(), jac=jac_func_final, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 50, 'ftol': 1e-12})
            if res.success or res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
