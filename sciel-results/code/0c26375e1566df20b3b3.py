import numpy as np
from scipy.optimize import minimize
import time

N = 100
TARGET_SUM = N / 2.0
DEADLINE = 135.0  # safety margin before 150s kill

def _correlation_max(h):
    """Compute the raw value: max correlation / N * 2."""
    corr = np.correlate(h, 1.0 - h, mode='full')
    return np.max(corr) / N * 2.0

def _corr_vector(h):
    """Return the full correlation vector (length 2N-1)."""
    return np.correlate(h, 1.0 - h, mode='full')

def _project_feasible(h):
    """Project h to satisfy 0 <= h <= 1 and sum(h) = N/2."""
    h = np.asarray(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    for _ in range(1000):
        diff = TARGET_SUM - np.sum(h)
        if abs(diff) < 1e-15:
            break
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        n_free = int(np.sum(mask))
        if n_free == 0:
            break
        h[mask] += diff / n_free
        h = np.clip(h, 0.0, 1.0)
    return h

def _smooth_objective(x, beta):
    """
    x = [h (N), t (1)]
    Objective: t + (1/beta) * log(sum_lag exp(beta * (corr_lag - t)))
    This is a smooth surrogate for max(t, max_lag corr_lag).
    """
    h = x[:N]
    t = x[N]
    corr = _corr_vector(h)
    # corr_lag - t
    shifted = beta * (corr - t)
    m = np.max(shifted)
    if m > 700:
        shifted = shifted - m
        logsum = m + np.log(np.sum(np.exp(shifted)))
    else:
        logsum = np.log(np.sum(np.exp(shifted)))
    return t + logsum / beta

def _smooth_objective_grad(x, beta):
    h = x[:N]
    t = x[N]
    corr = _corr_vector(h)
    shifted = beta * (corr - t)
    m = np.max(shifted)
    if m > 700:
        shifted = shifted - m
        logsum = m + np.log(np.sum(np.exp(shifted)))
    else:
        logsum = np.log(np.sum(np.exp(shifted)))
    # weights: p_lag = exp(beta*(corr_lag - t)) / sum_lag exp(...)
    p = np.exp(shifted)
    p_sum = np.sum(p)
    p = p / p_sum  # softmax weights over lags

    # Gradient w.r.t t is 0 (as derived)
    grad_t = 0.0

    # Gradient w.r.t h: sum_lag p_lag * d(corr_lag)/dh
    # corr_k = sum_i h[i] * (1 - h[i+k]) for valid i+k
    # d(corr_k)/dh[i] = (1 - h[i+k]) if 0 <= i+k < N  (from term where i is first index)
    #                 - h[i-k] if 0 <= i-k < N        (from term where i is second index)
    grad_h = np.zeros(N)
    
    # Vectorized approach: for each lag k, d(corr_k)/dh is a vector
    # We can compute this more efficiently using convolution-like operations
    # But for clarity and correctness, let's use a loop over lags but vectorize over i
    
    for k_idx in range(len(p)):
        pk = p[k_idx]
        if pk < 1e-15:
            continue
        k = k_idx - (N - 1)  # actual lag
        
        if k >= 0:
            # k >= 0: corr_k = sum_{i=0}^{N-1-k} h[i] * (1 - h[i+k])
            # d/dh[i] for i in [0, N-1-k]: (1 - h[i+k])
            # d/dh[i] for i in [N-k, N-1]: -h[i-k]
            if N - 1 - k >= 0:
                grad_h[:N-k] += pk * (1.0 - h[k:])
            if k > 0:
                grad_h[k:] -= pk * h[:N-k]
        else:
            # k < 0: let k' = -k > 0
            # corr_k = sum_{i=k'}^{N-1} h[i] * (1 - h[i-k'])
            # d/dh[i] for i in [k', N-1]: (1 - h[i-k'])
            # d/dh[i] for i in [0, k'-1]: -h[i+k']
            k_pos = -k
            if k_pos < N:
                grad_h[k_pos:] += pk * (1.0 - h[:N-k_pos])
                if k_pos > 0:
                    grad_h[:k_pos] -= pk * h[k_pos:]
    
    return np.concatenate([grad_h, [grad_t]])

def _parametric_h(params):
    """
    Generate h from 9 parameters: [a1, p1, f1, a2, p2, f2, a3, p3, f3]
    h_raw[i] = 0.5 + a1*sin(2*pi*f1*i/N + p1) + a2*sin(2*pi*f2*i/N + p2) + a3*sin(2*pi*f3*i/N + p3)
    Then project to feasible set.
    """
    a1, p1, f1, a2, p2, f2, a3, p3, f3 = params
    i = np.arange(N)
    h_raw = 0.5 + a1 * np.sin(2 * np.pi * f1 * i / N + p1) + \
                    a2 * np.sin(2 * np.pi * f2 * i / N + p2) + \
                    a3 * np.sin(2 * np.pi * f3 * i / N + p3)
    return _project_feasible(h_raw)

def _parametric_objective(params):
    h = _parametric_h(params)
    return _correlation_max(h)

def _parametric_objective_grad(params):
    """
    Numerical gradient of the parametric objective.
    """
    eps = 1e-6
    grad = np.zeros(len(params))
    f0 = _parametric_objective(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        grad[i] = (_parametric_objective(params_plus) - f0) / eps
    return grad

def _coordinate_polish(h, max_iter=200):
    """
    Coordinate descent: for each i, optimize h[i] given others,
    projecting to keep sum = TARGET_SUM and 0<=h<=1.
    """
    h = _project_feasible(h).copy()
    best_val = _correlation_max(h)
    for it in range(max_iter):
        improved = False
        for i in range(N):
            h_old = h[i]
            candidates = [0.0, 1.0, h_old]
            for dx in [-0.1, -0.05, 0.05, 0.1]:
                candidates.append(h_old + dx)
            best_x = h_old
            best_v = best_val
            for x in candidates:
                x = max(0.0, min(1.0, x))
                h_test = h.copy()
                h_test[i] = x
                h_test = _project_feasible(h_test)
                v = _correlation_max(h_test)
                if v < best_v - 1e-12:
                    best_v = v
                    best_x = h_test.copy()
            if best_v < best_val - 1e-12:
                h = best_x
                best_val = best_v
                improved = True
        if not improved:
            break
    return h, best_val

def construct():
    start_time = time.time()

    best_h = np.full(N, 0.5)
    best_val = _correlation_max(best_h)

    # Phase 1: Structured initializations
    starts = [np.full(N, 0.5)]

    # Alternating patterns
    for base in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        alt = np.where(np.arange(N) % 2 == 0, base, 1.0 - base)
        starts.append(_project_feasible(alt))

    # Sine waves
    for freq in [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]:
        for amp in [0.2, 0.3, 0.4, 0.45, 0.49]:
            t = np.arange(N)
            h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * t / N)
            starts.append(_project_feasible(h_sin))
            h_cos = 0.5 + amp * np.cos(2 * np.pi * freq * t / N)
            starts.append(_project_feasible(h_cos))

    # Two-block patterns
    for split in [20, 30, 40, 50, 60, 70, 80]:
        h_block = np.zeros(N)
        h_block[:split] = 1.0
        h_block = _project_feasible(h_block)
        starts.append(h_block)
        h_block2 = np.zeros(N)
        h_block2[split:] = 1.0
        h_block2 = _project_feasible(h_block2)
        starts.append(h_block2)

    # Linear ramps
    starts.append(_project_feasible(np.linspace(0.0, 1.0, N)))
    starts.append(_project_feasible(np.linspace(1.0, 0.0, N)))

    # Triangular
    t = np.arange(N)
    tri = 1.0 - np.abs(t - N / 2.0) / (N / 2.0)
    starts.append(_project_feasible(tri))
    tri2 = np.abs(t - N / 2.0) / (N / 2.0)
    starts.append(_project_feasible(tri2))

    # Random smooth seeds
    rng = np.random.RandomState(42)
    for _ in range(15):
        noise = rng.randn(N)
        kernel = np.ones(9) / 9.0
        smooth = np.convolve(noise, kernel, mode='same')
        h_pert = 0.5 + 0.4 * smooth
        starts.append(_project_feasible(h_pert))

    # Random blocky seeds
    for _ in range(10):
        h_rand = rng.rand(N)
        block_size = rng.randint(3, 15)
        for b in range(0, N, block_size):
            h_rand[b:b+block_size] = h_rand[b]
        starts.append(_project_feasible(h_rand))

    # Phase 2: Refine with smooth epigraph SLSQP
    bounds = [(0.0, 1.0) for _ in range(N)] + [(-1.0, 2.0)]
    beta = 20.0

    for h0 in starts:
        if time.time() - start_time > DEADLINE - 30:
            break
        try:
            x0 = np.concatenate([h0, [_correlation_max(h0) * N / 2.0 + 0.1]])
            res = minimize(
                _smooth_objective,
                x0,
                args=(beta,),
                jac=lambda x, b=beta: _smooth_objective_grad(x, b),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            h_res = _project_feasible(res.x[:N])
            val = _correlation_max(h_res)
            if val < best_val - 1e-10:
                best_val = val
                best_h = h_res.copy()
        except Exception:
            pass

    # Phase 3: Parametric optimization (9-dimensional sinusoid sum)
    if time.time() - start_time < DEADLINE - 20:
        # Try several parametric initializations
        param_inits = [
            [0.3, 0, 1, 0.2, np.pi/2, 2, 0.1, np.pi, 3],
            [0.4, 0, 1, 0.3, 0, 2, 0.2, 0, 3],
            [0.3, np.pi/4, 1, 0.2, 0, 3, 0.1, 0, 5],
            [0.2, 0, 2, 0.3, 0, 3, 0.1, 0, 4],
            [0.3, 0, 1, 0.2, 0, 4, 0.1, 0, 7],
            [0.25, 0, 1, 0.25, 0, 2, 0.15, 0, 3],
        ]
        
        param_bounds = [
            (0.0, 0.49), (0, 2*np.pi), (1, 15),
            (0.0, 0.49), (0, 2*np.pi), (1, 15),
            (0.0, 0.49), (0, 2*np.pi), (1, 15),
        ]
        
        for p0 in param_inits:
            if time.time() - start_time > DEADLINE - 15:
                break
            try:
                # Use Nelder-Mead for the 9-dimensional parametric problem
                res = minimize(
                    _parametric_objective,
                    p0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                h_res = _parametric_h(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass
        
        # Also try L-BFGS-B on the parametric form with numerical gradient
        for p0 in param_inits[:3]:
            if time.time() - start_time > DEADLINE - 10:
                break
            try:
                res = minimize(
                    _parametric_objective,
                    p0,
                    jac=_parametric_objective_grad,
                    method='L-BFGS-B',
                    bounds=param_bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                h_res = _parametric_h(res.x)
                val = _correlation_max(h_res)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_res.copy()
            except Exception:
                pass

    # Phase 4: Coordinate polish on best candidate
    if time.time() - start_time < DEADLINE - 8:
        try:
            h_polished, val_polished = _coordinate_polish(best_h, max_iter=150)
            if val_polished < best_val - 1e-12:
                best_val = val_polished
                best_h = h_polished
        except Exception:
            pass

    # Phase 5: Final high-precision SLSQP polish
    if time.time() - start_time < DEADLINE - 5:
        try:
            beta2 = 50.0
            x0 = np.concatenate([best_h, [_correlation_max(best_h) * N / 2.0 + 0.05]])
            res = minimize(
                _smooth_objective,
                x0,
                args=(beta2,),
                jac=lambda x, b=beta2: _smooth_objective_grad(x, b),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False}
            )
            h_final = _project_feasible(res.x[:N])
            val_final = _correlation_max(h_final)
            if val_final < best_val - 1e-13:
                best_val = val_final
                best_h = h_final
        except Exception:
            pass

    # Ensure final validity
    best_h = _project_feasible(best_h)

    # Phase 6: Additional coordinate descent passes with finer steps
    if time.time() - start_time < DEADLINE - 2:
        try:
            for _ in range(3):
                if time.time() - start_time > DEADLINE - 1:
                    break
                h_polished, val_polished = _coordinate_polish(best_h, max_iter=50)
                if val_polished < best_val - 1e-12:
                    best_val = val_polished
                    best_h = h_polished
        except Exception:
            pass

    # Final projection to ensure validity
    best_h = _project_feasible(best_h)

    return best_h
