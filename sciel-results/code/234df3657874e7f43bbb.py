"""Erdos minimum overlap: smooth-surrogate optimization with exact constraints.
Combines the epigraph formulation (t variable) from the inspiration with
vectorized gradient computation and L-BFGS-B via parameterization for the
sum constraint. Uses multiple diverse starts and annealed beta for the
log-sum-exp soft-max surrogate.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill
_TARGET = N / 2.0
_NFFT = 256  # next power of 2 >= 2*N


def _autocorr(h):
    """Linear autocorrelation r[k] = sum_{i=k}^{N-1} h[i]*h[i-k] for k=0..N-1 via FFT."""
    h_pad = np.zeros(_NFFT)
    h_pad[:N] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H))
    return corr[:N]


def _overlap_values(h):
    """C_k = (2/N)*(N/2 - r_k) for k=0..N-1."""
    n = len(h)
    r = _autocorr(h)
    return (2.0 / n) * (n / 2.0 - r)


def _score(h):
    """True (hard) maximum overlap value."""
    cs = _overlap_values(h)
    return float(np.max(cs))


def _project(h):
    """Project h to sum=N/2, values in [0,1] (Euclidean-ish projection)."""
    h = np.array(h, dtype=np.float64, copy=True)
    h = np.clip(h, 0.0, 1.0)
    diff = _TARGET - np.sum(h)
    if abs(diff) > 1e-14:
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        cnt = np.sum(mask)
        if cnt > 0:
            h[mask] += diff / cnt
            h = np.clip(h, 0.0, 1.0)
    # Final exact correction
    diff = _TARGET - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max_and_grad(x, beta):
    """Log-sum-exp soft-max of x and its gradient (softmax(beta*x))."""
    m = np.max(x)
    e = np.exp(beta * (x - m))
    s = np.sum(e)
    val = (m + np.log(s)) / beta
    grad = e / s  # gradient w.r.t. x
    return val, grad


def _smooth_obj_and_grad(h, beta):
    """f(h) = S_beta(C(h)); returns (f, df/dh). Vectorized."""
    cs = _overlap_values(h)
    val, g_c = _smooth_max_and_grad(cs, beta)
    n = N
    g_h = np.zeros(n)
    factor = -(2.0 / n)
    for k in range(n):
        gc = g_c[k]
        if gc == 0.0:
            continue
        if k == 0:
            g_h += factor * gc * (2.0 * h)
        else:
            g_h[k:] += factor * gc * h[:n - k]
            g_h[:n - k] += factor * gc * h[k:]
    return val, g_h


def _epigraph_obj_and_grad(x, beta):
    """x = [h (N), t (1)]. Objective = t + S_beta(C(h) - t).
    Returns (obj, grad)."""
    h = x[:N]
    t = x[N]
    cs = _overlap_values(h)
    shifted = cs - t
    sm, g_c = _smooth_max_and_grad(shifted, beta)
    obj = t + sm
    g_h = np.zeros(N)
    factor = -(2.0 / N)
    for k in range(N):
        gc = g_c[k]
        if gc == 0.0:
            continue
        if k == 0:
            g_h += factor * gc * (2.0 * h)
        else:
            g_h[k:] += factor * gc * h[:N - k]
            g_h[:N - k] += factor * gc * h[k:]
    g_t = 1.0 - np.sum(g_c)
    grad = np.concatenate([g_h, [g_t]])
    return obj, grad


def _refine_smooth(h0, beta, n_max_iter=3000, time_budget=25.0):
    """Refine h using SLSQP on the smooth surrogate with exact box + sum constraints.
    Parameterize with N-1 free variables to exactly satisfy sum=N/2.
    """
    h0 = _project(h0.copy())
    z0 = h0[:N - 1].copy()

    def _full(z):
        last = _TARGET - np.sum(z)
        h = np.empty(N)
        h[:N - 1] = z
        h[N - 1] = last
        return h

    def _obj_grad(z):
        h = _full(z)
        val, g_h = _smooth_obj_and_grad(h, beta)
        # Chain rule: dz_i/dz = I, d(last)/dz = -1
        g_z = g_h[:N - 1] - g_h[N - 1]
        return val, g_z

    bounds = [(0.0, 1.0)] * (N - 1)
    constraints = [
        {'type': 'ineq', 'fun': lambda z: _TARGET - np.sum(z),
         'jac': lambda z: -np.ones(N - 1)},  # last >= 0
        {'type': 'ineq', 'fun': lambda z: np.sum(z) - (_TARGET - 1.0),
         'jac': lambda z: np.ones(N - 1)},   # last <= 1
    ]

    t_start = time.time()
    res = minimize(
        _obj_grad, z0, jac=True, method='SLSQP', bounds=bounds,
        constraints=constraints,
        options={'maxiter': n_max_iter, 'ftol': 1e-14, 'gtol': 1e-10,
                 'disp': False, 'maxls': 50},
    )
    if time.time() - t_start > time_budget:
        return h0, _score(h0)
    z_opt = res.x
    h_opt = _full(z_opt)
    h_opt = _project(h_opt)
    return h_opt, _score(h_opt)


def _refine_epigraph(h0, beta, n_max_iter=2000, time_budget=20.0):
    """Refine using epigraph formulation with SLSQP.
    x = [h (N), t], minimize t + S_beta(C(h)-t) subject to box on h and sum(h)=N/2.
    """
    h0 = _project(h0.copy())
    t0 = max(0.0, _score(h0))
    x0 = np.concatenate([h0, [t0]])

    def _obj_grad(x):
        h = x[:N]
        t = x[N]
        cs = _overlap_values(h)
        shifted = cs - t
        sm, g_c = _smooth_max_and_grad(shifted, beta)
        obj = t + sm
        g_h = np.zeros(N)
        factor = -(2.0 / N)
        for k in range(N):
            gc = g_c[k]
            if gc == 0.0:
                continue
            if k == 0:
                g_h += factor * gc * (2.0 * h)
            else:
                g_h[k:] += factor * gc * h[:N - k]
                g_h[:N - k] += factor * gc * h[k:]
        g_t = 1.0 - np.sum(g_c)
        grad = np.concatenate([g_h, [g_t]])
        return obj, grad

    bounds = [(0.0, 1.0)] * N + [(-1.0, 2.0)]
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:N]) - _TARGET,
         'jac': lambda x: np.concatenate([np.ones(N), [0.0]])},
    ]

    t_start = time.time()
    res = minimize(
        _obj_grad, x0, jac=True, method='SLSQP', bounds=bounds,
        constraints=constraints,
        options={'maxiter': n_max_iter, 'ftol': 1e-14, 'gtol': 1e-10,
                 'disp': False, 'maxls': 50},
    )
    if time.time() - t_start > time_budget:
        return h0, _score(h0)
    h_opt = _project(res.x[:N])
    return h_opt, _score(h_opt)


def _construct_direct():
    """Direct multi-frequency construction to flatten autocorrelation."""
    i = np.arange(N)
    h = np.full(N, 0.5)
    freqs = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
    amplitudes = [0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
    phases = [0.0, 0.7, 1.3, 2.1, 0.4, 1.8, 2.5, 0.9, 1.6, 0.3]
    for f, a, p in zip(freqs, amplitudes, phases):
        h += a * np.cos(2.0 * np.pi * f * i / N + p)
    return _project(h)


def _random_start(rng):
    """Random feasible start."""
    h = rng.uniform(0.0, 1.0, N)
    return _project(h)


def _perturb(h, rng, scale=0.05):
    """Add small random perturbation and re-project."""
    h_p = h + rng.normal(0.0, scale, N)
    return _project(h_p)


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    rng = np.random.default_rng(42)

    best_h = None
    best_score = float('inf')

    def _update(h):
        nonlocal best_h, best_score
        h = _project(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Phase 1: Direct construction
    h_direct = _construct_direct()
    _update(h_direct)

    # Phase 2: Diverse frequency-based starts
    if time.time() - t0 < _DEADLINE - 60:
        i = np.arange(N)
        freq_sets = [
            ([1, 2, 3, 5], [0.15, 0.10, 0.08, 0.06]),
            ([1, 2, 4, 7], [0.14, 0.10, 0.07, 0.05]),
            ([1, 3, 4, 6, 9], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([2, 3, 5, 8, 13], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([1, 2, 3, 4, 5, 6], [0.10, 0.08, 0.06, 0.05, 0.04, 0.03]),
            ([3, 5, 7, 11], [0.12, 0.10, 0.08, 0.06]),
            ([1, 4, 6, 9, 12], [0.11, 0.09, 0.07, 0.05, 0.03]),
            ([2, 5, 8, 13, 21], [0.10, 0.08, 0.06, 0.04, 0.02]),
        ]
        for freqs, amps in freq_sets:
            if time.time() - t0 > _DEADLINE - 60:
                break
            h = np.full(N, 0.5)
            for f, a in zip(freqs, amps):
                h += a * np.cos(2.0 * np.pi * f * i / N)
            _update(h)

    # Phase 3: Random starts
    if time.time() - t0 < _DEADLINE - 50:
        for _ in range(8):
            if time.time() - t0 > _DEADLINE - 50:
                break
            h_rand = _random_start(rng)
            _update(h_rand)

    # Phase 4: Smooth surrogate refinement with annealed beta
    # Use both parameterized (sum-exact) and epigraph approaches
    beta_list = [15.0, 40.0, 100.0, 250.0, 500.0]
    
    starts_to_refine = []
    if best_h is not None:
        starts_to_refine.append(best_h.copy())
    
    # Add perturbed versions of best
    if best_h is not None and time.time() - t0 < _DEADLINE - 80:
        for _ in range(2):
            if time.time() - t0 > _DEADLINE - 80:
                break
            starts_to_refine.append(_perturb(best_h, rng, scale=0.03))
    
    # Add a couple random starts
    if time.time() - t0 < _DEADLINE - 70:
        for _ in range(2):
            if time.time() - t0 > _DEADLINE - 70:
                break
            starts_to_refine.append(_random_start(rng))

    for beta in beta_list:
        remaining = _DEADLINE - time.time()
        if remaining < 15:
            break
        n_starts = len(starts_to_refine)
        # Allocate time per start; use SLSQP (slower but exact constraints)
        time_per_start = min(18.0, (remaining - 8.0) / max(n_starts, 1))
        if time_per_start < 4.0:
            break
        
        for h_start in starts_to_refine:
            if time.time() - t0 > _DEADLINE - 12:
                break
            # Use parameterized approach (faster, exact sum)
            h_ref, s_ref = _refine_smooth(h_start, beta=beta, n_max_iter=2000,
                                          time_budget=time_per_start)
            _update(h_ref)
        
        # For the best, also try epigraph formulation at higher beta
        if best_h is not None and beta >= 100.0:
            remaining = _DEADLINE - time.time()
            if remaining > 10:
                h_ref, s_ref = _refine_epigraph(best_h, beta=beta, n_max_iter=1500,
                                                time_budget=min(12.0, remaining - 5.0))
                _update(h_ref)
        
        # Update starts: keep best and add perturbation
        if best_h is not None:
            starts_to_refine[0] = best_h.copy()
            # Replace worst with perturbation of best
            if len(starts_to_refine) > 1:
                starts_to_refine[-1] = _perturb(best_h, rng, scale=0.02)

    # Phase 5: Final high-precision polish with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        remaining = _DEADLINE - time.time() - 5.0
        if remaining > 5.0:
            h_polish, s_polish = _refine_smooth(best_h, beta=800.0, n_max_iter=5000,
                                                time_budget=min(18.0, remaining))
            _update(h_polish)
        
        # One more epigraph polish if time allows
        if best_h is not None and time.time() - t0 < _DEADLINE - 8:
            remaining = _DEADLINE - time.time() - 4.0
            if remaining > 4.0:
                h_polish, s_polish = _refine_epigraph(best_h, beta=800.0, n_max_iter=2000,
                                                      time_budget=min(10.0, remaining))
                _update(h_polish)

    # Phase 6: Local perturbation search around best
    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        for scale in [0.02, 0.01, 0.005]:
            if time.time() - t0 > _DEADLINE - 15:
                break
            for _ in range(5):
                if time.time() - t0 > _DEADLINE - 15:
                    break
                h_pert = _perturb(best_h, rng, scale=scale)
                _update(h_pert)
            if best_h is not None:
                # Quick refine of perturbed best
                remaining = _DEADLINE - time.time() - 8.0
                if remaining > 5.0:
                    h_ref, s_ref = _refine_smooth(best_h, beta=500.0, n_max_iter=1000,
                                                  time_budget=min(8.0, remaining))
                    _update(h_ref)

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _project(best_h)
