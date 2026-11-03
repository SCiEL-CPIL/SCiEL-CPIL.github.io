"""Erdos minimum overlap: smooth-surrogate + epigraph optimization.
Uses log-sum-exp soft-max of the overlap values as a differentiable surrogate
for the max, optimized with L-BFGS-B under box + sum constraints, with
multiple diverse starts and a final high-precision polish.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill
_TARGET = N / 2.0


def _autocorr(h):
    """Linear autocorrelation r[k] = sum_{i=k}^{N-1} h[i]*h[i-k] for k=0..N-1 via FFT."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    return corr[:n]


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
    """f(h) = S_beta(C(h)); returns (f, df/dh)."""
    cs = _overlap_values(h)
    val, g_c = _smooth_max_and_grad(cs, beta)
    # dC_k/dh_j: C_k = (2/N)(N/2 - r_k), r_k = sum_{i=k}^{N-1} h_i h_{i-k}
    # d r_k / d h_j = h_{j-k} (if j>=k) + h_{j+k} (if j+k<n), for j contributing
    n = len(h)
    g_h = np.zeros(n)
    factor = -(2.0 / n)
    for k in range(n):
        gc = g_c[k]
        if gc == 0.0:
            continue
        # contribution of r_k to C_k is factor*gc * d r_k/d h_j
        if k == 0:
            # r_0 = sum h_i^2, d/dh_j = 2 h_j
            g_h += factor * gc * (2.0 * h)
        else:
            # d r_k/d h_j = h_{j-k} for j in [k, n-1] plus h_{j+k} for j in [0, n-k-1]
            # g_h[j] += factor*gc*( h_{j-k} + h_{j+k} )
            # vectorize:
            # part A: j in [k, n-1]: h_{j-k} = h[0..n-k-1]
            g_h[k:] += factor * gc * h[:n - k]
            # part B: j in [0, n-k-1]: h_{j+k} = h[k..n-1]
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


def _refine_smooth(h0, beta, n_max_iter=2000, time_budget=30.0):
    """Refine h using L-BFGS-B on the smooth surrogate with box + sum constraint.
    We handle the sum constraint by optimizing over a free variable and projecting,
    but L-BFGS-B handles box constraints; we enforce sum via a penalty-free trick:
    optimize in N-1 free vars (last = target - sum(others)) to exactly satisfy sum.
    """
    h0 = _project(h0.copy())
    # Parameterize: z (N-1 free), last = target - sum(z). Enforce 0<=z<=1 and 0<=last<=1.
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
        # dz/dz = I for first N-1, d(last)/dz = -1
        g_z = np.empty(N - 1)
        g_z[:] = g_h[:N - 1] - g_h[N - 1]
        return val, g_z

    bounds = [(0.0, 1.0)] * (N - 1)
    # Also need last in [0,1]: that's a linear constraint on z: 0 <= target - sum(z) <= 1
    # i.e. target-1 <= sum(z) <= target. Add as linear constraints.
    A_eq = None
    A_ub = np.ones(N - 1)
    b_ub = _TARGET  # sum(z) <= target  -> last >= 0
    b_lb = _TARGET - 1.0  # sum(z) >= target-1 -> last <= 1
    # L-BFGS-B supports only bounds, not general linear constraints.
    # So we instead use a penalty-free approach: use SLSQP which supports A_eq/A_ub.
    # But SLSQP can be slower. Given N=100, it's fine.

    def _obj_grad_slqp(z):
        h = _full(z)
        val, g_h = _smooth_obj_and_grad(h, beta)
        g_z = g_h[:N - 1] - g_h[N - 1]
        return val, g_z

    constraints = [
        {'type': 'ineq', 'fun': lambda z: _TARGET - np.sum(z), 'jac': lambda z: -np.ones(N - 1)},  # last >= 0
        {'type': 'ineq', 'fun': lambda z: np.sum(z) - (_TARGET - 1.0), 'jac': lambda z: np.ones(N - 1)},  # last <= 1
    ]

    t_start = time.time()
    res = minimize(
        _obj_grad_slqp, z0, jac=True, method='SLSQP', bounds=bounds,
        constraints=constraints,
        options={'maxiter': n_max_iter, 'ftol': 1e-12, 'disp': False,
                 'maxls': 50},
    )
    if time.time() - t_start > time_budget:
        return h0, _score(h0)
    z_opt = res.x
    h_opt = _full(z_opt)
    h_opt = _project(h_opt)
    return h_opt, _score(h_opt)


def _construct_direct():
    """Direct multi-frequency construction to flatten autocorrelation."""
    i = np.arange(N)
    h = np.zeros(N)
    h += 0.5
    freqs = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0]
    amplitudes = [0.15, 0.10, 0.08, 0.06, 0.04, 0.03]
    phases = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    for f, a, p in zip(freqs, amplitudes, phases):
        h += a * np.cos(2.0 * np.pi * f * i / N + p)
    return h


def _random_start(rng):
    """Random feasible start."""
    h = rng.uniform(0.0, 1.0, N)
    return _project(h)


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

    # Phase 2: Several diverse frequency-based starts
    if time.time() - t0 < _DEADLINE - 40:
        i = np.arange(N)
        freq_sets = [
            ([1, 2, 3, 5], [0.15, 0.10, 0.08, 0.06]),
            ([1, 2, 4, 7], [0.14, 0.10, 0.07, 0.05]),
            ([1, 3, 4, 6, 9], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([2, 3, 5, 8, 13], [0.12, 0.10, 0.08, 0.06, 0.04]),
            ([1, 2, 3, 4, 5, 6], [0.10, 0.08, 0.06, 0.05, 0.04, 0.03]),
            ([3, 5, 7, 11], [0.12, 0.10, 0.08, 0.06]),
        ]
        for freqs, amps in freq_sets:
            if time.time() - t0 > _DEADLINE - 40:
                break
            h = np.full(N, 0.5)
            for f, a in zip(freqs, amps):
                h += a * np.cos(2.0 * np.pi * f * i / N)
            _update(h)

    # Phase 3: Random starts
    if time.time() - t0 < _DEADLINE - 30:
        for _ in range(4):
            if time.time() - t0 > _DEADLINE - 30:
                break
            h_rand = _random_start(rng)
            _update(h_rand)

    # Phase 4: Smooth surrogate refinement from best and a few others
    beta_list = [20.0, 40.0, 80.0]
    starts_to_refine = []
    if best_h is not None:
        starts_to_refine.append(best_h.copy())

    # Add a couple more diverse starts if available
    if time.time() - t0 < _DEADLINE - 60:
        for _ in range(2):
            if time.time() - t0 > _DEADLINE - 60:
                break
            starts_to_refine.append(_random_start(rng))

    for beta in beta_list:
        if time.time() - t0 > _DEADLINE - 20:
            break
        for h_start in starts_to_refine:
            if time.time() - t0 > _DEADLINE - 15:
                break
            h_ref, s_ref = _refine_smooth(h_start, beta=beta, n_max_iter=1500,
                                          time_budget=15.0)
            _update(h_ref)

    # Phase 5: Final high-precision polish with very high beta (near-exact max)
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_polish, s_polish = _refine_smooth(best_h, beta=200.0, n_max_iter=2000,
                                            time_budget=10.0)
        _update(h_polish)

    if best_h is None:
        best_h = np.full(N, 0.5)

    return _project(best_h)
