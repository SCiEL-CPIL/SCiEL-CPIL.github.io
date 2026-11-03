"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines FFT-accelerated linear overlap with a hybrid optimization strategy:
1. Structured sinusoidal parameterization (3-D search) for fast global exploration.
2. Smooth LogSumExp surrogate with analytical gradient for SLSQP/Nesterov refinement.
3. Proper Euclidean projection onto {0<=h<=1, sum=N/2} via bisection.
4. Coordinate descent and epigraph polish for final precision.
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


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective (LogSumExp).
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    shifted = beta * (cs - m)
    shifted = np.clip(shifted, -500, 500)
    exp_terms = np.exp(shifted)
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta

    grad = np.zeros(n)

    for k in range(n):
        if w[k] < 1e-10:
            continue
        factor = -(2.0 / n) * w[k]
        if k > 0:
            grad[k:] += factor * h[:n - k]
        if k < n:
            grad[:n - k] += factor * h[k:]

    return val, grad


def _project_sum_box(y, target_sum):
    """
    Project y onto {x : 0 <= x_i <= 1, sum(x) = target_sum} via bisection on shift tau.
    """
    lo = np.min(y) - 1.5
    hi = np.max(y) + 1.5

    def g(tau):
        x = np.clip(y + tau, 0.0, 1.0)
        return np.sum(x) - target_sum

    if g(lo) > 0:
        return np.clip(y + lo, 0.0, 1.0)
    if g(hi) < 0:
        return np.clip(y + hi, 0.0, 1.0)

    for _ in range(60):
        mid = (lo + hi) / 2.0
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid

    tau = (lo + hi) / 2.0
    return np.clip(y + tau, 0.0, 1.0)


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    h = _project_sum_box(h, N / 2.0)
    # Final tiny correction
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmax(h)
        else:
            idx = np.argmin(h)
        h[idx] += diff
        if h[idx] < 0:
            h[idx] = 0.0
        elif h[idx] > 1:
            h[idx] = 1.0
    return h


def _coordinate_descent(h, max_sweeps=50, tol=1e-10):
    """
    Coordinate descent to minimize the true max overlap.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    n = len(h)
    best_score = _score(h)

    for sweep in range(max_sweeps):
        improved = False
        for j in range(n):
            cs = _linear_overlap_values(h)
            current_max = np.max(cs)

            old_val = h[j]

            a = np.zeros(n)
            for k in range(n):
                val = 0.0
                if j >= k:
                    val += h[j - k]
                if j + k < n:
                    val += h[j + k]
                a[k] = val

            candidates = [old_val, 0.0, 1.0]

            top_k = np.argmax(cs)
            if a[top_k] > 0:
                candidates.append(min(1.0, old_val + 0.1))
                candidates.append(min(1.0, old_val + 0.01))
            elif a[top_k] < 0:
                candidates.append(max(0.0, old_val - 0.1))
                candidates.append(max(0.0, old_val - 0.01))

            top2 = np.argsort(cs)[-2:][::-1]
            if len(top2) == 2:
                k1, k2 = top2[0], top2[1]
                if abs(a[k1] - a[k2]) > 1e-12:
                    x_eq = old_val + (n / 2.0) * (cs[k1] - cs[k2]) / (a[k1] - a[k2])
                    if 0 <= x_eq <= 1:
                        candidates.append(x_eq)

            best_x = old_val
            best_local_max = current_max

            for x in candidates:
                x = max(0.0, min(1.0, x))
                h[j] = x
                new_cs = _linear_overlap_values(h)
                new_max = np.max(new_cs)
                if new_max < best_local_max - 1e-12:
                    best_local_max = new_max
                    best_x = x

            h[j] = best_x

            diff = N / 2.0 - np.sum(h)
            if abs(diff) > 1e-14:
                h = _project_sum_box(h, N / 2.0)

            if best_local_max < best_score - 1e-10:
                best_score = best_local_max
                improved = True

        if not improved:
            break

    return h, best_score


def _epigraph_optimize(h0, maxiter=200, ftol=1e-12):
    """
    Epigraph formulation: minimize t subject to C_k(h) <= t for all k,
    sum(h) = N/2, 0 <= h <= 1.
    """
    n = len(h0)
    x0 = np.concatenate([h0.copy(), [max(_score(h0), 0.5)]])

    def obj(x):
        return x[-1]

    def obj_grad(x):
        g = np.zeros(n + 1)
        g[-1] = 1.0
        return g

    def constraint_max(x):
        h = x[:n]
        t = x[-1]
        cs = _linear_overlap_values(h)
        return t - np.max(cs)

    def constraint_max_grad(x):
        h = x[:n]
        t = x[-1]
        cs = _linear_overlap_values(h)
        top_k = np.argmax(cs)
        g = np.zeros(n + 1)
        g[-1] = 1.0
        for j in range(n):
            val = 0.0
            if j >= top_k:
                val += h[j - top_k]
            if j + top_k < n:
                val += h[j + top_k]
            g[j] = (2.0 / n) * val
        return g

    cons = [
        {'type': 'eq', 'fun': lambda x: np.sum(x[:n]) - N / 2.0,
         'jac': lambda x: np.concatenate([np.ones(n), [0.0]])},
        {'type': 'ineq', 'fun': constraint_max,
         'jac': constraint_max_grad}
    ]

    bounds = [(0.0, 1.0)] * n + [(0.0, 2.0)]

    try:
        res = minimize(obj, x0, jac=obj_grad, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': maxiter, 'ftol': ftol})
        if res.success:
            return res.x[:n]
    except Exception:
        pass
    return None


def _nesterov_projected(h0, beta, max_iter=2000, lr_init=0.01, deadline=None):
    """
    Nesterov accelerated projected gradient descent on the smooth max objective.
    """
    h = h0.copy()
    v = np.zeros(N)
    lr = lr_init
    prev_val = _score(h)

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        val, grad = _smooth_max_grad(h, beta)

        v_new = 0.9 * v - lr * grad
        x_new = h + v_new
        x_new = _project_sum_box(x_new, N / 2.0)

        new_val = _score(x_new)
        if new_val < prev_val - 1e-12:
            h = x_new
            v = v_new
            prev_val = new_val
        else:
            lr *= 0.9
            v = 0.9 * v
            x_try = h - 0.1 * lr * grad
            x_try = _project_sum_box(x_try, N / 2.0)
            val_try = _score(x_try)
            if val_try < prev_val - 1e-12:
                h = x_try
                v = 0.9 * v - 0.1 * lr * grad
                prev_val = val_try

    return h


def _pgd_optimize(h0, beta, steps, lr, deadline):
    """
    Projected Gradient Descent optimization with backtracking.
    """
    h = h0.copy()
    current_lr = lr
    prev_val, _ = _smooth_max_grad(h, beta)

    for i in range(steps):
        if time.time() > deadline:
            break

        val, grad = _smooth_max_grad(h, beta)

        step_lr = current_lr
        new_h = h - step_lr * grad
        new_h_proj = _project_sum_box(new_h, N / 2.0)
        new_val, _ = _smooth_max_grad(new_h_proj, beta)

        while new_val > prev_val and step_lr > 1e-8:
            step_lr /= 2.0
            new_h = h - step_lr * grad
            new_h_proj = _project_sum_box(new_h, N / 2.0)
            new_val, _ = _smooth_max_grad(new_h_proj, beta)

        if new_val < prev_val:
            h = new_h_proj
            prev_val = new_val
            if i % 10 == 0:
                current_lr = min(current_lr * 1.05, 0.1)
        else:
            current_lr = max(current_lr * 0.5, 1e-6)

    return h


def _run_slsqp(h0, beta, maxiter, ftol):
    """Run SLSQP on the smooth max objective."""
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
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
        if res.success or res.fun < float('inf'):
            return res.x
    except Exception:
        pass
    return None


def _sinusoidal_candidate(freq, phase, amp):
    """
    Generate a candidate h from a sinusoidal parameterization:
    h_i = clip(0.5 + amp * sin(2*pi*freq*i/N + phase), 0, 1)
    Then project to sum=N/2.
    """
    i = np.arange(N)
    h = 0.5 + amp * np.sin(2.0 * np.pi * freq * i / N + phase)
    h = np.clip(h, 0.0, 1.0)
    h = _project_sum_box(h, N / 2.0)
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _project_sum_box(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Phase 1: Structured sinusoidal search (fast 3-D grid)
    # Explore frequency, phase, amplitude space
    i = np.arange(N)

    # Standard candidates
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        for phase in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]:
            for amp in [0.3, 0.5, 0.7, 0.9]:
                h_cand = _sinusoidal_candidate(freq, phase, amp)
                _update_best(h_cand)

    # Chirp-like candidates
    for alpha in [0.1, 0.2, 0.5, 1.0, 2.0]:
        for phase in [0, np.pi / 2]:
            h_chirp = 0.5 + 0.5 * np.cos(2.0 * np.pi * alpha * (i ** 2) / N + phase)
            h_chirp = _project_sum_box(h_chirp, N / 2.0)
            _update_best(h_chirp)

    # Random starts
    for _ in range(10):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project_sum_box(h_rand, N / 2.0)
        _update_best(h_rand)

    # Phase 2: Coarse PGD refinement
    if best_h is not None and time.time() - t0 < _DEADLINE - 80:
        h_opt = _pgd_optimize(best_h.copy(), beta=10.0, steps=300, lr=0.05,
                              deadline=_DEADLINE - 80)
        _update_best(h_opt)

    # Phase 3: SLSQP refinement
    if best_h is not None and time.time() - t0 < _DEADLINE - 50:
        h_slsqp = _run_slsqp(best_h.copy(), beta=50.0, maxiter=200, ftol=1e-9)
        if h_slsqp is not None:
            _update_best(h_slsqp)

    # Phase 4: Medium PGD
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h_opt = _pgd_optimize(best_h.copy(), beta=50.0, steps=500, lr=0.01,
                              deadline=_DEADLINE - 30)
        _update_best(h_opt)

    # Phase 5: Nesterov refinement
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        h_nest = _nesterov_projected(best_h.copy(), beta=50.0, max_iter=3000,
                                     lr_init=0.005, deadline=_DEADLINE - 20)
        _update_best(h_nest)

    # Phase 6: High-beta PGD polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_opt = _pgd_optimize(best_h.copy(), beta=100.0, steps=500, lr=0.005,
                              deadline=_DEADLINE - 10)
        _update_best(h_opt)

    # Phase 7: Coordinate descent
    if best_h is not None and time.time() - t0 < _DEADLINE - 6:
        h_cd, score_cd = _coordinate_descent(best_h.copy(), max_sweeps=30, tol=1e-10)
        if score_cd < best_score:
            best_h = h_cd
            best_score = score_cd

    # Phase 8: Ultra-high beta PGD
    if best_h is not None and time.time() - t0 < _DEADLINE - 3:
        h_opt = _pgd_optimize(best_h.copy(), beta=300.0, steps=300, lr=0.002,
                              deadline=_DEADLINE - 3)
        _update_best(h_opt)

    # Phase 9: Final coordinate descent polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 1:
        h_cd2, score_cd2 = _coordinate_descent(best_h.copy(), max_sweeps=20, tol=1e-11)
        if score_cd2 < best_score:
            best_h = h_cd2
            best_score = score_cd2

    # Phase 10: Epigraph final polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 0.5:
        h_epi = _epigraph_optimize(best_h.copy(), maxiter=100, ftol=1e-13)
        if h_epi is not None:
            _update_best(h_epi)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
