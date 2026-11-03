import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 135  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
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


def _project_sum_box(y, target_sum):
    """
    Project y onto {x : 0 <= x_i <= 1, sum(x) = target_sum}
    """
    y = np.asarray(y, dtype=np.float64)
    # Binary search for tau such that sum(clamp(y + tau, 0, 1)) = target_sum
    lo = np.min(y) - 1.0
    hi = np.max(y) + 1.0

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


def _epigraph_obj(x):
    """
    Epigraph objective: minimize t + penalty
    x = [h (N vars), t (1 var)]
    """
    h = x[:N]
    t = x[N]
    cs = _linear_overlap_values(h)
    # Penalty for cs > t
    violations = np.maximum(0, cs - t)
    penalty = 100.0 * np.sum(violations**2)
    # Penalty for bounds
    bound_pen = 0.0
    if np.any(h < -1e-8) or np.any(h > 1 + 1e-8):
        bound_pen = 1000.0 * (np.sum(np.minimum(0, h)**2) + np.sum(np.maximum(0, h - 1)**2))
    # Penalty for sum
    sum_pen = 100.0 * (np.sum(h) - N / 2.0)**2
    return t + penalty + bound_pen + sum_pen


def _epigraph_grad(x):
    """
    Gradient of epigraph objective.
    """
    h = x[:N]
    t = x[N]
    cs = _linear_overlap_values(h)
    n = N

    # d(t + penalty)/dt = 1 + 200 * sum(max(0, cs-t))
    violations = np.maximum(0, cs - t)
    dt = 1.0 + 200.0 * np.sum(violations)

    # d penalty / dh_j = sum_k 200 * max(0, cs_k - t) * dcs_k/dh_j
    # dcs_k/dh_j = -(2/n) * (h_{j-k} if j>=k + h_{j+k} if j+k<n)
    grad_h = np.zeros(n)
    for k in range(n):
        if violations[k] == 0.0:
            continue
        factor = 200.0 * violations[k] * (-(2.0 / n))
        if k > 0:
            grad_h[k:] += factor * h[:n-k]
        if k < n:
            grad_h[:n-k] += factor * h[k:]

    # Bound penalty gradient
    grad_h += 2000.0 * np.minimum(0, h) - 2000.0 * np.maximum(0, h - 1)
    # Sum penalty gradient
    grad_h += 200.0 * (np.sum(h) - N / 2.0)

    grad = np.concatenate([grad_h, [dt]])
    return grad


def _run_slsqp(h0, maxiter=200, time_limit=None):
    """
    Run SLSQP optimization on the epigraph formulation.
    """
    t0 = time.time()
    h = np.asarray(h0, dtype=np.float64)
    n = len(h)

    # Initial t: max of overlap values + small margin
    cs = _linear_overlap_values(h)
    t0_val = np.max(cs) + 0.01

    x0 = np.concatenate([h, [t0_val]])

    # Constraints: sum(h) = N/2
    def sum_constraint(x):
        return np.sum(x[:n]) - n / 2.0

    # Bounds: 0 <= h_i <= 1, t >= 0
    bounds = [(0.0, 1.0)] * n + [(0.0, 2.0)]

    constraints = [
        {'type': 'eq', 'fun': sum_constraint, 'jac': lambda x: np.concatenate([np.ones(n), [0.0]])}
    ]

    best_h = h.copy()
    best_score = _score(h)

    def _callback(xk):
        nonlocal best_h, best_score
        hk = xk[:n]
        hk = _project_sum_box(hk, n / 2.0)
        sk = _score(hk)
        if sk < best_score:
            best_h = hk.copy()
            best_score = sk
        if time_limit is not None and time.time() - t0 > time_limit:
            return True
        return False

    try:
        res = minimize(
            _epigraph_obj, x0, jac=_epigraph_grad,
            method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-10, 'disp': False},
            callback=_callback
        )
        # Check result
        h_res = res.x[:n]
        h_res = _project_sum_box(h_res, n / 2.0)
        s_res = _score(h_res)
        if s_res < best_score:
            best_h = h_res.copy()
            best_score = s_res
    except Exception:
        pass

    return best_h


def _pgd_optimize(h0, beta, steps=300, lr=0.01):
    """
    Projected Gradient Descent optimization with log-sum-exp smoothing.
    """
    h = np.asarray(h0, dtype=np.float64)
    n = len(h)
    target_sum = n / 2.0

    def _soft_max_val_grad(h):
        cs = _linear_overlap_values(h)
        m = np.max(cs)
        exp_terms = np.exp(beta * (cs - m))
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp
        val = m + np.log(sum_exp) / beta

        grad = np.zeros(n)
        for k in range(n):
            if w[k] < 1e-12:
                continue
            factor = -(2.0 / n) * w[k]
            if k > 0:
                grad[k:] += factor * h[:n-k]
            if k < n:
                grad[:n-k] += factor * h[k:]
        return val, grad

    current_lr = lr
    prev_val = _soft_max_val_grad(h)[0]

    for i in range(steps):
        if time.time() > _DEADLINE - 5:
            break

        val, grad = _soft_max_val_grad(h)

        step_lr = current_lr
        new_h = h - step_lr * grad
        new_h_proj = _project_sum_box(new_h, target_sum)
        new_val, _ = _soft_max_val_grad(new_h_proj)

        while new_val > prev_val and step_lr > 1e-8:
            step_lr /= 2.0
            new_h = h - step_lr * grad
            new_h_proj = _project_sum_box(new_h, target_sum)
            new_val, _ = _soft_max_val_grad(new_h_proj)

        if new_val < prev_val:
            h = new_h_proj
            prev_val = new_val
            if i % 10 == 0:
                current_lr = min(current_lr * 1.05, 0.1)
        else:
            current_lr = max(current_lr * 0.5, 1e-6)

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

    # 1. Initial Candidates
    i = np.arange(N)

    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _project_sum_box(h_cos, N / 2.0)
    _update_best(h_cos)

    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    h_sin = _project_sum_box(h_sin, N / 2.0)
    _update_best(h_sin)

    # Double frequency cosine
    h_cos2 = 0.5 + 0.5 * np.cos(4.0 * np.pi * i / N)
    h_cos2 = _project_sum_box(h_cos2, N / 2.0)
    _update_best(h_cos2)

    # Triangle wave
    h_tri = 0.5 + 0.5 * np.abs(2.0 * (i / N - np.floor(i / N + 0.5)))
    h_tri = _project_sum_box(h_tri, N / 2.0)
    _update_best(h_tri)

    # Random starts
    for _ in range(8):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project_sum_box(h_rand, N / 2.0)
        _update_best(h_rand)

    # 2. SLSQP optimization on best candidates
    if best_h is not None and time.time() - t0 < _DEADLINE - 60:
        h_opt = _run_slsqp(best_h, maxiter=150, time_limit=30)
        _update_best(h_opt)

    # 3. PGD with different betas
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h_opt = _pgd_optimize(best_h, beta=20.0, steps=300, lr=0.05)
        _update_best(h_opt)

    if best_h is not None and time.time() - t0 < _DEADLINE - 15:
        h_opt = _pgd_optimize(best_h, beta=100.0, steps=400, lr=0.01)
        _update_best(h_opt)

    # 4. Another SLSQP pass from the improved solution
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_opt = _run_slsqp(best_h, maxiter=100, time_limit=15)
        _update_best(h_opt)

    # 5. Final high-precision PGD polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 1:
        h_opt = _pgd_optimize(best_h, beta=500.0, steps=200, lr=0.001)
        _update_best(h_opt)

    if best_h is None:
        best_h = np.full(N, 0.5)

    # Final projection to ensure exact sum and bounds
    best_h = _project_sum_box(best_h, N / 2.0)

    return best_h
