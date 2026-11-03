import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 145


def _linear_overlap_values(h):
    """Compute all circular overlap values c_k for k=0..N-1."""
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
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0

    if np.all(h >= 0.0) and np.all(h <= 1.0) and abs(np.sum(h) - target) < 1e-14:
        return h

    alpha_low = -max(h) - 1.0
    alpha_high = 1.0 - min(h) + 1.0

    for _ in range(60):
        alpha_mid = (alpha_low + alpha_high) / 2.0
        h_new = np.clip(h + alpha_mid, 0.0, 1.0)
        s = np.sum(h_new)
        if s < target:
            alpha_low = alpha_mid
        else:
            alpha_high = alpha_mid

    h_final = np.clip(h + (alpha_low + alpha_high) / 2.0, 0.0, 1.0)

    diff = target - np.sum(h_final)
    if abs(diff) > 1e-14:
        mask = (h_final > 1e-10) & (h_final < 1.0 - 1e-10)
        if np.any(mask):
            h_final[mask] += diff / np.sum(mask)
            h_final = np.clip(h_final, 0.0, 1.0)

    diff = target - np.sum(h_final)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_final)
        else:
            idx = np.argmax(h_final)
        h_final[idx] += diff
        h_final = np.clip(h_final, 0.0, 1.0)

    return h_final


def _smooth_max_val_grad(h, beta):
    """Return smooth max value and gradient w.r.t. h using FFT-based convolution."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta

    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    conv_wh = np.fft.irfft(W * H, n=n_fft)[:n]

    h_flip = np.zeros(n)
    h_flip[0] = h[0]
    h_flip[1:] = h[1:][::-1]
    h_flip_pad = np.zeros(n_fft)
    h_flip_pad[:n] = h_flip
    HF = np.fft.rfft(h_flip_pad, n=n_fft)
    conv_whf = np.fft.irfft(W * HF, n=n_fft)[:n]

    grad = -(2.0 / n) * (conv_wh + conv_whf)
    return val, grad


def _coordinate_descent(h, beta, max_iter=500):
    """Coordinate descent on individual h[i] to minimize smooth max."""
    h = np.array(h, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    best_val = _score(h)
    best_h = h.copy()

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        improved = False
        for j in range(N):
            if time.time() > _DEADLINE - 5:
                break
            # Try small perturbations of h[j]
            base = h[j]
            for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                h[j] = base + delta
                if h[j] < 0.0 or h[j] > 1.0:
                    h[j] = base
                    continue
                # Fix sum by adjusting another coordinate
                diff = np.sum(h) - N / 2.0
                if abs(diff) < 1e-14:
                    pass
                else:
                    # Adjust the coordinate with most room
                    if diff > 0:
                        k = np.argmin(h)
                        if k != j:
                            h[k] -= diff
                            if h[k] < 0.0:
                                h[k] = 0.0
                                h[j] = base
                                continue
                    else:
                        k = np.argmax(h)
                        if k != j:
                            h[k] -= diff
                            if h[k] > 1.0:
                                h[k] = 1.0
                                h[j] = base
                                continue

                s = _score(h)
                if s < best_val - 1e-12:
                    best_val = s
                    best_h = h.copy()
                    improved = True
                    break
                else:
                    h[j] = base
            h[j] = base

        if not improved and it > 20:
            break

    return best_h, best_val


def _accelerated_smooth_max(h0, beta, max_iter=1000, lr_init=0.05):
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    h_prev = h.copy()
    t_k = 1.0
    lr = lr_init

    best_val = _score(h)
    best_h = h.copy()

    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break

        val, grad = _smooth_max_val_grad(h, beta)
        h_new = h - lr * grad
        h_new = _fix_sum(h_new)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        h_next = h_new + ((t_k - 1.0) / t_new) * (h_new - h_prev)
        h_next = _fix_sum(h_next)

        h_prev = h
        h = h_next
        t_k = t_new

        if it % 100 == 99:
            lr = max(0.001, min(0.1, lr * 0.95))

        if it % 50 == 49:
            current_score = _score(h)
            if current_score < best_val:
                best_val = current_score
                best_h = h.copy()

    return best_h, best_val


def _solve_sdp_like(h0, beta=50.0, max_iter=200):
    """
    Structural approach: treat this as minimizing max of quadratic forms.
    Use a projected gradient method with adaptive step size and
    periodic coordinate descent polish.
    """
    h = np.array(h0, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    best_val = _score(h)
    best_h = h.copy()

    lr = 0.02
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break

        val, grad = _smooth_max_val_grad(h, beta)
        h_new = h - lr * grad
        h_new = _fix_sum(h_new)

        # Adaptive step: if it didn't improve, halve lr
        s_new = _score(h_new)
        if s_new > val * 1.001:
            lr *= 0.5
            h_new = h
        else:
            h = h_new
            lr = min(lr * 1.1, 0.05)

        if it % 30 == 29:
            h_cd, s_cd = _coordinate_descent(h, beta, max_iter=50)
            if s_cd < best_val:
                best_val = s_cd
                best_h = h_cd.copy()
                h = h_cd.copy()

        if it % 50 == 49:
            s_curr = _score(h)
            if s_curr < best_val:
                best_val = s_curr
                best_h = h.copy()

    return best_h, best_val


def _random_restart_optimize(h0, beta, n_restarts=3):
    """Try small random perturbations and optimize from each."""
    best_h = h0.copy()
    best_val = _score(h0)
    np.random.seed(123)

    for r in range(n_restarts):
        if time.time() > _DEADLINE - 10:
            break
        h_pert = h0 + np.random.randn(N) * 0.05
        h_pert = _fix_sum(h_pert)
        h_opt, s_opt = _accelerated_smooth_max(h_pert, beta, max_iter=200)
        if s_opt < best_val:
            best_val = s_opt
            best_h = h_opt.copy()

    return best_h, best_val


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

    i = np.arange(N)

    # Phase 1: Generate diverse structured starts
    starts = []

    for _ in range(6):
        h_rand = np.random.rand(N)
        starts.append(_fix_sum(h_rand))

    # Sine/cosine starts with various frequencies
    for f in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]:
        for amp in [0.2, 0.3, 0.4, 0.45]:
            h_sine = 0.5 + amp * np.sin(2 * np.pi * f * i / N)
            starts.append(_fix_sum(h_sine))
            h_cos = 0.5 + amp * np.cos(2 * np.pi * f * i / N)
            starts.append(_fix_sum(h_cos))

    # Multi-frequency starts
    for f1, f2, a1, a2 in [(1, 2, 0.3, 0.2), (1, 3, 0.25, 0.25),
                            (2, 3, 0.2, 0.3), (1, 4, 0.3, 0.15),
                            (2, 5, 0.2, 0.2), (3, 5, 0.15, 0.25)]:
        h_multi = 0.5 + a1 * np.cos(2 * np.pi * f1 * i / N) + a2 * np.sin(2 * np.pi * f2 * i / N)
        starts.append(_fix_sum(h_multi))

    # Binary-like starts
    for period in [2, 3, 4, 5, 7, 10]:
        h_bin = np.array([1.0 if (j % period) < period // 2 else 0.0 for j in range(N)])
        h_pert = h_bin * 0.7 + 0.3 * np.random.rand(N)
        starts.append(_fix_sum(h_pert))

    # Sawtooth-like
    h_saw = np.linspace(0, 1, N)
    starts.append(_fix_sum(h_saw))
    h_saw2 = np.linspace(1, 0, N)
    starts.append(_fix_sum(h_saw2))

    # Triangle wave
    h_tri = 0.5 + 0.4 * np.abs(2 * (i / N - np.floor(i / N + 0.5)))
    starts.append(_fix_sum(h_tri))

    for h0 in starts:
        _update_best(h0)

    # Phase 2: Optimize from best start with adaptive method
    if best_h is not None and time.time() < _DEADLINE - 40:
        for beta in [30.0, 50.0, 80.0]:
            if time.time() > _DEADLINE - 20:
                break
            h_candidate, s_candidate = _solve_sdp_like(best_h, beta=beta, max_iter=150)
            if s_candidate < best_score:
                _update_best(h_candidate)

    # Phase 3: Nesterov acceleration
    if best_h is not None and time.time() < _DEADLINE - 20:
        for beta in [40.0, 60.0, 100.0]:
            if time.time() > _DEADLINE - 10:
                break
            h_candidate, s_candidate = _accelerated_smooth_max(best_h, beta=beta, max_iter=250)
            if s_candidate < best_score:
                _update_best(h_candidate)

    # Phase 4: Random restarts
    if best_h is not None and time.time() < _DEADLINE - 15:
        h_candidate, s_candidate = _random_restart_optimize(best_h, beta=50.0, n_restarts=3)
        if s_candidate < best_score:
            _update_best(h_candidate)

    # Phase 5: Coordinate descent polish
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_candidate, s_candidate = _coordinate_descent(best_h, beta=80.0, max_iter=300)
        if s_candidate < best_score:
            _update_best(h_candidate)

    # Phase 6: SLSQP polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        def _run_optimization(h0, beta, maxiter, ftol):
            nonlocal best_h, best_score
            cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                     'jac': lambda x: np.ones(N)}]
            bounds = [(0.0, 1.0)] * N

            def obj_func(x):
                val, _ = _smooth_max_val_grad(x, beta)
                return val

            def jac_func(x):
                _, grad = _smooth_max_val_grad(x, beta)
                return grad

            try:
                res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                               bounds=bounds, constraints=cons,
                               options={'maxiter': maxiter, 'ftol': ftol})
                if res.success or res.fun < best_score:
                    _update_best(res.x)
            except Exception:
                pass

        _run_optimization(best_h, beta=100.0, maxiter=80, ftol=1e-10)

        if time.time() < _DEADLINE - 3:
            _run_optimization(best_h, beta=200.0, maxiter=60, ftol=1e-12)

    # Final coordinate polish
    if best_h is not None and time.time() < _DEADLINE - 2:
        h_candidate, s_candidate = _coordinate_descent(best_h, beta=150.0, max_iter=100)
        if s_candidate < best_score:
            _update_best(h_candidate)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
