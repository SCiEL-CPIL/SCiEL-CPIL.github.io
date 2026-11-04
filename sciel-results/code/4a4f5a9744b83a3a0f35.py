import numpy as np
from scipy.optimize import minimize
import time

N = 400
_DEADLINE = 150.0
_START_TIME = None


def _fft_grad_conv(sw, h):
    """Return g[i] = sum_k sw[k] * (h[k-i] if 0<=k-i<N + h[i-k] if 0<=i-k<N).

    This is the circular convolution of sw (length 2N-1, zero-padded to 2N-1)
    with h (zero-padded to 2N-1), which equals the linear convolution because
    the result only has 2N-1 non-zero terms. Computed via FFT: O(N log N).
    """
    n = 2 * N - 1
    Fh = np.fft.rfft(h, n=2 * n)
    Fsw = np.fft.rfft(sw, n=2 * n)
    circ = np.fft.irfft(Fh * Fsw, n=2 * n)[:2 * n]
    # linear_conv[i] = circ[i] + circ[i + n]  for i in [0, 2N-2]
    lin = circ[:2 * n - 1] + circ[n:2 * n]
    return lin[:N]


def _soft_max_obj(x, temp=0.05):
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    smax = m + temp * np.log(np.sum(np.exp((conv - m) / temp)))
    return smax / (s * s)


def _soft_max_grad(x, temp=0.05):
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(x)
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    exp_terms = np.exp((conv - m) / temp)
    sw = exp_terms / np.sum(exp_terms)

    grad = _fft_grad_conv(sw, h)

    smax = m + temp * np.log(np.sum(exp_terms))
    d_obj = grad / (s * s) - 2.0 * smax / (s * s * s)
    return d_obj


def _objective_hard(x):
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    return abs(2.0 * N * m / (s * s))


def _topk_smooth_max_obj(x, k=8, temp=0.02):
    """Smooth max of the top-k terms of conv, divided by sum^2.

    Uses a softmax over the top-k terms (with the rest replaced by the k-th
    largest) to get a differentiable approximation that is much closer to the
    true max than a full softmax over 2N-1 terms.
    """
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    # Replace all but the top-k with the k-th largest value
    kth = np.partition(conv, -k)[-k]
    mask = conv >= kth
    topk = np.where(mask, conv, kth)
    m = np.max(topk)
    smax = m + temp * np.log(np.sum(np.exp((topk - m) / temp)))
    return smax / (s * s)


def _topk_smooth_max_grad(x, k=8, temp=0.02):
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.zeros_like(x)
    conv = np.convolve(h, h, mode='full')
    kth = np.partition(conv, -k)[-k]
    mask = conv >= kth
    topk = np.where(mask, conv, kth)
    m = np.max(topk)
    exp_terms = np.exp((topk - m) / temp)
    sw = exp_terms / np.sum(exp_terms)
    # sw is zero where mask is False (since topk == kth there, but we still
    # include them in the softmax; that's fine, just dilutes slightly)
    grad = _fft_grad_conv(sw, h)
    smax = m + temp * np.log(np.sum(exp_terms))
    d_obj = grad / (s * s) - 2.0 * smax / (s * s * s)
    return d_obj


def _pin_sum(h):
    """Subtract the mean so sum(h) == 0, then add a tiny constant to avoid s=0.

    Actually, for the polish we want sum(h) fixed at its current value.
    We'll handle that in the objective wrapper instead.
    """
    return h - np.mean(h)


def _polish_with_fixed_sum(h0, deadline):
    """Polish by optimizing max(conv) with sum(h) fixed at sum(h0).

    We parameterize as h = h0 + (x - x0) where x0 = h0, so sum(h) = sum(h0)
    automatically. But L-BFGS-B doesn't support constraints easily, so we just
    subtract the mean deviation: h = h0 + (x - mean(x) * N) ... no, simpler:
    h = h0 + (x - np.mean(x) * np.ones(N)) doesn't work either.

    Simplest: h = h0 + (x - x0) where x0 = h0, so h = x. But then sum changes.

    Alternative: use the fact that the objective is scale-invariant in a sense:
    value = 2N * max(conv) / sum^2. If we scale h by c, max(conv) scales by c^2
    and sum^2 scales by c^2, so value is unchanged. So we can fix sum(h) = 1
    without loss of generality!

    So: minimize max(conv) subject to sum(h) = 1. We'll use a penalty or just
    project: after each evaluation, rescale h so sum(h) = 1.
    """
    s0 = np.sum(h0)
    if abs(s0) < 1e-12:
        return h0, _objective_hard(h0)

    def obj_scaled(x):
        # Rescale so sum = 1
        s = np.sum(x)
        if abs(s) < 1e-12:
            return 1e10
        h = x / s
        return _topk_smooth_max_obj(h, k=8, temp=0.02)

    def grad_scaled(x):
        s = np.sum(x)
        if abs(s) < 1e-12:
            return np.zeros_like(x)
        h = x / s
        gh = _topk_smooth_max_grad(h, k=8, temp=0.02)
        # Chain rule: d(x/s)/dx = 1/s * I - x/s^2 * 1^T
        # So d(obj)/dx = (1/s) * gh - (x/s^2) * dot(gh, 1)
        return (1.0 / s) * gh - (x / (s * s)) * np.dot(gh, np.ones(N))

    try:
        res = minimize(
            obj_scaled,
            h0.copy(),
            jac=grad_scaled,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000}
        )
        if time.time() > deadline:
            return h0, _objective_hard(h0)
        # Rescale back to original sum
        s_res = np.sum(res.x)
        if abs(s_res) < 1e-12:
            return h0, _objective_hard(h0)
        h_final = res.x * (s0 / s_res)
        return h_final, _objective_hard(h_final)
    except Exception:
        return h0, _objective_hard(h0)


def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    best_val = 2.0
    best_h = np.ones(N)

    deadline = _START_TIME + _DEADLINE - 10.0  # 10s safety margin

    # Generate diverse initializations
    inits = []

    for i in range(16):
        r = np.random.RandomState(1000 + i * 137)
        inits.append(r.randn(N) * (0.3 + 0.1 * i))

    inits.append(np.linspace(-1, 1, N))
    inits.append(np.linspace(1, -1, N))

    for frac in [0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75]:
        h = np.zeros(N)
        h[:int(frac * N)] = 1.0
        h[int(frac * N):] = -0.3
        inits.append(h - np.mean(h))

    t = np.linspace(0, 2 * np.pi, N)
    for freq in [1, 2, 3, 5, 8]:
        inits.append(np.cos(freq * t) - np.mean(np.cos(freq * t)))
        inits.append(np.sin(freq * t) - np.mean(np.sin(freq * t)))

    x = np.linspace(-1, 1, N)
    inits.append(x**2 - np.mean(x**2))
    inits.append(x**3 - np.mean(x**3))
    inits.append(x**4 - np.mean(x**4))

    for period in [7, 11, 13, 17]:
        h = np.zeros(N)
        h[::period] = 1.0
        h[1::period] = -0.5
        inits.append(h - np.mean(h))

    # Multi-temperature annealing for each initialization
    for init_idx, init_h in enumerate(inits):
        if time.time() > deadline - 5:
            break

        h = init_h.astype(float).copy()
        s = np.sum(h)
        if abs(s) < 1e-12:
            h += 0.01 * np.random.RandomState(init_idx).randn(N)
            s = np.sum(h)
            if abs(s) < 1e-12:
                continue

        temps = [0.1, 0.05, 0.02, 0.01, 0.005]
        current_h = h.copy()

        for temp in temps:
            if time.time() > deadline - 5:
                break

            try:
                res = minimize(
                    lambda x, t=temp: _soft_max_obj(x, t),
                    current_h,
                    jac=lambda x, t=temp: _soft_max_grad(x, t),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7, 'maxfun': 800}
                )

                new_val = _objective_hard(res.x)
                if new_val < best_val:
                    best_val = new_val
                    best_h = res.x.copy()

                if res.fun < _soft_max_obj(current_h, temp):
                    current_h = res.x.copy()
            except Exception:
                pass

    # Final high-precision polish: first low-temp soft-max, then top-k max polish
    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _soft_max_obj(x, 0.001),
                best_h,
                jac=lambda x: _soft_max_grad(x, 0.001),
                method='L-BFGS-B',
                options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 4000}
            )
            val = _objective_hard(res.x)
            if val < best_val:
                best_val = val
                best_h = res.x.copy()
        except Exception:
            pass

    # Top-k smooth max polish (targets the true max term more directly)
    if time.time() < deadline - 1:
        try:
            h_polished, val_polished = _polish_with_fixed_sum(best_h, deadline)
            if val_polished < best_val:
                best_val = val_polished
                best_h = h_polished
        except Exception:
            pass

    return best_h
