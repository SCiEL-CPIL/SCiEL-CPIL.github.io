"""Second autocorrelation inequality: maximize ||h||^2_2 / (||h||_1 * ||h*h||_inf).

Discrete formulation on 50 equal intervals of [-1/4, 1/4], spacing dh = 1/20.
The continuous autoconvolution sampled at the grid is approximated by the
discrete convolution  (h*h)(x_i) = dh * sum_j h_j * h_k  over j+k = i + OFFSET,
where OFFSET = 1/2/dh = 10  (because x_j + x_k = x_i  <=>  j + k = i + 10).

The objective is scale-invariant, so we fix ||h||_2 = 1 and minimize
||h*h||_inf  (equivalently maximize  1 / (||h||_1 * ||h*h||_inf)).
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH

# Wall-clock deadline with safety margin (evaluator kills at 150 s).
_DEADLINE = 150.0
_SAFETY = 15.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _autoconv(h):
    """Discrete autoconvolution sampled on the same grid (length N)."""
    h = np.asarray(h, dtype=float)
    full = np.convolve(h, h, mode='full')  # length 2N-1, index j+k
    # (h*h)(x_i) uses j+k = i + OFFSET  ->  full[i + OFFSET]
    out = full[OFFSET:OFFSET + N]
    return DH * out


def _ratio(h):
    """Return the objective value R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)


def _ratio_neg(h):
    return -_ratio(h)


def _ratio_and_grad(h):
    """Value and (numerical) gradient of the ratio for L-BFGS-B."""
    h = np.asarray(h, dtype=float)
    val = _ratio(h)
    # numerical gradient
    g = np.zeros_like(h)
    eps = 1e-7
    for i in range(N):
        hp = h.copy(); hp[i] += eps
        hm = h.copy(); hm[i] = max(h[i] - eps, 0.0)
        g[i] = (_ratio(hp) - _ratio(hm)) / (hp[i] - hm[i] if hp[i] != hm[i] else eps)
    return val, g


def _logR_and_grad_analytic(h):
    """
    Analytical value and gradient of log(R) where
    R = l2sq / (l1 * acmax).
    log R = log(l2sq) - log(l1) - log(acmax)
    Using active-set approximation: acmax = ac[argmax(ac)]
    """
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 1e12, np.zeros_like(h)

    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 1e12, np.zeros_like(h)

    # d(log l2sq)/d(h_i) = 2*h_i / l2sq
    # d(log l1)/d(h_i) = 1 / l1
    # d(log acmax)/d(h_i) = (1/acmax) * d(ac[argmax])/d(h_i)
    # ac[m] = DH * sum_j h_j * h_{m + OFFSET - j}  (for valid indices)
    # d(ac[m])/d(h_i) = DH * (h_{m + OFFSET - i} + h_{m + OFFSET - i}) ... wait
    # ac[m] = DH * sum_j h[j] * h[m + OFFSET - j]
    # d/d(h[i]) = DH * (h[m + OFFSET - i] + h[m + OFFSET - i]) if i != m+OFFSET-i
    #            = 2*DH*h[i] if i == m+OFFSET-i
    # More carefully: ac[m] = DH * sum_j h[j] * h[m+OFFSET-j]
    # This is symmetric in j and (m+OFFSET-j).
    # d(ac[m])/d(h[i]) = DH * [h[m+OFFSET-i] + h[m+OFFSET-i]] = 2*DH*h[m+OFFSET-i]
    # But we must ensure indices are in [0, N-1].
    # Actually the sum is over j such that both j and (m+OFFSET-j) are in [0, N-1].
    # The derivative is DH * (h[m+OFFSET-i] if (m+OFFSET-i) in range else 0)
    #               + DH * (h[i] if i in range else 0) ... no.
    # Let's think again. ac[m] = DH * sum_j h[j] * h[k_j] where k_j = m + OFFSET - j.
    # d(ac[m])/d(h[i]) = DH * sum_j [delta_{j,i} * h[k_j] + h[j] * delta_{k_j,i}]
    #                  = DH * h[k_i] + DH * sum_j h[j] * delta_{m+OFFSET-j, i}
    #                  = DH * h[m+OFFSET-i] + DH * h[m+OFFSET-i]   (if both in range)
    #                  = 2*DH*h[m+OFFSET-i]  (when m+OFFSET-i is in [0,N-1])
    # Wait, the second term: delta_{m+OFFSET-j, i} means j = m+OFFSET-i.
    # So it's DH * h[m+OFFSET-i] (from first term) + DH * h[m+OFFSET-i] (from second term, where j=m+OFFSET-i)
    # = 2*DH*h[m+OFFSET-i], provided 0 <= m+OFFSET-i < N.
    # If m+OFFSET-i is out of range, the derivative is 0.

    m = int(np.argmax(ac))
    # Gradient contribution from -log(acmax)
    # d(-log acmax)/d(h_i) = -1/acmax * d(ac[m])/d(h_i)
    # d(ac[m])/d(h_i) = 2*DH*h[m+OFFSET-i] if 0 <= m+OFFSET-i < N else 0

    g = np.zeros(N)
    # log(l2sq) part: 2*h_i / l2sq
    g += 2.0 * h / l2sq
    # -log(l1) part: -1/l1
    g -= 1.0 / l1
    # -log(acmax) part: -1/acmax * 2*DH*h[m+OFFSET-i]
    idx = m + OFFSET - np.arange(N)
    valid = (idx >= 0) & (idx < N)
    if np.any(valid):
        contrib = np.zeros(N)
        contrib[valid] = 2.0 * DH * h[idx[valid]]
        g -= contrib / acmax

    logR = math.log(l2sq) - math.log(l1) - math.log(acmax)
    return logR, g


def _normalize(h):
    s = float(np.linalg.norm(h))
    if s <= 0:
        return h
    return h / s


def _local_optimize(h0, time_budget):
    """Run L-BFGS-B on the ratio with nonnegativity; return best h found."""
    h0 = np.maximum(np.asarray(h0, dtype=float), 0.0)
    h0 = _normalize(h0)
    best_h = h0.copy()
    best_v = _ratio(h0)
    t0 = time.time()

    # Use a smooth surrogate: maximize log R = log l2sq - log l1 - log acmax.
    # We optimize -log R via analytical gradient with L-BFGS-B, nonnegativity.
    def neg_logR(x):
        x = np.maximum(x, 0.0)
        v = _ratio(x)
        if v <= 0.0:
            return 1e12
        return -math.log(v)

    def neg_logR_grad(x):
        x = np.maximum(x, 0.0)
        logR, g = _logR_and_grad_analytic(x)
        if logR > 1e11:
            return 1e12, np.zeros_like(x)
        return -logR, -g

    bounds = [(0.0, None)] * N
    try:
        res = minimize(
            neg_logR, h0,
            method='L-BFGS-B',
            jac=neg_logR_grad,
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 10000, 'ftol': 1e-14,
                     'gtol': 1e-10, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > time_budget else x),
        )
        # The callback trick: if it returns nan, scipy stops.
        if res.x is not None:
            rx = np.maximum(res.x, 0.0)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass

    # A couple of Nelder-Mead polish passes on the best so far.
    try:
        for _ in range(3):
            if _time_left() <= 5.0:
                break
            res2 = minimize(
                _ratio_neg, best_h,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True, 'disp': False},
            )
            rx = np.maximum(res2.x, 0.0)
            rv = _ratio(rx)
            if rv > best_v:
                best_v = rv
                best_h = rx.copy()
    except Exception:
        pass

    return best_h, best_v


def _initializations():
    """Yield a set of promising starting points."""
    inits = []
    # Single spike at center
    for pos in [0, N // 2, N - 1, N // 4, 3 * N // 4]:
        s = np.zeros(N); s[pos] = 1.0
        inits.append(s)
    # Two spikes (spread to reduce autoconv peak overlap)
    for a in range(0, N, 10):
        for b in range(a + 10, N, 10):
            s = np.zeros(N); s[a] = 1.0; s[b] = 1.0
            inits.append(s)
    # Three spikes
    for a in range(0, N, 15):
        for b in range(a + 15, N, 15):
            for c in range(b + 15, N, 15):
                s = np.zeros(N); s[a] = 1.0; s[b] = 1.0; s[c] = 1.0
                inits.append(s)
    # Random sparse profiles
    rng = np.random.default_rng(12345)
    for _ in range(40):
        k = rng.integers(1, 12)
        idx = rng.choice(N, size=k, replace=False)
        s = np.zeros(N)
        s[idx] = 1.0
        inits.append(s)
    # Random dense (soft) profiles
    for _ in range(40):
        s = rng.random(N) ** 3.0  # sparse-ish
        inits.append(s)
    # Gaussian-ish bumps
    for center in range(0, N, 7):
        x = np.arange(N)
        s = np.exp(-0.5 * ((x - center) / 3.0) ** 2)
        inits.append(s)
    return inits


def construct():
    global _start
    _start = time.time()

    best_h = None
    best_v = 0.0

    inits = _initializations()
    # Pre-evaluate and sort initializations by ratio (best first) to spend
    # optimization budget where it pays.
    scored = []
    for s in inits:
        v = _ratio(s)
        scored.append((v, s))
        if v > best_v:
            best_v = v
            best_h = s.copy()
    scored.sort(key=lambda t: -t[0])

    # Optimize the top initializations within the time budget.
    budget_per = 4.0
    for v0, s in scored:
        if _time_left() <= budget_per + 2.0:
            break
        # Only bother if this init is plausibly competitive.
        if v0 < best_v * 0.5 and best_v > 0:
            # still try a few, but skip very weak ones after the first pass
            if best_v > 20.0:
                continue
        h_opt, v_opt = _local_optimize(s, time_budget=budget_per)
        if v_opt > best_v:
            best_v = v_opt
            best_h = h_opt.copy()
        if _time_left() <= 20.0:
            break

    # Final high-precision polish on the global best if time remains.
    if best_h is not None and _time_left() > 15.0:
        try:
            h_pol, v_pol = _local_optimize(best_h, time_budget=_time_left() - 18.0)
            if v_pol > best_v:
                best_v = v_pol
                best_h = h_pol.copy()
        except Exception:
            pass

    if best_h is None:
        best_h = np.ones(N)
    best_h = np.maximum(best_h, 0.0)
    # Return as a plain sequence of floats (step heights).
    return [float(x) for x in best_h]


if __name__ == '__main__':
    # quick self-test / timing
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio(h))
    print("time =", time.time() - t0)
