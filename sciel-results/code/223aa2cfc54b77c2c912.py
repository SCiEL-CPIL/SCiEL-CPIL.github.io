import numpy as np
import time

N = 400
L = 800  # full convolve length


def _full_conv(h):
    f = np.fft.rfft(h, n=2 * N)
    return np.fft.irfft(f * f, n=2 * N)[:L]


def _value(h):
    S = float(np.sum(h))
    if S == 0.0 or not np.isfinite(S):
        return float("inf")
    return abs(2.0 * N * float(np.max(_full_conv(h))) / (S * S))


def _soft_max_obj(h, beta):
    S = float(np.sum(h))
    if S == 0.0:
        return 1e18
    conv = _full_conv(h)
    m = float(np.max(conv))
    s = float(np.sum(np.exp(beta * (conv - m))))
    sm = m + np.log(s) / beta
    return sm / (S * S)


def _optimize_beta0(h0, beta, iters, deadline, base):
    h = h0.copy()
    S = float(np.sum(h))
    if S == 0.0:
        return h0, base
    invS2 = 1.0 / (S * S)
    for it in range(iters):
        if time.time() > deadline:
            break
        conv = _full_conv(h)
        e = np.exp(beta * (conv - float(np.max(conv))))
        num = 2.0 * N * invS2 * e  # d(conv/S^2)/dh
        den = S * S  # = sum h
        g = num - 2.0 * den * h
        # coordinate descent
        for _ in range(3):
            for i in range(N):
                if g[i] > 0.0:
                    h[i] = 0.0
                    g[i] -= num[i]
                    den -= h[i]
                    g += 2.0 * den * h
                    invS2 = 1.0 / den
        if time.time() > deadline:
            break
    return h, _value(h)


def _refine(h0, deadline):
    h = h0.copy()
    S = float(np.sum(h))
    if S == 0.0:
        return h0
    best = _value(h)
    for beta in (1.0, 3.0, 8.0):
        iters = 20000
        for it in range(iters):
            if time.time() > deadline:
                break
            conv = _full_conv(h)
            m = float(np.max(conv))
            e = np.exp(beta * (conv - m))
            s = float(np.sum(e))
            sm = m + np.log(s) / beta
            dconv = 2.0 * e
            dsm = (dconv * s + sm * dconv.sum() / s) / beta
            num = dsm / (S * S)
            den = S * S
            g = num - 2.0 * den * h
            for _ in range(3):
                for i in range(N):
                    if g[i] > 0.0:
                        h[i] = 0.0
                        g[i] -= num[i]
                        den -= h[i]
                        g += 2.0 * den * h
        v = _value(h)
        if v < best:
            best = v
    return h


def construct():
    t0 = time.time()
    deadline = t0 + 130.0

    best_h = np.ones(N)
    best_v = _value(best_h)  # 2.0

    # Family A: negative core + single positive spike
    # h = [0.., spike, 0.., core, 0..]
    for w in range(200, 399, 15):
        if time.time() > deadline:
            break
        core = -np.ones(w)
        for pos in range(0, N - w, 25):
            if time.time() > deadline:
                break
            for sp in (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0):
                if time.time() > deadline:
                    break
                h = np.zeros(N)
                h[pos:pos + w] = core
                h[pos - 1] = sp  # spike just before core
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()

    # Family B: negative core + spike just after core (lag +1)
    for w in range(200, 399, 25):
        if time.time() > deadline:
            break
        for pos in range(0, N - w, 40):
            if time.time() > deadline:
                break
            for sp in (0.5, 1.0, 2.0, 3.0):
                if time.time() > deadline:
                    break
                h = np.zeros(N)
                h[pos:pos + w] = -1.0
                h[pos + w] = sp
                v = _value(h)
                if v < best_v:
                    best_v = v
                    best_h = h.copy()

    # Random restarts in the sparse-spike family
    rng = np.random.default_rng(7)
    for _ in range(300):
        if time.time() > deadline:
            break
        w = rng.integers(150, 399)
        pos = rng.integers(0, N - w)
        h = np.zeros(N)
        h[pos:pos + w] = -rng.uniform(0.5, 2.0)
        k = rng.integers(0, N)
        h[k] = rng.uniform(0.2, 5.0)
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()

    # Soft-max coordinate refinement on the best candidate
    if time.time() < deadline - 15.0:
        refined = _refine(best_h, deadline - 10.0)
        v = _value(refined)
        if v < best_v:
            best_v = v
            best_h = refined

    # Safety: ensure finite, non-zero sum
    if not np.all(np.isfinite(best_h)) or np.sum(best_h) == 0.0:
        return np.ones(N)
    return best_h
