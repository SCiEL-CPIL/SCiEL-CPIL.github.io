import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 50
DEADLINE = 145.0


def _autoconv_sup(h):
    """Sup norm of autoconvolution of the step function with heights h."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    pad = 2 * n - 1
    a = np.zeros(pad)
    a[:n] = h
    c = np.fft.ifft(np.fft.fft(a) ** 2).real
    return float(np.max(c)) * (1.0 / n)


def _ratio(h):
    h = np.maximum(h, 0.0)
    s1 = float(np.sum(h))
    if s1 < 1e-15:
        return 0.0
    s2 = float(np.sum(h * h))
    sup = _autoconv_sup(h)
    if sup < 1e-15:
        return 0.0
    return s2 / (s1 * sup)


def _multiplicative_descent(h0, deadline, eta0=0.15, patience=60):
    """Sinkhorn-style multiplicative updates: reduce mass at lags that drive
    the autoconvolution peak. Structural move that directly penalises the
    contributing coordinates of the peak lag."""
    h = np.maximum(h0, 0.0).copy()
    s = h.sum()
    if s < 1e-15:
        return h0, _ratio(h0)
    h = h / s
    best_h = h.copy()
    best_r = _ratio(h)
    stale = 0
    eta = eta0
    rng = np.random.default_rng()
    for it in range(8000):
        if time.time() > deadline:
            break
        conv = np.fft.ifft(np.fft.fft(np.r_[h, np.zeros(N - 1)]) ** 2).real[:2 * N - 1]
        sup_conv = conv.max()
        if sup_conv < 1e-15:
            break
        k_star = int(np.argmax(conv))
        grad = np.zeros(N)
        for i in range(N):
            v = 0.0
            j1 = k_star - i
            if 0 <= j1 < N:
                v += h[j1]
            j2 = i - k_star
            if 0 <= j2 < N:
                v += h[j2]
            grad[i] = v
        h = h * np.exp(-eta * grad)
        s = h.sum()
        if s < 1e-15:
            break
        h = h / s
        r = _ratio(h)
        if r > best_r:
            best_r = r
            best_h = h.copy()
            stale = 0
        else:
            stale += 1
            if stale > patience:
                eta *= 0.92
                if eta < 1e-4:
                    k = int(rng.integers(4, 14))
                    h = np.zeros(N)
                    h[rng.choice(N, size=k, replace=False)] = 1.0
                    h = h / h.sum()
                    eta = eta0
                    stale = 0
    return best_h, best_r


def _spectral_start(rng, n_bins=4):
    """Nonnegative starting point = |IFFT of a sparse random spectral mask|."""
    mask = np.zeros(N, dtype=complex)
    idx = rng.choice(N, size=n_bins, replace=False)
    mask[idx] = (rng.uniform(0.5, 2.0, n_bins) +
                 1j * rng.uniform(-1.0, 1.0, n_bins))
    mask[0] += float(np.sum(mask.real)) * 0.3
    h = np.abs(np.fft.ifft(mask))
    if h.sum() < 1e-15:
        h = np.ones(N)
    return h / h.sum()


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)

    # ---- Build diverse seeds ----
    seeds = []

    def sidon_set(max_size):
        s = [0]
        used = {0}
        for c in range(1, N):
            ok = all((i + c) not in used for i in s)
            if ok:
                s.append(c)
                used.add(c + c)
                for i in s[:-1]:
                    used.add(i + c)
            if len(s) >= max_size:
                break
        return s

    # Sidon-like spike patterns
    for sz in (6, 8, 10, 12, 14, 16):
        h = np.zeros(N)
        for i in sidon_set(sz):
            h[i] = 1.0
        seeds.append(h / h.sum())

    # Sparse random spike patterns
    for _ in range(30):
        k = int(rng.integers(4, 16))
        h = np.zeros(N)
        h[rng.choice(N, size=k, replace=False)] = rng.uniform(0.3, 1.5, k)
        seeds.append(h / h.sum())

    # Spectral-mask starts
    for nb in (2, 3, 4, 5, 6, 8, 10):
        for _ in range(4):
            seeds.append(_spectral_start(rng, n_bins=nb))

    # Classical shapes
    t = np.linspace(-1, 1, N)
    for base in (np.ones(N),
                 np.maximum(1 - t ** 2, 0.01),
                 np.maximum(t ** 2 + 0.1, 0.01),
                 np.exp(-t ** 2 / 0.3),
                 np.maximum(1 + 2 * np.exp(-((t - 0.6) ** 2) / 0.1) +
                            2 * np.exp(-((t + 0.6) ** 2) / 0.1), 0.05)):
        seeds.append(base / base.sum())
    alt = np.zeros(N); alt[::2] = 1.0
    seeds.append(alt / alt.sum())
    sp3 = np.zeros(N); sp3[::3] = 1.0
    seeds.append(sp3 / sp3.sum())
    sp4 = np.zeros(N); sp4[::4] = 1.0
    seeds.append(sp4 / sp4.sum())

    # Evaluate all seeds
    best_h = seeds[0].copy()
    best_score = _ratio(best_h)
    for s in seeds[1:]:
        sc = _ratio(s)
        if sc > best_score:
            best_score = sc
            best_h = s.copy()

    # ---- Phase 1: Multiplicative descent from top diverse seeds ----
    # Run MD from multiple diverse starting points
    seed_scores = [( _ratio(s), s) for s in seeds]
    seed_scores.sort(reverse=True)
    top_seeds = [s for _, s in seed_scores[:15]]

    for seed_h in top_seeds:
        if time.time() > DEADLINE - 30:
            break
        h_md, r_md = _multiplicative_descent(seed_h.copy(), DEADLINE - 25.0)
        if r_md > best_score:
            best_score = best_score  # placeholder
            best_score = r_md
            best_h = h_md.copy()

    # ---- Phase 2: Differential evolution (global search) ----
    def neg_ratio(x):
        x = np.maximum(x, 0.0)
        s = float(np.sum(x))
        if s < 1e-15:
            return 0.0
        return -_ratio(x / s)

    def elapsed_ok(margin):
        return (time.time() - t0) < DEADLINE - margin

    bounds = [(0.0, 1.0)] * N
    if elapsed_ok(25):
        try:
            res = differential_evolution(
                neg_ratio, bounds, seed=7, maxiter=50, popsize=30,
                tol=1e-9, init='sobol', polish=True,
                updating='immediate', mutation=(0.4, 1.0), recombination=0.7)
            h = np.maximum(res.x, 0.0)
            if h.sum() > 1e-15:
                h = h / h.sum()
                sc = _ratio(h)
                if sc > best_score:
                    best_score, best_h = sc, h.copy()
        except Exception:
            pass

    # ---- Phase 3: More multiplicative descent from current best and DE result ----
    if elapsed_ok(20):
        try:
            h_md, r_md = _multiplicative_descent(best_h.copy(), DEADLINE - 15.0)
            if r_md > best_score:
                best_score, best_h = r_md, h_md
        except Exception:
            pass

    # ---- Phase 4: Nelder-Mead local refinement ----
    if elapsed_ok(15):
        try:
            res3 = minimize(neg_ratio, best_h.copy(), method='Nelder-Mead',
                            options={'maxiter': 3000, 'xatol': 1e-10,
                                     'fatol': 1e-12, 'adaptive': True})
            h = np.maximum(res3.x, 0.0)
            if h.sum() > 1e-15:
                h = h / h.sum()
                sc = _ratio(h)
                if sc > best_score:
                    best_score, best_h = sc, h.copy()
        except Exception:
            pass

    # ---- Phase 5: Coordinate descent polish ----
    if elapsed_ok(12):
        try:
            cur = best_h.copy()
            improved = True
            while improved and elapsed_ok(12):
                improved = False
                cur_base = _ratio(cur)
                for i in range(N):
                    for step in (0.3, 0.5, 1.0, 2.0):
                        for sign in (1.0, -1.0):
                            trial = cur.copy()
                            trial[i] = max(0.0, cur[i] + sign * step * max(cur[i], 0.05))
                            sc = _ratio(trial)
                            if sc > cur_base + 1e-12:
                                cur = trial
                                cur_base = sc
                                improved = True
                                if sc > best_score:
                                    best_score, best_h = sc, cur.copy()
        except Exception:
            pass

    # ---- Phase 6: Final multiplicative descent pass ----
    if elapsed_ok(10):
        try:
            h_md2, r_md2 = _multiplicative_descent(best_h.copy(), DEADLINE - 3.0)
            if r_md2 > best_score:
                best_score, best_h = r_md2, h_md2
        except Exception:
            pass

    # Final safety: ensure nonnegative and valid
    best_h = np.maximum(best_h, 0.0)
    if best_h.sum() < 1e-15:
        best_h = np.ones(N) / N

    return best_h.tolist()
