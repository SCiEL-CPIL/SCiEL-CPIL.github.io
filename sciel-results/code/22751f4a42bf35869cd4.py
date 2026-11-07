"""Erdos minimum overlap: minimize max_k (h,1-h)_k via smooth gradient search."""
import numpy as np

N = 100
S = N / 2.0  # 50
TOL = 1e-9 * N


def _elapsed():
    import time
    return time.time()


def _raw_value(h):
    h = np.asarray(h, dtype=float)
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c) / N * 2.0)


def _is_valid(h):
    h = np.asarray(h, dtype=float)
    if h.shape != (N,):
        return False
    if not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-9) or np.any(h > 1.0 + 1e-9):
        return False
    if abs(np.sum(h) - S) > TOL:
        return False
    return True


def _project(h):
    """Project onto {x in [0,1]^N : sum(x)=S} via bisection on threshold."""
    h = np.clip(h, 0.0, 1.0)
    if abs(np.sum(h) - S) <= 1e-12:
        return h
    # find threshold t so that sum(clip(h - t, 0, 1)) == S
    lo, hi = -5.0, 5.0
    for _ in range(80):
        t = 0.5 * (lo + hi)
        s = np.sum(np.clip(h - t, 0.0, 1.0))
        if s > S:
            lo = t
        else:
            hi = t
    t = 0.5 * (lo + hi)
    return np.clip(h - t, 0.0, 1.0)


def _correlation_matrix(h):
    """C[k] = sum_i h[i] h[i-k] for k in -(N-1)..(N-1)."""
    H = np.fft.rfft(h, n=2 * N)
    C = np.fft.irfft(H * np.conj(H), n=2 * N)[:N]
    return C


def _objective_and_grad(h, beta):
    C = _correlation_matrix(h)          # length N, C[k]
    M = float(np.max(C))
    # log-sum-exp smooth max
    m = M
    LSE = m + np.log(np.sum(np.exp(beta * (C - m))))
    # d LSE / d h_i = beta * sum_k w_k * dC_k/dh_i
    w = beta * np.exp(beta * (C - m))
    w = w / w.sum()
    # C[k] = sum_i h[i] h[i-k]; dC_k/dh_i = h[i-k] + h[i+k] (with 0 outside)
    # g_i = sum_k w_k (h[i-k] + h[i+k])
    # = sum_k w_k h[i-k] + sum_k w_k h[i+k] = (w * h)[i] + (w * h)[i] reversed
    wh = np.correlate(w, h, mode="full")  # length 2N-1
    g = wh[:N] + wh[2 * N - 1 - np.arange(N)]
    return LSE, g


def construct():
    import time
    t0 = time.time()
    DEADLINE = 110.0  # safety margin under 120s

    rng = np.random.default_rng(12345)

    # --- candidate initializers ---
    inits = [np.full(N, 0.5)]
    # random
    for _ in range(6):
        inits.append(rng.uniform(0.0, 1.0, size=N))
    # periodic 0/1 patterns (low autocorrelation)
    for period in (2, 3, 4, 5, 7):
        pat = np.zeros(N)
        for i in range(N):
            if i % period == 0:
                pat[i] = 1.0
        inits.append(pat)
    # alternating
    alt = np.zeros(N); alt[::2] = 1.0; inits.append(alt)
    alt2 = np.zeros(N); alt2[1::2] = 1.0; inits.append(alt2)
    # random low-density
    for _ in range(4):
        r = rng.uniform(0.0, 1.0, size=N)
        inits.append(np.where(r < 0.5, 1.0, 0.0))

    def _improve(h0, budget):
        h = _project(h0.copy())
        best_h = h.copy()
        best_val = _raw_value(h)
        beta = 20.0
        lr = 0.05
        it = 0
        while it < 20000:
            if time.time() - t0 > budget:
                break
            # batch: do a few steps
            for _ in range(50):
                LSE, g = _objective_and_grad(h, beta)
                h = h - lr * g
                h = _project(h)
                # occasional value check
                v = _raw_value(h)
                if v < best_val:
                    best_val = v
                    best_h = h.copy()
                if time.time() - t0 > budget:
                    break
            it += 50
            # anneal
            beta *= 1.05
            lr *= 0.995
            if lr < 1e-6:
                lr = 1e-6
        # final polish: small steps high beta
        beta = 200.0
        for _ in range(2000):
            if time.time() - t0 > budget:
                break
            LSE, g = _objective_and_grad(h, beta)
            h = h - 1e-4 * g
            h = _project(h)
            v = _raw_value(h)
            if v < best_val:
                best_val = v
                best_h = h.copy()
        return best_h, best_val

    # time budget per init
    n_init = len(inits)
    per_init = (DEADLINE - (time.time() - t0)) / n_init
    per_init = max(3.0, min(per_init, 12.0))

    best_h = None
    best_val = float("inf")
    for i, h0 in enumerate(inits):
        if time.time() - t0 > DEADLINE:
            break
        h, v = _improve(h0, DEADLINE - 3.0)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    if best_h is None:
        best_h = np.full(N, 0.5)

    # ensure validity
    if not _is_valid(best_h):
        best_h = _project(best_h)
    # exact sum fix
    diff = S - np.sum(best_h)
    best_h = best_h + diff / N
    best_h = np.clip(best_h, 0.0, 1.0)
    # re-project to keep sum exact
    best_h = _project(best_h)
    # final tiny correction to hit sum within tol
    for _ in range(50):
        d = S - np.sum(best_h)
        if abs(d) <= 1e-12:
            break
        best_h = best_h + d / N
        best_h = np.clip(best_h, 0.0, 1.0)

    if not _is_valid(best_h):
        # fallback
        best_h = np.full(N, 0.5)

    return best_h
