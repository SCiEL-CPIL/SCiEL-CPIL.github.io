"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def _ratio(h):
    """Compute the objective value for a nonnegative height vector h.
    Ratio = ||h||_2^2 / (||h||_1 * ||h*h||_inf) using discrete autoconvolution.
    """
    h = np.asarray(h, dtype=float)
    if h.size == 0:
        return 0.0
    l1 = float(np.sum(h))
    if l1 <= 0.0:
        return 0.0
    l2sq = float(np.sum(h * h))
    if l2sq <= 0.0:
        return 0.0
    # discrete autoconvolution (full), sup norm
    conv = np.convolve(h, h, mode='full')
    sup = float(np.max(conv))
    if sup <= 0.0:
        return 0.0
    return l2sq / (l1 * sup)

def _normalize_one(h):
    s = float(np.sum(h))
    if s <= 0.0:
        return np.ones(N) / N
    return h / s

def construct():
    deadline = time.time() + 110.0   # safety margin before 120s kill
    rng = np.random.default_rng(12345)

    # Candidate pool of shapes
    candidates = []

    # 1) uniform
    candidates.append(np.ones(N))
    # 2) triangular
    x = np.linspace(-1, 1, N)
    candidates.append(np.maximum(0.0, 1.0 - np.abs(x)))
    # 3) gaussian bumps at various centers / widths
    for center in [0.0, -0.3, 0.3, -0.6, 0.6]:
        for width in [0.15, 0.3, 0.5, 0.8, 1.2]:
            g = np.exp(-((x - center) ** 2) / (2.0 * width ** 2))
            candidates.append(g)
    # 4) single peak (delta-like)
    for k in [0, N // 2, N - 1, N // 4, 3 * N // 4]:
        d = np.zeros(N)
        d[k] = 1.0
        candidates.append(d)
        # small plateau
        p = np.zeros(N)
        p[max(0, k - 1):min(N, k + 2)] = 1.0
        candidates.append(p)
    # 5) random smooth shapes
    for _ in range(40):
        r = rng.random(N)
        # smooth it
        k = 3
        ker = np.ones(k) / k
        r = np.convolve(r, ker, mode='same')
        candidates.append(np.maximum(r, 0.0))

    best = None
    best_val = -1.0
    for c in candidates:
        cn = _normalize_one(c)
        v = _ratio(cn)
        if v > best_val:
            best_val = v
            best = cn.copy()

    # Local polish via projected gradient ascent on log-soft objective
    # Maximize f = log(l2sq) - log(l1) - log(sup).  Work on normalized simplex.
    cur = best.copy()
    cur = _normalize_one(cur)
    cur_best = _ratio(cur)

    def neg_log_ratio(h):
        h = np.maximum(h, 0.0)
        s = float(np.sum(h))
        if s <= 0:
            return 1e18
        hn = h / s
        l2sq = float(np.sum(hn * hn))
        conv = np.convolve(hn, hn, mode='full')
        sup = float(np.max(conv))
        if l2sq <= 0 or sup <= 0:
            return 1e18
        return -(np.log(l2sq) - np.log(s) - np.log(sup))

    step = 0.05
    it = 0
    while time.time() < deadline - 5.0:
        h = np.maximum(cur, 0.0)
        s = float(np.sum(h))
        if s <= 0:
            h = np.ones(N) / N
            s = 1.0
        hn = h / s
        l2sq = float(np.sum(hn * hn))
        conv = np.convolve(hn, hn, mode='full')
        sup = float(np.max(conv))
        if l2sq <= 0 or sup <= 0:
            break
        # gradient of -log ratio wrt h (unconstrained), then project
        # d/dh log l2sq = 2 h / l2sq ; d/dh log s = 1/s ; d/dh log sup (approx) via argmax
        idx = int(np.argmax(conv))
        # derivative of conv at idx wrt h: for full conv, conv[idx] = sum_i h[i] h[idx-i] (clamped)
        # d conv[idx]/d h[j] = h[idx-j] + h[j-idx] (where indices valid)
        grad_sup = np.zeros(N)
        for j in range(N):
            a = idx - j
            b = j - idx
            term = 0.0
            if 0 <= a < N:
                term += hn[a]
            if 0 <= b < N:
                term += hn[b]
            grad_sup[j] = term
        grad = 2.0 * hn / l2sq - (1.0 / s) * np.ones(N) - grad_sup / sup
        # project: move h (unnormalized) by -grad, keep nonnegative, renormalize
        cand = np.maximum(h - step * grad, 0.0)
        cs = float(np.sum(cand))
        if cs <= 0:
            cand = np.ones(N) / N
            cs = 1.0
        cand = cand / cs
        v = _ratio(cand)
        if v > cur_best:
            cur_best = v
            cur = cand.copy()
            step = min(step * 1.05, 0.5)
        else:
            step *= 0.7
            if step < 1e-5:
                break
        it += 1
        if it > 200000:
            break

    if cur_best > best_val:
        best_val = cur_best
        best = cur.copy()

    # Final high-precision polish: fine coordinate ascent
    if time.time() < deadline - 3.0:
        cur = best.copy()
        cur_best = _ratio(cur)
        for _ in range(200):
            if time.time() > deadline - 3.0:
                break
            improved = False
            for j in range(N):
                base = cur[j]
                for f in [0.5, 0.75, 1.25, 1.5, 2.0, 0.25, 3.0]:
                    trial = cur.copy()
                    trial[j] = base * f
                    v = _ratio(trial)
                    if v > cur_best:
                        cur_best = v
                        cur = trial
                        base = trial[j]
                        improved = True
            if not improved:
                break

    return [float(x) for x in _normalize_one(np.maximum(cur, 0.0))]
# EVOLVE-BLOCK-END
