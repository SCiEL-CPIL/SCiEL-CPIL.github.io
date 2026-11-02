"""Suite task: signed equal-width step functions minimizing max|autoconv|/integral^2."""
import time
import numpy as np

N = 400


def _score(x):
    """Raw objective: max|autoconv| / (sum)^2  (lower is better)."""
    ac = np.correlate(x, x, mode='full')
    S = x.sum()
    if S == 0:
        return float('inf')
    return float(np.max(np.abs(ac)) / (S * S))


def construct():
    t0 = time.time()
    DEADLINE = 140.0          # safety margin before the 150s wall-clock kill
    rng = np.random.default_rng(12345)

    # ---- Base flat-top (Hanning) window: zero-sidelobe magnitude spectrum ----
    base = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N) / N))
    base = base / base.max()
    W = base.sum()            # integral of the window

    # ---- Initial sign pattern: maximize |sum| (all +1) ----
    s = np.ones(N)

    def autoconv_from_s(sgn):
        return np.correlate(sgn * base, sgn * base, mode='full')

    # ---- Smooth surrogate: log-sum-exp of |ac| (soft max) ----
    def softmax_ac(ac, beta):
        a = np.abs(ac)
        m = a.max()
        if m == 0:
            return 0.0
        return float(m + np.log(np.sum(np.exp(beta * (a - m)))))

    def obj_s(sgn, beta):
        ac = autoconv_from_s(sgn)
        S = (sgn * base).sum()
        if abs(S) < 1e-12:
            return float('inf')
        return softmax_ac(ac, beta) / (S * S)

    # ---- Simulated annealing over sign flips ----
    cur = s.copy()
    cur_o = obj_s(cur, 100.0)
    best = cur.copy()
    best_o = cur_o

    T0, T1 = 1.0, 1e-4
    iters = 60000
    for it in range(iters):
        if time.time() > DEADLINE:
            break
        T = T0 * (T1 / T0) ** (it / iters)
        i = rng.integers(0, N)
        old = cur[i]
        cur[i] = -old
        o = obj_s(cur, 100.0)
        if o < cur_o or rng.random() < np.exp((cur_o - o) / max(T, 1e-9)):
            cur_o = o
            if o < best_o:
                best_o = o
                best = cur.copy()
        else:
            cur[i] = old

    # ---- Final local sign-flip polish (exact objective, accept improving flips) ----
    s = best.copy()
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        for i in range(N):
            if time.time() > DEADLINE:
                break
            old = s[i]
            s[i] = -old
            o = _score(s * base)
            if o < best_o:
                best_o = o
                best = s.copy()
                improved = True
            else:
                s[i] = old
        s = best.copy()

    # ---- Return the signed step heights ----
    return best * base


if __name__ == "__main__":
    x = construct()
    print("score:", _score(x))
    print("seed value (all-ones):", 1.0 / N)
