"""Rewritten: global continuous optimization (SLSQP/L-BFGS-B) + local refinement."""
import time
import numpy as np

N = 100
DEADLINE = 140.0  # safety margin under 150s


def _value(h):
    h = np.asarray(h)
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(c)) / N * 2.0


def _smooth_max(c, s=20.0):
    m = c.max()
    return m + np.log(np.sum(np.exp(s * (c - m)))) / s


def _objective(h, s=20.0):
    h = np.clip(h, 0.0, 1.0)
    c = np.correlate(h, 1.0 - h, mode="full")
    return float(_smooth_max(c, s))


def _local_refine(h, time_limit=18.0):
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0).copy()
    target = 50.0
    # fix sum
    diff = target - h.sum()
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    h += (target - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)
    best = h.copy()
    bestv = _value(best)
    t0 = time.time()
    step = 0.2
    while time.time() - t0 < time_limit and step > 1e-4:
        improved = False
        for _ in range(3000):
            if time.time() - t0 > time_limit:
                break
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
            # sum-preserving swap: move delta from i to j
            delta = np.random.uniform(-step, step)
            ni = best[i] - delta
            nj = best[j] + delta
            if ni < 0 or ni > 1 or nj < 0 or nj > 1:
                continue
            trial = best.copy()
            trial[i] = ni
            trial[j] = nj
            v = _value(trial)
            if v < bestv - 1e-12:
                best = trial
                bestv = v
                improved = True
        if not improved:
            step *= 0.7
    return best, bestv


def _init_kinds():
    inits = []
    # 1) constant
    inits.append(np.full(N, 0.5))
    # 2) alternating 0/1 blocks (balanced)
    a = np.zeros(N)
    a[0::2] = 1.0
    inits.append(a)
    # 3) gradient
    inits.append(np.linspace(0.0, 1.0, N))
    # 4) sawtooth
    x = np.arange(N)
    inits.append((x % 10) / 9.0 * 2.0 - 1.0 + 0.5)
    inits[-1] = np.clip(inits[-1], 0, 1)
    # 5) random
    rng = np.random.default_rng(12345)
    r = rng.uniform(0, 1, N)
    r += (50.0 - r.sum()) / N
    inits.append(np.clip(r, 0, 1))
    # 6) two-tone block
    b = np.zeros(N)
    b[:50] = 1.0
    inits.append(b)
    # 7) sinusoid
    inits.append(0.5 + 0.4 * np.sin(2 * np.pi * np.arange(N) / N * 3))
    inits[-1] = np.clip(inits[-1], 0, 1)
    return inits


def construct():
    t_start = time.time()
    best = None
    bestv = float("inf")

    def consider(h):
        nonlocal best, bestv
        h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
        h += (50.0 - h.sum()) / N
        h = np.clip(h, 0.0, 1.0)
        v = _value(h)
        if v < bestv:
            bestv = v
            best = h.copy()
        return v

    inits = _init_kinds()
    from scipy.optimize import minimize

    for k, h0 in enumerate(inits):
        if time.time() - t_start > DEADLINE - 40.0:
            break
        h0 = np.clip(np.asarray(h0, dtype=float), 0.0, 1.0)
        h0 += (50.0 - h0.sum()) / N
        h0 = np.clip(h0, 0.0, 1.0)
        consider(h0)
        # global continuous optimization
        for s in (12.0, 25.0):
            if time.time() - t_start > DEADLINE - 40.0:
                break
            try:
                res = minimize(
                    _objective, h0.copy(), args=(s,),
                    method="SLSQP",
                    bounds=[(0.0, 1.0)] * N,
                    constraints=[{"type": "eq", "fun": lambda x: x.sum() - 50.0}],
                    options={"maxiter": 300, "ftol": 1e-10},
                )
                consider(res.x)
            except Exception:
                pass
        # refine
        if time.time() - t_start < DEADLINE - 25.0:
            h2, v2 = _local_refine(best if best is not None else h0,
                                   time_limit=min(18.0, max(1.0, DEADLINE - 25.0 - (time.time() - t_start))))
            consider(h2)

    # final global local search on best
    if best is not None and time.time() - t_start < DEADLINE - 5.0:
        h2, v2 = _local_refine(best, time_limit=max(1.0, DEADLINE - 5.0 - (time.time() - t_start)))
        consider(h2)

    if best is None:
        best = np.full(N, 0.5)
    return best
