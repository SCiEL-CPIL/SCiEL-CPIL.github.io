"""Combined program: epigraph convex formulation + smooth log-sum-exp + local refinement."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE = 140.0  # safety margin under 150s limit


def _value(h):
    """Compute the raw evaluator value (lower is better)."""
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (N,):
        return np.inf
    if np.any(~np.isfinite(h)):
        return np.inf
    h_clipped = np.clip(h, 0.0, 1.0)
    corr = np.correlate(h_clipped, 1.0 - h_clipped, mode="full")
    return float(np.max(corr)) / N * 2.0


def _max_corr(h):
    """Return max of full correlation vector."""
    h = np.asarray(h, dtype=np.float64)
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr))


def _smooth_objective(h, temperature=0.1):
    """Smooth approximation of max(corr) using log-sum-exp."""
    one_minus_h = 1.0 - h
    corr = np.correlate(h, one_minus_h, mode="full")
    max_corr = np.max(corr)
    if max_corr < 0:
        max_corr = 0
    shifted = (corr - max_corr) / temperature
    log_sum_exp = np.log(np.sum(np.exp(shifted)))
    return max_corr + temperature * log_sum_exp


def _epigraph_objective(z):
    """Epigraph objective: minimize t = z[N]."""
    return z[N]


def _epigraph_solve(h0, time_limit=20.0):
    """
    Solve epigraph formulation:
    min t s.t. C_k <= t for all k, sum(h)=50, 0<=h_i<=1
    """
    from scipy.optimize import minimize

    c0 = np.correlate(h0, 1.0 - h0, mode="full")
    t0 = float(c0.max()) + 0.05

    z0 = np.concatenate([h0, [t0]])

    bounds = [(0.0, 1.0)] * N + [(0.0, float('inf'))]

    def sum_constraint(x):
        return x[:N].sum() - 50.0

    constraints = [{'type': 'eq', 'fun': sum_constraint}]

    # Add inequality constraints: t - C_k >= 0
    for k in range(2 * N - 1):
        def make_ineq(k):
            def ineq(x):
                h = x[:N]
                t = x[N]
                c = np.correlate(h, 1.0 - h, mode="full")
                return t - c[k]
            return ineq
        constraints.append({'type': 'ineq', 'fun': make_ineq(k)})

    t_start = time.time()

    def callback(x):
        if time.time() - t_start > time_limit:
            raise Exception("timeout")

    try:
        res = minimize(
            _epigraph_objective,
            z0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-10},
            callback=callback,
        )
        if hasattr(res, 'x') and res.x is not None:
            h_sol = np.clip(res.x[:N], 0.0, 1.0)
            h_sol += (50.0 - h_sol.sum()) / N
            h_sol = np.clip(h_sol, 0.0, 1.0)
            return h_sol
    except Exception:
        pass

    return None


def _local_refine(h, time_limit=15.0):
    """Local search with sum-preserving swaps."""
    h = np.clip(np.asarray(h, dtype=np.float64), 0.0, 1.0).copy()
    diff = 50.0 - h.sum()
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    h += (50.0 - h.sum()) / N
    h = np.clip(h, 0.0, 1.0)

    best = h.copy()
    bestv = _value(best)
    t0 = time.time()
    step = 0.2

    while time.time() - t0 < time_limit and step > 1e-5:
        improved = False
        for _ in range(5000):
            if time.time() - t0 > time_limit:
                break
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j:
                continue
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


def _smooth_solve(h0, temperature, time_limit=15.0):
    """Solve using smooth log-sum-exp objective with SLSQP."""
    bounds_h = [(0.0, 1.0)] * N
    constraints_h = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 50.0}]

    t_start = time.time()

    def callback(x):
        if time.time() - t_start > time_limit:
            raise Exception("timeout")

    try:
        res = minimize(
            lambda h: _smooth_objective(h, temperature=temperature),
            h0,
            method='SLSQP',
            bounds=bounds_h,
            constraints=constraints_h,
            options={'maxiter': 300, 'ftol': 1e-10},
            callback=callback,
        )
        if hasattr(res, 'x') and res.x is not None:
            h_sol = np.clip(res.x, 0.0, 1.0)
            h_sol += (50.0 - h_sol.sum()) / N
            h_sol = np.clip(h_sol, 0.0, 1.0)
            return h_sol
    except Exception:
        pass

    return None


def _make_inits():
    """Generate diverse initializations."""
    inits = []

    # Constant
    inits.append(np.full(N, 0.5))

    # Alternating
    a = np.zeros(N)
    a[0::2] = 1.0
    inits.append(a)

    # Gradient
    inits.append(np.linspace(0.0, 1.0, N))

    # Reverse gradient
    inits.append(np.linspace(1.0, 0.0, N))

    # Random
    rng = np.random.default_rng(42)
    r = rng.uniform(0, 1, N)
    r += (50.0 - r.sum()) / N
    inits.append(np.clip(r, 0, 1))

    # Two-tone block
    b = np.zeros(N)
    b[:50] = 1.0
    inits.append(b)

    # Sinusoidal
    for freq in [1, 2, 3, 4, 5]:
        h = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N)
        h = np.clip(h, 0, 1)
        h += (50.0 - h.sum()) / N
        h = np.clip(h, 0, 1)
        inits.append(h)

    # Multi-frequency sinusoidal
    h = 0.5
    h += 0.25 * np.cos(2 * np.pi * 1 * np.arange(N) / N)
    h += 0.2 * np.cos(2 * np.pi * 3 * np.arange(N) / N)
    h = np.clip(h, 0, 1)
    h += (50.0 - h.sum()) / N
    h = np.clip(h, 0, 1)
    inits.append(h)

    # Periodic patterns
    inits.append(np.tile([0.0, 0.0, 1.0, 1.0, 0.5], 20))
    inits.append(np.tile([0.0, 0.5, 1.0, 0.5], 25))

    # Sawtooth
    x = np.arange(N)
    inits.append(np.clip(((x % 10) / 9.0 * 2.0 - 1.0 + 0.5), 0, 1))

    # Normalize all
    processed = []
    for h in inits:
        h = np.clip(np.asarray(h, dtype=np.float64), 0.0, 1.0)
        h += (50.0 - h.sum()) / N
        h = np.clip(h, 0.0, 1.0)
        processed.append(h)

    return processed


def construct():
    t_start = time.time()
    best_h = np.full(N, 0.5)
    best_val = _value(best_h)

    def try_update(h_candidate):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h_candidate)):
            return
        h_candidate = np.clip(h_candidate, 0.0, 1.0)
        diff = 50.0 - np.sum(h_candidate)
        if abs(diff) > 1e-9:
            h_candidate += diff / N
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
        val = _value(h_candidate)
        if val < best_val:
            best_val = val
            best_h = h_candidate.copy()

    inits = _make_inits()

    # Phase 1: Try epigraph convex solve on each init
    for i, h_init in enumerate(inits):
        if time.time() - t_start > DEADLINE - 40.0:
            break
        try_update(h_init)

        remaining = DEADLINE - 30.0 - (time.time() - t_start)
        if remaining > 5.0:
            h_sol = _epigraph_solve(h_init, time_limit=min(15.0, remaining))
            if h_sol is not None:
                try_update(h_sol)

    # Phase 2: Smooth log-sum-exp optimization from best found so far
    if time.time() - t_start < DEADLINE - 20.0:
        for temp in [0.5, 0.2, 0.1, 0.05]:
            remaining = DEADLINE - 15.0 - (time.time() - t_start)
            if remaining < 3.0:
                break
            h_sol = _smooth_solve(best_h, temp, time_limit=min(10.0, remaining))
            if h_sol is not None:
                try_update(h_sol)

    # Phase 3: Another round of epigraph from best
    if time.time() - t_start < DEADLINE - 15.0:
        remaining = DEADLINE - 10.0 - (time.time() - t_start)
        if remaining > 3.0:
            h_sol = _epigraph_solve(best_h, time_limit=min(12.0, remaining))
            if h_sol is not None:
                try_update(h_sol)

    # Phase 4: Local refinement
    if time.time() - t_start < DEADLINE - 5.0:
        remaining = DEADLINE - 3.0 - (time.time() - t_start)
        if remaining > 2.0:
            h_refined, v_refined = _local_refine(best_h, time_limit=remaining)
            try_update(h_refined)

    # Final validation
    best_h = np.clip(best_h, 0.0, 1.0)
    diff = 50.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)

    return best_h
