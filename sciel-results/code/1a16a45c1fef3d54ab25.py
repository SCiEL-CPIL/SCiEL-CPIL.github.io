"""
Erdos minimum overlap - optimized approach.

Strategy:
1. Fast global search using differential evolution over a compact parameterization.
2. Multi-start SLSQP with log-sum-exp smoothing for the max-correlation objective.
3. Coordinate descent polish with sum-preserving swaps.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0


def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.asarray(h, dtype=float).copy()
    h = np.clip(h, 0.0, 1.0)
    target = N / 2.0
    for _ in range(50):
        diff = np.sum(h) - target
        if abs(diff) < 1e-15:
            break
        if diff > 0:
            mask = h > 1e-15
            cnt = np.sum(mask)
            if cnt == 0:
                break
            h[mask] -= diff / cnt
        else:
            mask = h < 1.0 - 1e-15
            cnt = np.sum(mask)
            if cnt == 0:
                break
            h[mask] += (-diff) / cnt
        h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max_corr(h, lam=100.0):
    """Log-sum-exp approximation of max correlation, scaled to match evaluator."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    m = np.max(corr)
    if not np.isfinite(m):
        return 1e10
    shifted = corr - m
    lse = m + np.log(np.sum(np.exp(shifted)) / len(corr))
    return lse / N * 2.0


def _construct_from_params(params):
    """
    Build h from parameters:
    params[0:6] - amplitudes for sin/cos at freq 1,2,3
    params[6:8] - linear and quadratic coefficients
    """
    i = np.arange(N, dtype=float)
    t = i / N - 0.5
    h = np.full(N, 0.5)
    h += params[0] * np.sin(2 * np.pi * i / N)
    h += params[1] * np.cos(2 * np.pi * i / N)
    h += params[2] * np.sin(4 * np.pi * i / N)
    h += params[3] * np.cos(4 * np.pi * i / N)
    h += params[4] * np.sin(6 * np.pi * i / N)
    h += params[5] * np.cos(6 * np.pi * i / N)
    h += params[6] * t
    h += params[7] * (t * t - 1.0 / 12.0)
    return h


def _slsqp_optimize(h_start, deadline, lam=200.0):
    """Run SLSQP with smooth objective."""
    if time.time() > deadline:
        return h_start

    def obj(x):
        if time.time() > deadline:
            raise TimeoutError()
        return _smooth_max_corr(x, lam=lam)

    try:
        res = minimize(
            obj,
            h_start,
            method='SLSQP',
            bounds=[(0.0, 1.0)] * N,
            constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}],
            options={'maxiter': 300, 'ftol': 1e-15}
        )
        if res.success or res.fun < _correlation_value(h_start):
            h_opt = _project_to_feasible(res.x)
            return h_opt
    except (TimeoutError, Exception):
        pass
    return h_start


def _coordinate_descent(h, max_rounds=100, deadline=None):
    """
    Coordinate descent: for each pair (i,j), try swapping small amounts
    to reduce the max correlation while preserving sum.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = N

    for rnd in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            for j in range(i + 1, n):
                # Try moving delta from i to j (or vice versa)
                for delta in [0.01, 0.02, 0.05, 0.1]:
                    for sign in [+1, -1]:
                        new_h = best_h.copy()
                        new_h[i] += sign * delta
                        new_h[j] -= sign * delta
                        if new_h[i] < -1e-12 or new_h[i] > 1 + 1e-12:
                            continue
                        if new_h[j] < -1e-12 or new_h[j] > 1 + 1e-12:
                            continue
                        new_h = np.clip(new_h, 0.0, 1.0)
                        val = _correlation_value(new_h)
                        if val < best_val - 1e-14:
                            best_val = val
                            best_h = new_h
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break

    return best_h


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)

    # Phase 1: Differential evolution over compact parameter space
    n_params = 8
    bounds = [(-0.4, 0.4)] * 6 + [(-0.3, 0.3), (-0.3, 0.3)]
    de_deadline = start_time + 30.0

    try:
        def de_obj(params):
            if time.time() > de_deadline:
                raise TimeoutError()
            h = _construct_from_params(params)
            h = _project_to_feasible(h)
            return _correlation_value(h)

        result = differential_evolution(
            de_obj,
            bounds=bounds,
            maxiter=80,
            popsize=12,
            tol=1e-7,
            mutation=(0.4, 1.0),
            recombination=0.8,
            seed=42,
            polish=False,
            init='sobol'
        )
        if result.x is not None:
            h_de = _project_to_feasible(_construct_from_params(result.x))
            val_de = _correlation_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except (TimeoutError, Exception):
        pass

    # Phase 2: Generate diverse starting points and optimize with SLSQP
    rng = np.random.default_rng(42)

    starts = [best_h.copy()]

    # Alternating patterns
    for period in [2, 3, 4, 5, 6, 7, 10]:
        h_alt = np.zeros(N)
        for i in range(N):
            h_alt[i] = 1.0 if (i % period) < period // 2 else 0.0
        starts.append(_project_to_feasible(h_alt))

    # Sine waves
    for freq in [1, 2, 3, 4, 5, 7, 8, 10, 12, 15]:
        for amp in [0.2, 0.3, 0.4, 0.49]:
            h_sin = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
            starts.append(_project_to_feasible(h_sin))
            h_cos = 0.5 + amp * np.cos(2 * np.pi * freq * np.arange(N) / N)
            starts.append(_project_to_feasible(h_cos))

    # Random feasible
    for _ in range(10):
        h_rand = rng.uniform(0, 1, N)
        starts.append(_project_to_feasible(h_rand))

    # Triangular waves
    for period in [5, 10, 14, 20, 25, 33]:
        h_tri = np.zeros(N)
        for i in range(N):
            t = (i % period) / period
            if t < 0.5:
                h_tri[i] = 2 * t
            else:
                h_tri[i] = 2 - 2 * t
        starts.append(_project_to_feasible(h_tri))

    slsqp_deadline = start_time + 125.0

    # Sort starts by initial value to process best ones first
    starts_with_vals = []
    for s in starts:
        v = _correlation_value(s)
        starts_with_vals.append((v, s))
    starts_with_vals.sort(key=lambda x: x[0])

    for idx, (v_start, h_start) in enumerate(starts_with_vals):
        if time.time() > slsqp_deadline:
            break
        if v_start > best_val * 1.5:
            continue  # Skip very bad starts

        h_opt = _slsqp_optimize(h_start, slsqp_deadline - 2.0, lam=200.0)
        val = _correlation_value(h_opt)
        if val < best_val:
            best_val = val
            best_h = h_opt.copy()

        # Second pass with tighter lambda
        if time.time() < slsqp_deadline - 5.0:
            h_opt2 = _slsqp_optimize(h_opt, slsqp_deadline - 3.0, lam=500.0)
            val2 = _correlation_value(h_opt2)
            if val2 < best_val:
                best_val = val2
                best_h = h_opt2.copy()

    # Phase 3: Coordinate descent polish
    cd_deadline = start_time + 138.0
    if time.time() < cd_deadline:
        best_h = _coordinate_descent(best_h, max_rounds=50, deadline=cd_deadline)
        best_val = _correlation_value(best_h)

    # Final projection
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val

    return best_h
