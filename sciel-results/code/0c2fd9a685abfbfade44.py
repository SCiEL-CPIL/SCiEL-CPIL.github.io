"""Piecewise-linear control point parameterization for Erdos minimum overlap.

The 100 step heights are represented as a linear interpolation of K control
points. This low-dimensional smooth parameterization lets the optimizer
discover global shapes (bimodal, oscillatory) that are hard to reach when
optimizing 100 independent variables. A smooth offset enforces the sum
constraint. A log-sum-exp soft-max surrogate with ramping lambda sharpens
the nonsmooth max-correlation objective for L-BFGS-B.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 100
K = 12  # number of control points
DEADLINE = 145.0  # safety margin before 150s kill


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return float(np.max(corr)) / N * 2.0


def _correlation_array(h):
    return np.correlate(h, 1 - h, mode="full")


def _interp_from_control(c):
    """Linearly interpolate K control points to N samples.
    Control points are placed at positions (i + 0.5) * N / K for i in 0..K-1,
    i.e., at the centers of K equal bins. We interpolate at the N sample
    positions.
    """
    c = np.asarray(c, dtype=float)
    cx = (np.arange(K) + 0.5) * (N / K)
    x = np.arange(N, dtype=float)
    return np.interp(x, cx, c)


def _enforce_sum(h, target=None):
    """Add a constant offset to make sum(h) == target, then clip and re-correct.
    Returns a feasible h (0<=h<=1, sum==target within tolerance).
    """
    if target is None:
        target = N / 2.0
    h = h.copy()
    diff = target - np.sum(h)
    h = h + diff / N
    for _ in range(50):
        h = np.clip(h, 0.0, 1.0)
        diff = target - np.sum(h)
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-12
        else:
            mask = h > 1e-12
        if not np.any(mask):
            break
        h[mask] += diff / np.sum(mask)
    return h


def _feasible_from_control(c, target=None):
    """Interpolate control points, then enforce sum and bounds."""
    h = _interp_from_control(c)
    return _enforce_sum(h, target)


def _soft_max(corr, lam):
    """Soft max via log-sum-exp, numerically stable."""
    m = np.max(corr)
    if lam <= 0:
        return m
    shifted = corr - m
    return m + np.log(np.sum(np.exp(lam * shifted)) / len(corr)) / lam


def _epigraph_objective(c, t, lam):
    """Epigraph formulation: minimize t + penalty for corr > t.
    This provides a smoother landscape than pure soft-max.
    """
    h = _feasible_from_control(c)
    corr = _correlation_array(h)
    # Penalty for exceeding t
    excess = np.maximum(0, corr - t)
    penalty = np.sum(np.exp(lam * excess)) / lam
    return t + penalty


def _objective_soft(c, lam):
    """Soft-max surrogate objective for control points c."""
    h = _feasible_from_control(c)
    corr = _correlation_array(h)
    return _soft_max(corr, lam)


def _objective_exact(c):
    h = _feasible_from_control(c)
    return _correlation_value(h)


def _local_polish_control(c, max_iters=40, step_sizes=None):
    """Local search on control points with sum re-enforcement."""
    best_c = np.array(c, dtype=float)
    best_val = _objective_exact(best_c)
    if step_sizes is None:
        step_sizes = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    n = len(best_c)
    for it in range(max_iters):
        improved = False
        for i in range(n):
            for d in step_sizes:
                for sign in (+1, -1):
                    new_c = best_c.copy()
                    new_c[i] += sign * d
                    val = _objective_exact(new_c)
                    if val < best_val - 1e-13:
                        best_val = val
                        best_c = new_c
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best_c


def _local_polish_h(h, max_iters=50, step_sizes=None):
    """Local search directly on h (100-dim), maintaining sum via a partner swap.
    Focuses on positions near the max-correlation lag for efficiency."""
    best_h = np.array(h, dtype=float)
    best_val = _correlation_value(best_h)
    if step_sizes is None:
        step_sizes = [0.002, 0.005, 0.01, 0.02, 0.05]
    n = len(best_h)
    
    # Find the lag with max correlation to focus search
    corr = np.correlate(best_h, 1 - best_h, mode="full")
    max_lag = np.argmax(corr)
    
    for it in range(max_iters):
        improved = False
        # Focus on positions that affect the max-correlation lag
        # Lag k means correlation between h[i] and (1-h)[i+k] or h[i+k] and (1-h)[i]
        focus_positions = set()
        for offset in range(-5, 6):
            pos = max_lag + offset
            if 0 <= pos < n:
                focus_positions.add(pos)
            # Also positions that pair with max_lag
            pos2 = max_lag - offset
            if 0 <= pos2 < n:
                focus_positions.add(pos2)
        
        # Convert to sorted list for deterministic order
        focus_list = sorted(focus_positions)
        
        for i in focus_list:
            for d in step_sizes:
                for sign in (+1, -1):
                    new_h = best_h.copy()
                    new_h[i] += sign * d
                    if new_h[i] < 0.0 or new_h[i] > 1.0:
                        continue
                    needed = -sign * d
                    # Try partner swaps with positions in focus list first, then others
                    cands = [j for j in focus_list if j != i]
                    cands += [j for j in range(n) if j != i and j not in focus_positions]
                    found = False
                    for j in cands:
                        val_j = best_h[j] + needed
                        if 0.0 <= val_j <= 1.0:
                            new_h[j] = val_j
                            val = _correlation_value(new_h)
                            if val < best_val - 1e-13:
                                best_val = val
                                best_h = new_h
                                improved = True
                                found = True
                                break
                    if found:
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

    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)

    def _elapsed():
        return time.time() - start_time

    def _check():
        if _elapsed() > DEADLINE:
            raise TimeoutError()

    # ---- Phase 1: Differential evolution over K control points ----
    de_deadline = start_time + 40.0
    try:
        c_lo = np.full(K, 0.0)
        c_hi = np.full(K, 1.0)

        def de_obj(c):
            if time.time() > de_deadline:
                raise TimeoutError()
            return _objective_soft(c, lam=20.0)

        res_de = differential_evolution(
            de_obj,
            bounds=list(zip(c_lo, c_hi)),
            maxiter=50,
            popsize=14,
            tol=1e-7,
            mutation=(0.4, 1.0),
            recombination=0.7,
            seed=7,
            polish=False,
            init='sobol'
        )
        if res_de.x is not None:
            h_de = _feasible_from_control(res_de.x)
            v_de = _correlation_value(h_de)
            if v_de < best_val:
                best_val = v_de
                best_h = h_de.copy()
    except TimeoutError:
        pass
    except Exception:
        pass

    _check()

    # ---- Phase 2: Multi-start L-BFGS-B with ramping lambda soft-max ----
    rng = np.random.default_rng(2024)

    starts_c = []

    # From current best: bin averages
    bin_avg = np.array([best_h[i * (N // K):(i + 1) * (N // K)].mean()
                        for i in range(K)])
    starts_c.append(bin_avg.copy())

    # Random feasible control points
    for _ in range(12):
        starts_c.append(rng.uniform(0.0, 1.0, K))

    # Structured: alternating high/low
    for phase in (0, 1):
        c_alt = np.array([0.8 if (i + phase) % 2 == 0 else 0.2 for i in range(K)])
        starts_c.append(c_alt)

    # Structured: bimodal
    c_bi = np.linspace(0.8, 0.2, K)
    starts_c.append(c_bi.copy())
    starts_c.append(c_bi[::-1].copy())

    # Structured: oscillatory at several frequencies
    for freq in (1, 2, 3, 4, 5):
        t = np.arange(K)
        c_osc = 0.5 + 0.35 * np.sin(2 * np.pi * freq * t / K)
        starts_c.append(np.clip(c_osc, 0.0, 1.0))

    # Structured: triangular wave
    for phase in (0, 0.5):
        t = np.arange(K)
        c_tri = 0.5 + 0.35 * np.abs(2 * ((t + phase * K) / K) % 2 - 1) - 0.35
        starts_c.append(np.clip(c_tri, 0.0, 1.0))

    slsqp_deadline = start_time + 135.0

    for idx, c_start in enumerate(starts_c):
        if time.time() > slsqp_deadline - 12.0:
            break
        _check()

        current_c = np.clip(np.asarray(c_start, dtype=float), 0.0, 1.0)

        # Ramping lambda soft-max optimization with more aggressive ramp
        for lam in (10.0, 50.0, 200.0, 800.0, 2000.0, 5000.0, 10000.0):
            if time.time() > slsqp_deadline - 5.0:
                break

            def obj(x, lam=lam):
                return _objective_soft(x, lam)

            try:
                res = minimize(
                    obj,
                    current_c,
                    method='L-BFGS-B',
                    bounds=[(0.0, 1.0)] * K,
                    options={'maxiter': 800, 'ftol': 1e-18, 'gtol': 1e-14}
                )
                c_opt = np.clip(res.x, 0.0, 1.0)
                h_opt = _feasible_from_control(c_opt)
                v = _correlation_value(h_opt)
                if v < best_val:
                    best_val = v
                    best_h = h_opt.copy()
                current_c = c_opt
            except Exception:
                pass

        # Final exact polish on control points with epigraph formulation
        if time.time() > slsqp_deadline - 4.0:
            break
        try:
            # First try direct exact
            def obj_exact(x):
                return _objective_exact(x)

            res_e = minimize(
                obj_exact,
                current_c,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * K,
                options={'maxiter': 1200, 'ftol': 1e-20, 'gtol': 1e-16}
            )
            c_e = np.clip(res_e.x, 0.0, 1.0)
            h_e = _feasible_from_control(c_e)
            v_e = _correlation_value(h_e)
            if v_e < best_val:
                best_val = v_e
                best_h = h_e.copy()
                current_c = c_e
            
            # Then try epigraph formulation for additional refinement
            if time.time() < slsqp_deadline - 2.0:
                t_init = best_val + 0.01
                def epigraph_obj(x, t, lam=100.0):
                    h = _feasible_from_control(x)
                    corr = _correlation_array(h)
                    excess = np.maximum(0, corr - t)
                    penalty = np.sum(np.exp(lam * excess)) / lam
                    return t + penalty
                
                # Optimize over [c, t] jointly
                x0 = np.concatenate([current_c, [t_init]])
                bounds_ep = [(0.0, 1.0)] * K + [(0.0, 1.0)]
                
                def epigraph_obj_full(z, lam=100.0):
                    c_part = z[:K]
                    t_part = z[K]
                    return epigraph_obj(c_part, t_part, lam)
                
                res_ep = minimize(
                    epigraph_obj_full,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds_ep,
                    options={'maxiter': 600, 'ftol': 1e-18, 'gtol': 1e-14}
                )
                c_ep = np.clip(res_ep.x[:K], 0.0, 1.0)
                h_ep = _feasible_from_control(c_ep)
                v_ep = _correlation_value(h_ep)
                if v_ep < best_val:
                    best_val = v_ep
                    best_h = h_ep.copy()
                    current_c = c_ep
        except Exception:
            pass

    _check()

    # ---- Phase 3: Local polish on control points, then on h ----
    if time.time() < start_time + DEADLINE - 10.0:
        try:
            bin_avg = np.array([best_h[i * (N // K):(i + 1) * (N // K)].mean()
                                for i in range(K)])
            c_pol = _local_polish_control(bin_avg, max_iters=80)
            h_pol = _feasible_from_control(c_pol)
            v_pol = _correlation_value(h_pol)
            if v_pol < best_val:
                best_val = v_pol
                best_h = h_pol.copy()
        except Exception:
            pass

    if time.time() < start_time + DEADLINE - 5.0:
        try:
            best_h = _local_polish_h(best_h, max_iters=60)
            v = _correlation_value(best_h)
            if v < best_val:
                best_val = v
        except Exception:
            pass
    
    # Final high-precision refinement pass
    if time.time() < start_time + DEADLINE - 2.0:
        try:
            # One more round of very fine local search
            best_h = _local_polish_h(best_h, max_iters=20, step_sizes=[0.001, 0.002, 0.005])
            v = _correlation_value(best_h)
            if v < best_val:
                best_val = v
        except Exception:
            pass

    # ---- Final feasibility guarantee ----
    best_h = _enforce_sum(best_h)
    if not np.all(np.isfinite(best_h)):
        best_h = np.full(N, 0.5)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val

    return best_h
