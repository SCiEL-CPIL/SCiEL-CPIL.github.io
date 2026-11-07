"""Piecewise-linear control-point parameterization for Erdos minimum overlap.

Key insight: represent h as a piecewise linear function defined by 10 control
points, linearly interpolated to 100 samples. This low-dimensional smooth
parameterization lets the optimizer explore global shapes (bimodal, multi-modal)
more effectively than optimizing 100 independent variables.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 100
N_CP = 10  # number of control points


def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2


def _interpolate_cp(cp):
    """
    Interpolate N_CP control points to N samples using linear interpolation.
    Control points are defined at equally spaced positions.
    """
    # Control point positions: 0, 1/(N_CP-1), 2/(N_CP-1), ..., 1
    cp_pos = np.linspace(0, 1, N_CP)
    # Sample positions: 0, 1/(N-1), 2/(N-1), ..., 1
    sample_pos = np.linspace(0, 1, N)
    # Linear interpolation
    h = np.interp(sample_pos, cp_pos, cp)
    return h


def _project_to_feasible(h):
    """Project h onto the feasible set: sum = N/2, 0 <= h <= 1."""
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - N / 2.0
    for _ in range(50):
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h > 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        else:
            mask = h < 1 - 1e-12
            if not np.any(mask):
                break
            h[mask] -= diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - N / 2.0
    return h


def _construct_from_cp(cp):
    """Build h from control points and project to feasible set."""
    h = _interpolate_cp(cp)
    h = _project_to_feasible(h)
    return h


def _objective_from_cp(cp):
    """Objective: compute correlation value from control points."""
    h = _construct_from_cp(cp)
    return _correlation_value(h)


def _local_polish(h, max_iters=50, step_sizes=None):
    """
    Local search: try small adjustments to individual elements.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    n = len(best_h)
    
    if step_sizes is None:
        step_sizes = [0.005, 0.01, 0.02, 0.05, 0.1]
    
    for it in range(max_iters):
        improved = False
        for i in range(n):
            for d in step_sizes:
                for sign in [+1, -1]:
                    new_h = best_h.copy()
                    new_h[i] += sign * d
                    if new_h[i] < 0 or new_h[i] > 1:
                        continue
                    needed = -sign * d
                    # Find candidate j to adjust
                    best_j = -1
                    best_j_val = np.inf
                    for j in range(n):
                        if j == i:
                            continue
                        new_hj = best_h[j] + needed
                        if 0 <= new_hj <= 1:
                            # Heuristic: prefer j that minimizes correlation impact
                            # Simple: pick the one farthest from i
                            dist = abs(i - j)
                            if dist > (best_j_val if best_j >= 0 else -1):
                                best_j = j
                                best_j_val = dist
                    if best_j < 0:
                        continue
                    new_h[best_j] = best_h[best_j] + needed
                    val = _correlation_value(new_h)
                    if val < best_val - 1e-13:
                        best_val = val
                        best_h = new_h
                        improved = True
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
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential evolution over control points
    bounds = [(0.0, 1.0)] * N_CP
    
    def de_objective(cp):
        return _objective_from_cp(cp)
    
    de_deadline = start_time + 60.0
    
    try:
        def timed_de_objective(cp):
            if time.time() > de_deadline:
                raise TimeoutError()
            return de_objective(cp)
        
        result = differential_evolution(
            timed_de_objective,
            bounds=bounds,
            maxiter=200,
            popsize=20,
            tol=1e-10,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        if result.x is not None:
            h_de = _construct_from_cp(result.x)
            val_de = _correlation_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Phase 2: Multi-start SLSQP on control points
    rng = np.random.default_rng(123)
    
    # Generate diverse starting points in control-point space
    starts = [best_h.copy()[:N_CP]]  # Approximate: use first N_CP values as initial cp
    
    # Actually, let's generate proper control point starts
    starts = []
    
    # From current best: find control points that best approximate
    # For simplicity, use evenly sampled values from best_h
    cp_from_best = np.interp(np.linspace(0, 1, N_CP), np.linspace(0, 1, N), best_h)
    starts.append(cp_from_best)
    
    # Random feasible starts
    for _ in range(10):
        cp_rand = rng.uniform(0, 1, N_CP)
        starts.append(cp_rand)
    
    # Structured: alternating high/low
    cp_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N_CP)])
    starts.append(cp_alt)
    
    # Structured: bimodal (high at ends, low in middle)
    cp_bi = np.array([1.0, 0.8, 0.3, 0.2, 0.2, 0.2, 0.2, 0.3, 0.8, 1.0])
    starts.append(cp_bi)
    
    # Structured: unimodal peak in middle
    cp_uni = np.array([0.0, 0.2, 0.5, 0.8, 1.0, 1.0, 0.8, 0.5, 0.2, 0.0])
    starts.append(cp_uni)
    
    # Sine wave patterns
    for freq in [1, 2, 3]:
        cp_sin = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.linspace(0, 1, N_CP))
        cp_sin = np.clip(cp_sin, 0, 1)
        starts.append(cp_sin)
    
    # Cosine wave patterns
    for freq in [1, 2, 3]:
        cp_cos = 0.5 + 0.4 * np.cos(2 * np.pi * freq * np.linspace(0, 1, N_CP))
        cp_cos = np.clip(cp_cos, 0, 1)
        starts.append(cp_cos)
    
    # Linear ramps
    cp_ramp_up = np.linspace(0, 1, N_CP)
    cp_ramp_down = np.linspace(1, 0, N_CP)
    starts.append(cp_ramp_up)
    starts.append(cp_ramp_down)
    
    # Triangular wave
    cp_tri = np.array([0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5])
    starts.append(cp_tri)
    
    slsqp_deadline = start_time + 105.0
    
    for idx, cp_start in enumerate(starts):
        if time.time() > slsqp_deadline - 8.0:
            break
        
        cp_start = np.clip(cp_start, 0, 1)
        
        # Optimize control points with SLSQP
        try:
            def cp_objective(cp):
                h = _construct_from_cp(cp)
                return _correlation_value(h)
            
            # Use the sum constraint via projection, so we optimize cp freely
            # and project after
            res = minimize(
                cp_objective,
                cp_start,
                method='SLSQP',
                bounds=[(0.0, 1.0)] * N_CP,
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            cp_opt = np.clip(res.x, 0, 1)
            h_opt = _construct_from_cp(cp_opt)
            val = _correlation_value(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Phase 3: Local search polish on the full h vector
    if time.time() < deadline - 5.0:
        try:
            best_h = _local_polish(best_h, max_iters=30)
            best_val = _correlation_value(best_h)
        except Exception:
            pass
    
    # Final projection
    best_h = _project_to_feasible(best_h)
    final_val = _correlation_value(best_h)
    if final_val < best_val:
        best_val = final_val
    
    return best_h
