"""Erdos minimum overlap: minimize max autocorrelation of h vs 1-h."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

N = 100
TARGET_SUM = N / 2.0  # 50.0
TOL = 1e-9 * N        # 1e-7

def _value(h):
    """Compute the objective value (lower is better)."""
    g = 1.0 - h
    corr = np.correlate(h, g, mode="full")
    return np.max(corr) / N * 2.0

def _project(h):
    """Project h onto the feasible set: 0 <= h[i] <= 1, sum(h) = TARGET_SUM."""
    h = np.clip(h, 0.0, 1.0)
    diff = TARGET_SUM - h.sum()
    if abs(diff) > 1e-12:
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        # Second projection pass for tighter sum
        diff = TARGET_SUM - h.sum()
        if abs(diff) > 1e-12:
            h += diff / N
            h = np.clip(h, 0.0, 1.0)
    return h

def construct():
    """Construct 100 step heights minimizing the Erdos overlap value."""
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with seed
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _value(best_h)
    
    # Constraints: 0 <= h[i] <= 1, sum(h) = N/2
    bounds = Bounds(0.0, 1.0)
    A_sum = np.ones((1, N))
    sum_constraint = LinearConstraint(A_sum, TARGET_SUM, TARGET_SUM)
    constraints = [sum_constraint]
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initial points
    initial_points = [h0.copy()]
    
    # Random initializations
    for i in range(8):
        h_rand = rng.uniform(0, 1, N)
        h_rand = _project(h_rand)
        initial_points.append(h_rand.copy())
    
    # Sinusoidal patterns with different frequencies
    for freq in [1, 2, 3, 5, 7, 10, 15, 20, 25]:
        t = np.linspace(0, 2 * np.pi * freq, N)
        h_sine = 0.5 + 0.4 * np.sin(t)
        h_sine = _project(h_sine)
        initial_points.append(h_sine.copy())
    
    # Block patterns
    for block_size in [2, 3, 4, 5, 10]:
        h_block = np.zeros(N)
        for i in range(0, N, block_size * 2):
            h_block[i:i+block_size] = 1.0
        h_block = _project(h_block)
        initial_points.append(h_block.copy())
    
    # Alternating patterns
    for period in [2, 3, 4, 6, 8, 12]:
        h_alt = np.zeros(N)
        for i in range(N):
            h_alt[i] = 1.0 if (i % period) < period // 2 else 0.0
        h_alt = _project(h_alt)
        initial_points.append(h_alt.copy())
    
    # Optimize from each initial point
    for idx, h_init in enumerate(initial_points):
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(
                _objective,
                h_init,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            if res.success and np.all(np.isfinite(res.x)):
                h_candidate = _project(res.x)
                val = _value(h_candidate)
                if val < best_val - 1e-10:
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Final polishing with tighter tolerance
    if time.time() < deadline - 10:
        try:
            res = minimize(
                _objective,
                best_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-15}
            )
            if res.success and np.all(np.isfinite(res.x)):
                h_candidate = _project(res.x)
                val = _value(h_candidate)
                if val < best_val:
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Ensure final validity
    best_h = _project(best_h)
    
    return best_h

def _objective(h):
    """Objective: maximize the minimum margin to make max correlation small.
    We minimize max(corr) subject to constraints."""
    g = 1.0 - h
    corr = np.correlate(h, g, mode="full")
    return np.max(corr)
