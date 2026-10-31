"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import time

def _compute_score(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    n = len(h)
    max_overlap = 0.0
    for k in range(1, n):
        shifted = np.roll(h, -k)
        overlap = np.sum(h * (1.0 - shifted))
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _construct_from_params(params):
    """Construct h from parameters: [a1, a2, ..., am, period, phase] where we use a repeating pattern."""
    # Interpret params as: first 'm' values are the pattern, then period and phase
    # Actually, let's use a simpler parametric form
    pass

def _construct_from_pattern(pattern_values, period, phase=0):
    """Create h by repeating pattern_values with given period, offset by phase."""
    n = len(pattern_values)
    if n == 0:
        return np.full(N, 0.5)
    # Scale pattern to have mean 0.5
    mean_val = np.mean(pattern_values)
    # We need sum = N/2, so mean must be 0.5
    # Adjust: h[i] = pattern_values[i % period] + (0.5 - mean_val)
    adjusted = pattern_values + (0.5 - mean_val)
    # Clip to [0, 1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    # Create full array
    idx = np.arange(N)
    h = adjusted[(idx + phase) % period]
    # Ensure exact sum by adjusting
    current_sum = np.sum(h)
    target_sum = N / 2.0
    diff = target_sum - current_sum
    if abs(diff) > 1e-10:
        # Distribute the difference evenly
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
        # Re-adjust if clipping changed sum
        current_sum = np.sum(h)
        diff = target_sum - current_sum
        if abs(diff) > 1e-10:
            # Find positions where we can adjust
            mask = (h > 0) & (h < 1)
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
    return h

def _objective(params):
    """Objective function for optimization."""
    period = int(params[0])
    if period < 2 or period > N:
        return 1000.0
    phase = int(params[1]) % period
    pattern_values = params[2:2+period]
    h = _construct_from_pattern(pattern_values, period, phase)
    # Check sum constraint
    if abs(np.sum(h) - N/2.0) > 1e-6:
        return 1000.0
    # Check bounds
    if np.any(h < -1e-9) or np.any(h > 1.0 + 1e-9):
        return 1000.0
    return _compute_score(h)

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # Start with the baseline
    best_h = np.full(N, 0.5)
    best_score = _compute_score(best_h)
    
    # Try a few different periods with differential evolution
    periods_to_try = [4, 5, 10, 20, 25, 50]
    
    for period in periods_to_try:
        if time.time() - start_time > deadline - 30:
            break
        
        # Define bounds for this period
        # params: [period, phase, pattern_values...]
        bounds = [(period, period), (0, period-1)] + [(0.0, 1.0)] * period
        
        def obj(params):
            return _objective(params)
        
        try:
            result = differential_evolution(
                obj, bounds, maxiter=50, popsize=10, seed=42,
                tol=1e-4, polish=False, init='latinhypercube'
            )
            if result.fun < best_score:
                params = result.x
                period_val = int(params[0])
                phase_val = int(params[1]) % period_val
                pattern = params[2:2+period_val]
                h = _construct_from_pattern(pattern, period_val, phase_val)
                if abs(np.sum(h) - N/2.0) < 1e-6 and np.all(h >= -1e-9) and np.all(h <= 1.0 + 1e-9):
                    best_h = h
                    best_score = result.fun
        except Exception:
            pass
    
    # Final check
    if abs(np.sum(best_h) - N/2.0) > 1e-6:
        best_h = np.full(N, 0.5)
    
    return best_h

# EVOLVE-BLOCK-END
