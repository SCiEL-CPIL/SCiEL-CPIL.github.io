"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import logsumexp

_start_time = time.time()
_TIME_LIMIT = 130  # safety margin before 150s kill

def _compute_overlap_grid(h, n_shifts=200):
    """Compute overlap of h with shifted (1-h) on a grid of shifts."""
    # h is array of N step heights, each step width = 2/N
    # Domain is [0, 2]
    # For shift s in [0, 2), overlap(s) = integral of h(x) * (1-h(x+s)) dx
    # We need to handle wrap-around or zero-padding? Let's assume zero outside [0,2]
    
    step_width = 2.0 / N
    shifts = np.linspace(0, 2.0, n_shifts, endpoint=False)
    
    overlaps = np.zeros(n_shifts)
    
    for idx, s in enumerate(shifts):
        # For each step i of h, compute integral of h_i * (1 - h_shifted(x)) over the step
        # h_shifted(x) = h(x+s) if x+s in [0,2], else 0
        # So 1 - h_shifted(x) = 1 - h(x+s) if x+s in [0,2], else 1
        
        # Discretize: for step i, x in [i*sw, (i+1)*sw)
        # x+s in [(i*sw+s), ((i+1)*sw+s))
        # We need to find which steps of h(x+s) fall in this interval
        
        # This is complex for continuous shifts. Let's use a fine grid approach.
        # Actually, let's just numerically integrate using many points.
        
        # Simpler: use a fine grid of x values
        x_fine = np.linspace(0, 2.0, 2000, endpoint=False)
        h_vals = np.interp(x_fine, np.linspace(0, 2.0, N+1, endpoint=False)[:-1] + step_width/2, h, period=2.0)
        # Wait, this is getting complicated. Let me use a different approach.
        
        # Actually for step functions, I can compute this more directly.
        # Let me use the fact that both h and 1-h are step functions.
        
        pass
    
    return overlaps

# Let me try a cleaner approach
def _overlap_at_shift(h, s, N):
    """Compute overlap integral of h(x) * (1-h(x+s)) dx over [0,2]."""
    sw = 2.0 / N
    # h is defined on [0, 2] as steps
    # h(x+s) is defined on [-s, 2-s], we only care about x in [0,2] where x+s in [0,2]
    # So x in [0, min(2, 2-s)] for s >= 0
    
    if s >= 2.0:
        return 0.0
    
    # For the part where x+s is in [0,2]: x in [0, 2-s]
    # For x in [2-s, 2]: h(x+s) = 0 (outside domain), so 1-h(x+s) = 1
    
    # Part 1: x in [0, 2-s], overlap = integral of h(x)*(1-h(x+s)) dx
    # Part 2: x in [2-s, 2], overlap = integral of h(x)*1 dx
    
    # Part 2:
    # Find which steps of h are in [2-s, 2]
    x_start_p2 = 2.0 - s
    # Step indices for x in [x_start_p2, 2]
    i_start = int(np.ceil(x_start_p2 / sw - 1e-12))
    i_end = N  # exclusive
    if i_start >= i_end:
        part2 = 0.0
    else:
        # For each step i in [i_start, i_end), the portion in [x_start_p2, 2]
        part2 = 0.0
        for i in range(max(0, i_start), min(N, i_end)):
            step_start = i * sw
            step_end = (i + 1) * sw
            # Intersection of [step_start, step_end] with [x_start_p2, 2.0]
            inter_start = max(step_start, x_start_p2)
            inter_end = min(step_end, 2.0)
            if inter_end > inter_start:
                part2 += h[i] * (inter_end - inter_start)
    
    # Part 1: x in [0, 2-s], h(x)*(1-h(x+s))
    # For step i of h (x in [i*sw, (i+1)*sw)), x+s in [(i*sw+s), ((i+1)*sw+s))
    # We need h(x+s) which is step j where j*sw <= x+s < (j+1)*sw
    # This requires careful handling of partial overlaps
    
    # Let me just do it numerically with a fine grid for simplicity
    n_pts = 4000
    x = np.linspace(0, 2.0 - s, n_pts, endpoint=False) if s > 0 else np.linspace(0, 2.0, n_pts, endpoint=False)
    # h at x
    idx_x = np.minimum((x / sw).astype(int), N - 1)
    h_x = h[idx_x]
    # h at x+s
    xs = x + s
    idx_xs = np.minimum((xs / sw).astype(int), N - 1)
    h_xs = h[idx_xs]
    
    dx = (2.0 - s) / n_pts if s > 0 else 2.0 / n_pts
    part1 = np.sum(h_x * (1 - h_xs)) * dx
    
    return part1 + part2

def _max_overlap(h, n_shifts=100):
    """Maximum overlap over a grid of shifts."""
    shifts = np.linspace(0, 2.0, n_shifts, endpoint=False)
    overlaps = [_overlap_at_shift(h, s, N) for s in shifts]
    return max(overlaps)

def _soft_max_overlap(h, n_shifts=100, beta=50):
    """Log-sum-exp smoothed max overlap."""
    shifts = np.linspace(0, 2.0, n_shifts, endpoint=False)
    overlaps = np.array([_overlap_at_shift(h, s, N) for s in shifts])
    return logsumexp(beta * overlaps) / beta

def _objective(h):
    """Objective: soft max overlap."""
    return _soft_max_overlap(h)

def _constraints_check(h):
    return np.all(h >= -1e-10) and np.all(h <= 1 + 1e-10) and abs(np.sum(h) - N/2) < 1e-6

def construct():
    global _start_time
    deadline = _start_time + _TIME_LIMIT
    
    # Initial: all 0.5
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _max_overlap(best_h, 200)
    
    # Try scipy optimization
    try:
        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - N/2},
        ]
        bounds = [(0, 1)] * N
        
        # Use SLSQP
        result = minimize(
            _objective, h0, method='SLSQP',
            bounds=bounds, constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-8}
        )
        
        if result.success and _constraints_check(result.x):
            val = _max_overlap(result.x, 200)
            if val < best_val:
                best_h = result.x.copy()
                best_val = val
        
        if time.time() > deadline - 20:
            return best_h
    
    except Exception:
        pass
    
    # Try a different initial condition: alternating pattern
    if time.time() < deadline - 30:
        try:
            # Try h_i = 1 for even, 0 for odd (or some pattern)
            h1 = np.zeros(N)
            h1[::2] = 1.0
            # Adjust to satisfy sum = N/2
            # Current sum = N/2 already if N is even
            if _constraints_check(h1):
                result2 = minimize(
                    _objective, h1, method='SLSQP',
                    bounds=bounds, constraints=constraints,
                    options={'maxiter': 500, 'ftol': 1e-8}
                )
                if result2.success and _constraints_check(result2.x):
                    val2 = _max_overlap(result2.x, 200)
                    if val2 < best_val:
                        best_h = result2.x.copy()
                        best_val = val2
        except Exception:
            pass
    
    # Polish with more iterations if time permits
    if time.time() < deadline - 10:
        try:
            result3 = minimize(
                _objective, best_h, method='SLSQP',
                bounds=bounds, constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )
            if result3.success and _constraints_check(result3.x):
                val3 = _max_overlap(result3.x, 200)
                if val3 < best_val:
                    best_h = result3.x.copy()
                    best_val = val3
        except Exception:
            pass
    
    # Ensure constraints are met
    best_h = np.clip(best_h, 0, 1)
    # Fix sum
    diff = N/2 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
