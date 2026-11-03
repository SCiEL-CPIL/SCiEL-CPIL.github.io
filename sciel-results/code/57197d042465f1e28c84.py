"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_overlap(h):
    """
    Compute the maximum overlap of h with shifted (1-h).
    Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    We want to minimize the maximum over all shifts k.
    """
    N = len(h)
    h = np.asarray(h, dtype=float)
    one_minus_h = 1.0 - h
    
    max_overlap = 0.0
    for k in range(N):
        shifted = np.roll(h, k)
        overlap = np.dot(shifted, one_minus_h)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap

def _objective(x):
    """
    Objective for optimization: maximize the minimum 'gap' from the constraint,
    or equivalently minimize a smooth upper bound on the max overlap.
    
    We use a log-sum-exp based soft-max to make the objective smooth.
    The overlaps are O_k = sum_i h[i] * (1 - h[(i+k)%N])
    We want to minimize max_k O_k.
    
    Using log-sum-exp: log(sum(exp(beta * O_k))) / beta approximates max O_k for large beta.
    """
    N = len(x)
    # Ensure h is in [0,1] and sum is N/2
    # We'll work with raw variables and use a penalty or parameterization.
    # For simplicity in the inner loop, let's assume x is already valid h.
    # But for optimization, we need a differentiable objective.
    
    # Let's compute all overlaps
    overlaps = np.zeros(N)
    one_minus_x = 1.0 - x
    for k in range(N):
        shifted = np.roll(x, k)
        overlaps[k] = np.dot(shifted, one_minus_x)
    
    # Soft-max with beta
    beta = 50.0
    # To avoid overflow, subtract max
    max_ov = np.max(overlaps)
    log_sum = np.log(np.sum(np.exp(beta * (overlaps - max_ov)))) + beta * max_ov
    soft_max = log_sum / beta
    
    return soft_max

def _construct_valid(h_raw):
    """
    Project h_raw to a valid sequence: in [0,1], sum = N/2.
    Simple projection: clip to [0,1], then adjust to meet sum constraint.
    """
    N = len(h_raw)
    h = np.clip(h_raw, 0.0, 1.0)
    # Adjust sum
    target_sum = N / 2.0
    current_sum = np.sum(h)
    if current_sum > 0:
        h = h * (target_sum / current_sum)
        h = np.clip(h, 0.0, 1.0)
        # If clipping changed the sum, do a simple iterative correction
        for _ in range(10):
            current_sum = np.sum(h)
            diff = target_sum - current_sum
            if abs(diff) < 1e-12:
                break
            # Add/subtract uniformly from elements that have room
            if diff > 0:
                mask = h < 1.0
                if np.any(mask):
                    h[mask] += diff / np.sum(mask)
                else:
                    break
            else:
                mask = h > 0.0
                if np.any(mask):
                    h[mask] += diff / np.sum(mask)
                else:
                    break
            h = np.clip(h, 0.0, 1.0)
    else:
        h = np.full(N, 0.5)
    return h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_h = np.full(N, 0.5)
    best_score = _compute_overlap(best_h)
    
    # Try several initial points
    np.random.seed(42)
    
    initial_points = []
    
    # 1. Uniform
    initial_points.append(np.full(N, 0.5))
    
    # 2. Square wave
    sq = np.zeros(N)
    sq[::2] = 1.0
    sq = sq * 0.5  # Scale to have mean 0.5
    initial_points.append(sq)
    
    # 3. High frequency
    t = np.arange(N)
    hf = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t / N)
    initial_points.append(hf)
    
    # 4. Random
    for _ in range(5):
        r = np.random.rand(N)
        r = _construct_valid(r)
        initial_points.append(r)
    
    # Optimize from each initial point
    for i, h0 in enumerate(initial_points):
        if time.time() > deadline - 10.0:
            break
            
        h0 = _construct_valid(h0)
        
        # Use L-BFGS-B with bounds
        bounds = [(0.0, 1.0)] * N
        
        # Constraint: sum(h) = N/2
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
        
        try:
            result = minimize(
                _objective,
                h0,
                method='L-BFGS-B',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = _construct_valid(result.x)
            score = _compute_overlap(h_opt)
            
            if score < best_score:
                best_score = score
                best_h = h_opt
        except Exception as e:
            pass
    
    # Final polish: try a few more random restarts if time permits
    while time.time() < deadline - 20.0:
        r = np.random.rand(N)
        r = _construct_valid(r)
        
        try:
            result = minimize(
                _objective,
                r,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}],
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = _construct_valid(result.x)
            score = _compute_overlap(h_opt)
            
            if score < best_score:
                best_score = score
                best_h = h_opt
        except Exception:
            pass
    
    return best_h
# EVOLVE-BLOCK-END
