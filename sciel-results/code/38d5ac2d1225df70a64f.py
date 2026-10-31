"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_start_time = time.time()
_DEADLINE = 110  # seconds, safety margin before 150s kill


def _compute_max_overlap(h):
    """Compute max overlap of h with shifted 1-h over all cyclic shifts."""
    n = len(h)
    overlaps = np.zeros(n)
    for k in range(n):
        shifted = np.roll(h, k)
        overlaps[k] = np.dot(h, 1.0 - shifted)
    return np.max(overlaps)


def _objective(params):
    """Objective: soft-max of overlaps, with constraint handled via penalty."""
    h = np.clip(params, 0.0, 1.0)
    n = len(h)
    # Compute overlaps
    overlaps = np.zeros(n)
    for k in range(n):
        shifted = np.roll(h, k)
        overlaps[k] = np.dot(h, 1.0 - shifted)
    
    # Use log-sum-exp for a smooth approximation of max
    # beta controls the sharpness. Larger beta is closer to max.
    beta = 10.0
    max_overlap = np.log(np.sum(np.exp(beta * overlaps))) / beta
    
    # Penalty for constraint violation: sum(h) should be N/2
    # Use a large coefficient to enforce the constraint
    sum_penalty = (np.sum(h) - N / 2.0) ** 2 * 100.0
    
    return max_overlap + sum_penalty


def _project_to_constraint(h):
    """Project h to satisfy 0<=h<=1 and sum(h)=N/2 exactly."""
    h = np.clip(h, 0.0, 1.0)
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-15:
        # Distribute difference
        h = h + diff / N
        # Re-clip if distribution pushed out of bounds
        h = np.clip(h, 0.0, 1.0)
        # If still off, adjust the element closest to the bound that allows adjustment
        diff2 = N / 2.0 - np.sum(h)
        if abs(diff2) > 1e-15:
            if diff2 > 0:
                # Need to increase sum, pick index with max room (1 - h)
                idx = np.argmax(1.0 - h)
                add = min(diff2, 1.0 - h[idx])
                h[idx] += add
                # If still not enough, distribute remaining
                rem = diff2 - add
                if abs(rem) > 1e-15:
                    h = h + rem / N
                    h = np.clip(h, 0.0, 1.0)
            else:
                # Need to decrease sum, pick index with min h
                idx = np.argmin(h)
                sub = min(-diff2, h[idx])
                h[idx] -= sub
                rem = -diff2 - sub
                if abs(rem) > 1e-15:
                    h = h - rem / N
                    h = np.clip(h, 0.0, 1.0)
    return h

def _try_gradient_descent(h_init, max_iter=1000, tol=1e-9):
    """Use scipy minimize with L-BFGS-B."""
    result = minimize(
        _objective,
        h_init,
        method='L-BFGS-B',
        bounds=[(0, 1)] * N,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8}
    )
    h = _project_to_constraint(result.x)
    return h, _compute_max_overlap(h)


def construct():
    global _start_time
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Try multiple random initializations
    np.random.seed(42)
    n_tries = 0
    max_tries = 50
    
    while time.time() - _start_time < _DEADLINE:
        n_tries += 1
        # Generate random initial point
        # Mix of uniform random and structured patterns
        if n_tries % 3 == 0:
            # Alternating pattern
            h_init = np.where(np.arange(N) % 2 == 0, 1.0, 0.0)
            # Perturb
            h_init += np.random.uniform(-0.1, 0.1, N)
        elif n_tries % 3 == 1:
            # Block pattern
            h_init = np.zeros(N)
            h_init[:N//2] = 1.0
            h_init += np.random.uniform(-0.1, 0.1, N)
        else:
            # Uniform random
            h_init = np.random.uniform(0, 1, N)
        
        # Project to constraint
        h_init = _project_to_constraint(h_init)
        
        h, score = _try_gradient_descent(h_init, max_iter=500)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        if n_tries >= max_tries:
            break
    
    # Final projection to ensure exact sum
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        # Distribute adjustment evenly, then fix any out-of-bounds
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
        # If still not exact, adjust one element
        diff2 = N / 2.0 - np.sum(best_h)
        if abs(diff2) > 1e-12:
            idx = np.argmin(best_h) if diff2 > 0 else np.argmax(best_h)
            best_h[idx] += diff2
    
    return best_h
# EVOLVE-BLOCK-END
