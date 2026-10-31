"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_c3_bound(x):
    """Compute the C3 autoconvolution upper bound for a signed step function."""
    # x is a vector of N signed step heights
    # Compute autoconvolution
    conv = np.convolve(x, x, mode='full')
    # The autoconvolution has length 2*N - 1
    # C3 bound is related to the maximum sidelobe magnitude
    # The main lobe is at the center, sidelobes are elsewhere
    center = N - 1
    # Sidelobes are all elements except the center
    sidelobes = np.concatenate([conv[:center], conv[center+1:]])
    return np.max(np.abs(sidelobes))


def construct():
    """Construct a signed step function to minimize the C3 autoconvolution upper bound."""
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin before 120s limit
    
    # Initialize with random signs
    x = np.random.choice([-1.0, 1.0], size=N)
    best_x = x.copy()
    best_val = _compute_c3_bound(best_x)
    
    # Simple local search: flip random elements if it improves
    np.random.seed(42)
    x = np.random.choice([-1.0, 1.0], size=N)
    best_x = x.copy()
    best_val = _compute_c3_bound(best_x)
    
    iters = 0
    while time.time() < deadline:
        iters += 1
        # Flip a random position
        idx = np.random.randint(N)
        x[idx] *= -1
        val = _compute_c3_bound(x)
        
        if val < best_val:
            best_val = val
            best_x = x.copy()
        else:
            # Revert if worse
            x[idx] *= -1
    
    return best_x
# EVOLVE-BLOCK-END
