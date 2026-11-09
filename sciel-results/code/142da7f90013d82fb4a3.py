"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_autocorr(x):
    """Compute the aperiodic autocorrelation of a signed sequence."""
    n = len(x)
    ac = np.correlate(x, x, mode='full')[n-1:]
    return ac


def _compute_bound(x):
    """Compute the upper bound on C3 from the autocorrelation sidelobes."""
    ac = _compute_autocorr(x)
    peak = ac[0]
    if peak == 0:
        return 1.0
    sidelobes = np.abs(ac[1:])
    max_sidelobe = np.max(sidelobes) if len(sidelobes) > 0 else 0.0
    return max_sidelobe / peak


def _flip_bit(x, i):
    """Flip the sign at position i."""
    x[i] = -x[i]


def _local_search(x, max_time=100.0):
    """Perform local search to minimize the autoconvolution bound."""
    start_time = time.time()
    current = x.copy()
    current_bound = _compute_bound(current)
    best = current.copy()
    best_bound = current_bound
    
    n = len(current)
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        improved = False
        # Try flipping each bit
        for i in range(n):
            if time.time() - start_time > max_time - 5:
                break
            
            # Flip bit i
            current[i] = -current[i]
            new_bound = _compute_bound(current)
            
            if new_bound < current_bound:
                current_bound = new_bound
                improved = True
                if new_bound < best_bound:
                    best_bound = new_bound
                    best = current.copy()
            else:
                # Revert
                current[i] = -current[i]
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 10:
                break
        else:
            no_improve_count = 0
    
    return best, best_bound


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Try multiple random initializations
    best_result = None
    best_score = float('inf')
    
    n_attempts = 0
    while time.time() < deadline - 10:
        n_attempts += 1
        # Random signed initialization
        x = np.random.choice([-1, 1], size=N)
        
        # Run local search with remaining time budget
        remaining = deadline - time.time() - 2
        if remaining < 1:
            break
        
        result, score = _local_search(x, max_time=remaining)
        
        if score < best_score:
            best_score = score
            best_result = result.copy()
        
        # If we've found a very good result, stop early
        if best_score < 0.05:
            break
    
    # If no good result found, fall back to a simple random
    if best_result is None:
        best_result = np.random.choice([-1, 1], size=N)
    
    return best_result
# EVOLVE-BLOCK-END
