"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _count_sq_nonzero(coeffs):
    """
    Compute the number of non-zero terms in the square of the polynomial
    defined by the coefficient list 'coeffs'.
    """
    # coeffs is a list of integers, representing P(x) = sum c_i x^i
    # We use numpy convolution for speed
    c = np.array(coeffs, dtype=np.int64)
    sq = np.convolve(c, c)
    return np.count_nonzero(sq)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # We search for a polynomial with 12 non-zero terms.
    # Hypothesis: The best polynomial has consecutive exponents 0..11.
    # This minimizes the degree of the square (22), concentrating the terms
    # in a small range [0, 22] where cancellations are most likely.
    
    # The polynomial is represented by a list of 12 coefficients c_0..c_11.
    # All c_i must be non-zero and in [-64, 64].
    # We restrict search to small integers {-3, -2, -1, 1, 2, 3} for efficiency,
    # as larger integers are less likely to cancel perfectly in sums of products.
    
    valid_vals = [-3, -2, -1, 1, 2, 3]
    
    best_count = 23  # Max possible is 23 (no cancellation)
    best_coeffs = [1] * 12
    
    # Phase 1: Heuristic Initialization
    # Try some structured patterns known to have good cancellation properties
    inits = [
        [1] * 12,
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, 2, -1, -2, 1, 2, -1, -2, 1, 2, -1, -2],
        [1, -2, 1, -2, 1, -2, 1, -2, 1, -2, 1, -2],
        [2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1],
    ]
    
    for init in inits:
        cnt = _count_sq_nonzero(init)
        if cnt < best_count:
            best_count = cnt
            best_coeffs = init[:]
            
    # Random initializations
    for _ in range(20):
        rand_init = [random.choice(valid_vals) for _ in range(12)]
        cnt = _count_sq_nonzero(rand_init)
        if cnt < best_count:
            best_count = cnt
            best_coeffs = rand_init[:]

    # Phase 2: Local Search (Hill Climbing with Restarts)
    current_coeffs = best_coeffs[:]
    current_count = best_count
    
    no_improve_steps = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        # Pick a random index to mutate
        idx = random.randint(0, 11)
        old_val = current_coeffs[idx]
        
        # Pick a new value
        new_val = random.choice(valid_vals)
        if new_val == old_val:
            continue
            
        current_coeffs[idx] = new_val
        new_count = _count_sq_nonzero(current_coeffs)
        
        if new_count <= current_count:
            current_count = new_count
            no_improve_steps = 0
            if new_count < best_count:
                best_count = new_count
                best_coeffs = current_coeffs[:]
        else:
            # Revert
            current_coeffs[idx] = old_val
            no_improve_steps += 1
            
        # Restart if stuck
        if no_improve_steps > max_no_improve:
            current_coeffs = best_coeffs[:]
            current_count = best_count
            no_improve_steps = 0
            
            # Optional: Small random perturbation on restart
            for _ in range(3):
                idx = random.randint(0, 11)
                current_coeffs[idx] = random.choice(valid_vals)
            current_count = _count_sq_nonzero(current_coeffs)

    # Final validation and formatting
    # The problem requires a list in ascending degree, length <= 41.
    # Our representation is c_0..c_11, which corresponds to exponents 0..11.
    # The list returned should be the coefficients.
    # Ensure exactly 12 non-zero terms (guaranteed by construction with valid_vals)
    # Ensure last coefficient is non-zero (guaranteed)
    # Ensure entries in [-64, 64] (guaranteed)
    
    # Return the dense list of coefficients
    # If we used exponents 0..11, the list is just best_coeffs.
    # Length is 12, which is <= 41.
    
    return best_coeffs
# EVOLVE-BLOCK-END
