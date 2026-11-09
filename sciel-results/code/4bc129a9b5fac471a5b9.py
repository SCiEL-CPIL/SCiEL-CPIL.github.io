import time
import random
import numpy as np

def count_nonzero(coeffs):
    """Count nonzero terms in a coefficient list."""
    if isinstance(coeffs, list):
        return sum(1 for c in coeffs if c != 0)
    return np.count_nonzero(coeffs)

def compute_square(coeffs):
    """Compute the exact square of a polynomial using numpy convolution."""
    arr = np.array(coeffs, dtype=np.int64)
    return np.convolve(arr, arr)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s deadline (5s safety margin)
    
    # Initial candidate: P(x) = 1 + x + x^2 + ... + x^11
    # Square has 23 nonzero terms
    best_coeffs = [1] * 12
    best_square = compute_square(best_coeffs)
    best_nonzero = count_nonzero(best_square)
    
    random.seed(42)
    
    # Precompute valid values
    valid_values = [i for i in range(-64, 65) if i != 0]
    
    while time.time() < deadline:
        # Choose a random length L between 12 and 41
        L = random.randint(12, 41)
        
        # Choose 12 distinct indices for non-zero coefficients
        # One of them MUST be L-1 to satisfy "nonzero final coefficient"
        last_idx = L - 1
        other_indices = random.sample(range(L - 1), 11)
        non_zero_indices = set(other_indices) | {last_idx}
        
        # Initialize coeffs
        coeffs = [0] * L
        
        # Assign random non-zero values
        for idx in non_zero_indices:
            coeffs[idx] = random.choice(valid_values)
            
        # Evaluate
        square = compute_square(coeffs)
        nonzero_count = count_nonzero(square)
        
        if nonzero_count < best_nonzero:
            best_nonzero = nonzero_count
            best_coeffs = coeffs[:]
            best_square = square
            
    return best_coeffs
