import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Precompute powers of z for a dense grid
    # We will use 20000 points for initial screening, then refine
    M = 20000
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    # z^k for k=0..23
    # z = exp(i * angles)
    # We can compute z^k iteratively or use np.exp
    # For speed, let's precompute the Vandermonde-like matrix or just evaluate directly
    # Direct evaluation: P(z) = sum c_k z^k
    # Let's create an array of z^k for all k
    # z = exp(i * angles)
    z = np.exp(1j * angles)
    # Z[k, m] = z[m]^k
    Z = np.zeros((24, M), dtype=complex)
    Z[0, :] = 1.0
    for k in range(1, 24):
        Z[k, :] = Z[k-1, :] * z
    
    def evaluate(coeffs):
        # coeffs is a numpy array of shape (24,) with values +1/-1
        # P(z) = sum c_k z^k
        # = coeffs @ Z (where Z is 24xM)
        vals = coeffs @ Z
        return np.max(np.abs(vals))
    
    # Initial Rudin-Shapiro sequence
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=float)
    best_score = evaluate(best_coeffs)
    
    # Local search: flip bits
    # We can try flipping each bit, or pairs of bits
    # Since N=24, there are 2^24 possibilities. Exhaustive is too slow.
    # Greedy local search with random restarts.
    
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Start from random or best
        if r == 0:
            current = best_coeffs.copy()
        else:
            # Random start with some bias towards best? Or just random
            current = np.random.choice([-1, 1], size=24).astype(float)
            
        current_score = evaluate(current)
        
        improved = True
        while improved:
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(24):
                # Try flipping bit i
                original_val = current[i]
                current[i] = -original_val
                new_score = evaluate(current)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    current[i] = original_val
        
        if current_score < best_score:
            best_score = current_score
            best_coeffs = current.copy()
            
    return [int(c) for c in best_coeffs]
# EVOLVE-BLOCK-END
