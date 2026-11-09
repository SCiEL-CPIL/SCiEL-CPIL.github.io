import numpy as np
import time
import random

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s buffer for safety
    
    # 1. Generate Golay-Rudin-Shapiro sequence as initial guess
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=float)
    
    # 2. Setup fast evaluation
    # Use a dense grid of points on the unit circle
    num_points = 20000
    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    # Precompute the Vandermonde-like matrix E where E[j, k] = exp(i * theta_j * k)
    # Shape: (num_points, 24)
    k_indices = np.arange(24)
    # E = np.exp(1j * np.outer(thetas, k_indices))
    # To save memory and computation, we can compute on the fly if needed, but 20000x24 is small.
    E = np.exp(1j * np.outer(thetas, k_indices))
    
    def evaluate(c):
        # c is a numpy array of shape (24,) with values +1 or -1
        # Compute P(z) for all z in grid
        P = E @ c
        # Max modulus
        return np.max(np.abs(P))
    
    # Initial score
    best_score = evaluate(best_coeffs)
    current_coeffs = best_coeffs.copy()
    current_score = best_score
    
    # 3. Simulated Annealing
    T = 1.0
    alpha = 0.999
    min_T = 1e-6
    
    while T > min_T and time.time() < deadline:
        # Generate neighbor: flip 1 or 2 random coefficients
        num_flips = 1 if random.random() < 0.8 else 2
        indices = np.random.choice(24, num_flips, replace=False)
        
        new_coeffs = current_coeffs.copy()
        new_coeffs[indices] *= -1
        
        new_score = evaluate(new_coeffs)
        
        # Acceptance probability
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_coeffs = new_coeffs
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs.copy()
        
        T *= alpha
        
        # Optional: Restart logic or local search could be added, but SA is robust.
        
    return [int(x) for x in best_coeffs]
