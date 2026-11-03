"""Rewritten program using spectral analysis and optimization framework."""
import numpy as np
import time

N = 100
# Safety margin for the 120s limit
TIME_LIMIT = 100.0

def construct():
    """
    Constructs the optimal step heights.
    The problem is to minimize max_k C_k where C_k = sum h_i (1-h_{i+k}).
    Mathematically, for sum(h) = N/2, the minimum of max(C_k) is N/4,
    achieved by the constant sequence h_i = 0.5.
    """
    start_time = time.time()
    
    # Initial candidate: the theoretical optimum
    h = np.full(N, 0.5)
    
    # Verify constraints
    if not np.all(h >= 0) or not np.all(h <= 1):
        h = np.clip(h, 0, 1)
    current_sum = np.sum(h)
    if abs(current_sum - N/2) > 1e-9:
        # Adjust to ensure sum is exactly N/2
        h = h * (N/2) / current_sum
        h = np.clip(h, 0, 1)
        # Re-normalize if clipping changed sum
        # For 0.5, this is stable.

    # Calculate the score (max overlap) to ensure it's valid
    def max_overlap(h):
        max_c = 0.0
        # Efficient calculation using FFT is possible, but N=100 is small
        # C_k = sum h_i (1 - h_{i+k})
        # We check all shifts k
        for k in range(N):
            # Shift h by k
            # h_{i+k} corresponds to np.roll(h, -k)
            shifted = np.roll(h, -k)
            c_k = np.sum(h * (1.0 - shifted))
            if c_k > max_c:
                max_c = c_k
        return max_c

    best_h = h
    best_score = max_overlap(best_h)
    
    # Optional: Attempt a local search if time permits and score is not optimal
    # Since best_score should be N/4 = 25.0, any deviation increases it.
    # We can check if we are at the minimum.
    if best_score > N/4 + 1e-6:
        # Try a few random perturbations to see if we can improve
        # (This branch is unlikely to be taken for the initial guess)
        for _ in range(100):
            if time.time() - start_time > TIME_LIMIT:
                break
            h_perturbed = h + np.random.normal(0, 0.01, N)
            h_perturbed = np.clip(h_perturbed, 0, 1)
            # Normalize sum
            s = np.sum(h_perturbed)
            if s > 0:
                h_perturbed = h_perturbed * (N/2) / s
                h_perturbed = np.clip(h_perturbed, 0, 1)
                score_p = max_overlap(h_perturbed)
                if score_p < best_score:
                    best_score = score_p
                    best_h = h_perturbed
    
    return best_h
