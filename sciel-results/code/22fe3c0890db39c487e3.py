"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Construct a signed step function with 400 equal intervals on [-1/4, 1/4].
    The goal is to minimize the autoconvolution upper bound C3.
    The seed program returns all ones, which gives a baseline score of 1.0.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 115s out of 150s wall-clock limit
    
    # Initialize with a pattern that reduces autoconvolution
    # Alternating +1 and -1 reduces the L1 norm of the autoconvolution
    x = np.ones(N)
    x[::2] = -1.0  # Alternating pattern
    
    def compute_c3(x):
        """Compute the autoconvolution upper bound C3."""
        # Autoconvolution: c[k] = sum_j x[j] * x[j+k] for each lag k
        # We want to minimize some upper bound on |c[k]| or sum of |c[k]|
        # The paper mentions "autoconvolution upper bound"
        # Let's use the L2 norm of the autoconvolution as the metric
        c = np.correlate(x, x, mode='full')
        return np.sum(c**2)  # L2 norm squared of autoconvolution
    
    best_x = x.copy()
    best_score = compute_c3(x)
    
    # Local search: try flipping individual positions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            # Try flipping position i
            old_val = x[i]
            x[i] = -old_val
            new_score = compute_c3(x)
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
                improved = True
            else:
                x[i] = old_val
    
    # Try some structured patterns
    patterns = [
        # Alternating
        lambda n: np.where(np.arange(n) % 2 == 0, 1.0, -1.0),
        # Block pattern
        lambda n: np.where(np.arange(n) % 4 < 2, 1.0, -1.0),
        # Random
        lambda n: np.random.choice([-1.0, 1.0], size=n),
    ]
    
    for _ in range(50):
        if time.time() > deadline:
            break
        for pattern_func in patterns:
            if time.time() > deadline:
                break
            x = pattern_func(N)
            score = compute_c3(x)
            if score < best_score:
                best_score = score
                best_x = x.copy()
            # Local search from this pattern
            for i in range(N):
                if time.time() > deadline:
                    break
                old_val = x[i]
                x[i] = -old_val
                new_score = compute_c3(x)
                if new_score < best_score:
                    best_score = new_score
                    best_x = x.copy()
                else:
                    x[i] = old_val
    
    return best_x
# EVOLVE-BLOCK-END
