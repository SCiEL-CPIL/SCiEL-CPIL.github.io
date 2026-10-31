"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
def _compute_max_overlap(h):
    """Compute the maximum overlap of h with shifted (1-h)."""
    n = len(h)
    max_ov = 0.0
    for s in range(n):
        shifted = np.roll(h, s)
        ov = np.sum(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    # Start with a pattern-based initialization
    # Try a pattern where we alternate blocks to reduce autocorrelation
    # Use a construction based on low-discrepancy or balanced sequences
    
    # First, try a simple optimization from a good starting point
    # Initialize with a sawtooth-like or block pattern
    
    # Try different initial patterns
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Try a block pattern: alternating high and low blocks
    # Pattern: 0.7, 0.3 repeating, adjusted to sum to N/2
    for block_size in [1, 2, 3, 4, 5, 7, 10, 14, 25]:
        if N % block_size == 0:
            n_blocks = N // block_size
            # Create a pattern with values a and b alternating
            # Sum = (n_blocks/2) * block_size * (a + b) = N/2
            # So a + b = 1, and we want to choose a, b to minimize max overlap
            a = 0.7
            b = 0.3
            h = np.tile([a] * block_size + [b] * block_size, n_blocks)
            # Ensure sum is exactly N/2
            current_sum = np.sum(h)
            if abs(current_sum - N/2) < 1e-10:
                score = _compute_max_overlap(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Try a linear ramp pattern
    for slope in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        h = np.linspace(0.5 - slope/2, 0.5 + slope/2, N)
        h = np.clip(h, 0, 1)
        # Adjust to make sum exactly N/2
        diff = N/2 - np.sum(h)
        h += diff / N
        h = np.clip(h, 0, 1)
        # Re-adjust
        diff = N/2 - np.sum(h)
        h += diff / N
        if np.all(h >= -1e-10) and np.all(h <= 1 + 1e-10):
            h = np.clip(h, 0, 1)
            score = _compute_max_overlap(h)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Try a sinusoidal pattern
    for freq in range(1, 11):
        for amp in [0.1, 0.2, 0.3, 0.4, 0.49]:
            h = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
            h = np.clip(h, 0, 1)
            diff = N/2 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0, 1)
            diff = N/2 - np.sum(h)
            h += diff / N
            if np.all(h >= -1e-10) and np.all(h <= 1 + 1e-10):
                h = np.clip(h, 0, 1)
                score = _compute_max_overlap(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Try a two-level pattern with varying proportions
    for level1 in np.arange(0.1, 0.9, 0.05):
        for level2 in np.arange(0.1, 0.9, 0.05):
            if abs(level1 + level2 - 1.0) > 0.01:
                continue
            # Create a pattern with level1 and level2
            h = np.zeros(N)
            h[::2] = level1
            h[1::2] = level2
            diff = N/2 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0, 1)
            diff = N/2 - np.sum(h)
            h += diff / N
            if np.all(h >= -1e-10) and np.all(h <= 1 + 1e-10):
                h = np.clip(h, 0, 1)
                score = _compute_max_overlap(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Try a random initialization with local search
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        h = np.random.uniform(0, 1, N)
        h = (h - h.mean()) * 0.5 + 0.5  # center around 0.5
        h = np.clip(h, 0, 1)
        diff = N/2 - np.sum(h)
        h += diff / N
        h = np.clip(h, 0, 1)
        diff = N/2 - np.sum(h)
        h += diff / N
        if not (np.all(h >= -1e-10) and np.all(h <= 1 + 1e-10)):
            continue
        h = np.clip(h, 0, 1)
        score = _compute_max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Ensure the result is valid
    best_h = np.clip(best_h, 0, 1)
    diff = N/2 - np.sum(best_h)
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
