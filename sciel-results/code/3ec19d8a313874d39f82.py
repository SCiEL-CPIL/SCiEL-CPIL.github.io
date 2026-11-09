"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_overlap(h, N, shifts=None):
    """Compute overlap at given shifts (in step units). Returns max overlap."""
    if shifts is None:
        shifts = np.arange(N)
    # h has N steps on [0,2], each width 2/N
    # overlap(k) = (2/N) * sum_i h[i] * (1 - h[(i+k) % N])
    max_ov = 0.0
    for k in shifts:
        k = int(k) % N
        ov = np.sum(h * (1.0 - np.roll(h, -k))) * (2.0 / N)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _compute_overlap(best_h, N)
    
    # Try various patterns
    patterns = []
    
    # Sinusoidal: h[i] = 0.5 + a*sin(2*pi*m*i/N)
    for m in range(1, 20):
        for a in [0.1, 0.2, 0.3, 0.4, 0.49]:
            h = 0.5 + a * np.sin(2 * np.pi * m * np.arange(N) / N)
            h = np.clip(h, 0, 1)
            # Adjust to ensure sum is exactly N/2
            diff = N / 2.0 - np.sum(h)
            h = h + diff / N
            h = np.clip(h, 0, 1)
            # Re-adjust
            diff2 = N / 2.0 - np.sum(h)
            if abs(diff2) > 1e-10:
                # Distribute the difference
                idx = np.argsort(h)[::-1] if diff2 < 0 else np.argsort(h)
                step = diff2 / N
                h += step
                h = np.clip(h, 0, 1)
            score = _compute_overlap(h, N)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Try cosine patterns
    for m in range(1, 20):
        for a in [0.1, 0.2, 0.3, 0.4, 0.49]:
            h = 0.5 + a * np.cos(2 * np.pi * m * np.arange(N) / N)
            h = np.clip(h, 0, 1)
            diff = N / 2.0 - np.sum(h)
            h = h + diff / N
            h = np.clip(h, 0, 1)
            diff2 = N / 2.0 - np.sum(h)
            if abs(diff2) > 1e-10:
                step = diff2 / N
                h += step
                h = np.clip(h, 0, 1)
            score = _compute_overlap(h, N)
            if score < best_score:
                best_score = score
                best_h = h.copy()
    
    # Try binary patterns with different periods
    for period in range(2, 50):
        if N % period != 0:
            continue
        n_ones = period // 2
        if n_ones == 0:
            continue
        # Try all binary patterns of given period with n_ones ones
        # This is expensive, so just try a few structured ones
        # Pattern: first n_ones are 1, rest are 0
        base = np.zeros(period)
        base[:n_ones] = 1.0
        h = np.tile(base, N // period)
        score = _compute_overlap(h, N)
        if score < best_score:
            best_score = score
            best_h = h.copy()
        
        # Alternating within period
        if period >= 2:
            base2 = np.zeros(period)
            for i in range(n_ones):
                base2[2 * i] = 1.0
            if np.sum(base2) == n_ones:
                h = np.tile(base2, N // period)
                score = _compute_overlap(h, N)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()
    
    # Check time
    if time.time() > deadline:
        return best_h
    
    # Try random binary patterns
    np.random.seed(42)
    n_ones = N // 2
    for trial in range(200):
        if time.time() > deadline:
            break
        idx = np.random.choice(N, n_ones, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        score = _compute_overlap(h, N)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    return best_h
# EVOLVE-BLOCK-END
