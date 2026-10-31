"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def compute_score(x):
    """Compute max|autoconv| / (sum(x))^2."""
    S = np.sum(x)
    if abs(S) < 1e-12:
        return np.inf
    # Full autoconvolution
    C = np.correlate(x, x, mode='full')
    max_abs_C = np.max(np.abs(C))
    return max_abs_C / (S * S)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_x = np.ones(N)
    best_score = compute_score(best_x)
    
    # Try a few alternative patterns
    patterns = []
    
    # All ones (baseline)
    patterns.append(np.ones(N))
    
    # One -1 at various positions
    for pos in [0, N//2, N-1]:
        x = np.ones(N)
        x[pos] = -1
        patterns.append(x)
    
    # Two -1s
    for pos1, pos2 in [(0, N-1), (0, N//2), (N//4, 3*N//4)]:
        x = np.ones(N)
        x[pos1] = -1
        x[pos2] = -1
        patterns.append(x)
    
    # Alternating
    x_alt = np.ones(N)
    x_alt[1::2] = -1
    patterns.append(x_alt)
    
    # Blocks of +1 and -1
    x_block = np.ones(N)
    x_block[N//2:] = -1
    patterns.append(x_block)
    
    for x in patterns:
        if time.time() > deadline:
            break
        s = compute_score(x)
        if s < best_score:
            best_score = s
            best_x = x
    
    return best_x
# EVOLVE-BLOCK-END
