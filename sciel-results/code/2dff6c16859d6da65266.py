"""Low discrepancy on all arithmetic progressions - Vectorized Simulated Annealing."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _get_max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    Returns (max_disc, worst_start, worst_step, worst_len)
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int16)
    
    max_abs = 0
    best_start = 0
    best_step = 1
    best_len = 1
    
    # We iterate through all steps d
    for d in range(1, n + 1):
        # For a given step d, starts are 0..d-1 (but must be < n)
        # We check all valid starts
        for i in range(min(d, n)):
            # Sequence is s[i], s[i+d], s[i+2d], ...
            # s[i:n:d] gives the sequence
            seq = s[i:n:d]
            if len(seq) == 0:
                continue
            
            cumsum = np.cumsum(seq)
            abs_cumsum = np.abs(cumsum)
            current_max = np.max(abs_cumsum)
            
            if current_max > max_abs:
                max_abs = current_max
                best_start = i
                best_step = d
                best_len = np.argmax(abs_cumsum) + 1
    
    return int(max_abs), best_start, best_step, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit to be safe
    
    n = 256
    rng = np.random.default_rng(42)
    
    # Initial random signs
    signs = rng.choice([-1, 1], size=n).astype(np.int16)
    
    current_disc, _, _, _ = _get_max_ap_discrepancy(signs)
    best_signs = signs.copy()
    best_disc = current_disc
    
    # Simulated Annealing Parameters
    # We want to escape local minima. Greedy gets stuck.
    # We will try flipping a single random sign at each step.
    # To speed up, we could batch, but single flip is O(N^2) per eval.
    # With 50s, we can probably do ~10k-50k evals if optimized.
    
    # Let's estimate eval time.
    # N=256. Sum of N/d for d=1..N is N * H_N ~ 256 * 6 ~ 1500 cumsums.
    # Each cumsum is small. This is fast.
    
    T = 10.0
    T_min = 0.1
    alpha = 0.9995
    
    # Pre-compute indices for faster access? 
    # No, just use the vectorized function.
    
    while time.time() < deadline:
        # Pick a random index to flip
        idx = rng.integers(0, n)
        
        # Flip
        signs[idx] = -signs[idx]
        
        # Evaluate
        new_disc, _, _, _ = _get_max_ap_discrepancy(signs)
        
        # Accept/Reject
        if new_disc < current_disc:
            # Accept (improvement)
            current_disc = new_disc
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = signs.copy()
        else:
            # Deterioration. Accept with probability exp(-delta/T)
            delta = new_disc - current_disc
            if delta == 0:
                # Tie, accept to move around
                current_disc = new_disc
            else:
                # Probability of acceptance
                # If T is high, we accept more.
                # Note: delta is integer.
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_disc = new_disc
                else:
                    # Reject: revert
                    signs[idx] = -signs[idx]
        
        # Cool down
        T = max(T * alpha, T_min)
        
        # Optional: If we are stuck for a long time at T_min, we might restart?
        # But SA with low T becomes greedy.
        # Let's add a simple restart if no improvement for a while?
        # Actually, just letting it run is fine.
        
    return best_signs.tolist()

# EVOLVE-BLOCK-END
