"""Low discrepancy on all arithmetic progressions - Simulated Annealing with Multi-bit Flips."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _get_max_ap_discrepancy(signs):
    """
    Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list or array of +1/-1 indexed 0..n-1.
    Returns (max_abs, d, i).
    """
    n = len(signs)
    s = np.asarray(signs, dtype=np.int16)
    
    max_abs = 0
    best_d = 1
    best_i = 0
    
    for d in range(1, n + 1):
        # For a given d, iterate over all starting indices i in [0, d)
        # such that i < n.
        for i in range(min(d, n)):
            seq = s[i:n:d]
            if len(seq) == 0:
                continue
            cumsum = np.cumsum(seq)
            current_max = np.max(np.abs(cumsum))
            if current_max > max_abs:
                max_abs = current_max
                best_d = d
                best_i = i
    return int(max_abs), best_d, best_i

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    rng = np.random.default_rng(42)
    
    # Initial random signs
    signs = rng.choice([-1, 1], size=n).astype(np.int16)
    
    # Evaluate initial
    current_disc, _, _ = _get_max_ap_discrepancy(signs)
    best_signs = signs.copy()
    best_disc = current_disc
    
    # Simulated Annealing Parameters
    T_start = 1.0
    T_end = 0.05
    cooling_rate = 0.999
    
    # We will run SA for a fixed number of steps or until time limit
    # Estimate steps: ~5-10ms per eval. 85s / 10ms = 8500 steps.
    # Let's aim for ~5000-10000 steps depending on speed.
    
    T = T_start
    step = 0
    
    while time.time() < deadline:
        # Determine number of flips (multi-bit flip to escape local minima)
        # Early: more flips (exploration). Late: fewer flips (exploitation).
        # Probability of k flips:
        # k=1: always possible
        # k=2: common
        # k=3: less common
        # k=4: rare
        
        # Simple heuristic:
        # 70% chance 1 flip, 20% chance 2 flips, 10% chance 3 flips
        r = rng.random()
        if r < 0.7:
            num_flips = 1
        elif r < 0.9:
            num_flips = 2
        else:
            num_flips = 3
        
        # Select indices to flip
        flip_indices = rng.choice(n, size=num_flips, replace=False)
        
        # Apply flips
        old_signs = signs[flip_indices].copy()
        signs[flip_indices] = -signs[flip_indices]
        
        # Evaluate
        new_disc, _, _ = _get_max_ap_discrepancy(signs)
        
        # Acceptance criterion
        if new_disc < current_disc:
            # Always accept improvement
            current_disc = new_disc
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
        else:
            # Accept with probability exp(-(new_disc - current_disc) / T)
            delta = new_disc - current_disc
            prob = np.exp(-delta / T)
            if rng.random() < prob:
                current_disc = new_disc
            else:
                # Revert
                signs[flip_indices] = old_signs
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
        step += 1
        
        # Optional: Early stop if we reach a very low discrepancy?
        # Theoretical lower bound is not 0, but let's see.
        # If best_disc <= 2, we might be close to optimal.
        # But let's keep running to be sure, or use remaining time for restarts?
        # Given the "greedy" nature of the previous score, SA should help.
        
    return best_signs.tolist()

# EVOLVE-BLOCK-END
