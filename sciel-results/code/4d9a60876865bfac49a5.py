"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import linprog

_N = 256
_DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _all_ap_sums(signs):
    """
    Compute all arithmetic progression sums for a given sign vector.
    Returns max absolute sum.
    """
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int32)
    max_disc = 1  # singletons always give |sum| = 1
    
    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            if len(seq) > 1:
                csum = np.cumsum(seq)
                m = np.max(np.abs(csum))
                if m > max_disc:
                    max_disc = m
    return max_disc

def _lp_init(L_min=4, L_max=8):
    """
    Solve LP relaxation: minimize D subject to -D <= sum(signs[AP]) <= D
    for all APs with length in [L_min, L_max].
    Then round to ±1 signs.
    """
    n = _N
    # Collect APs with length in [L_min, L_max]
    ap_indices = []
    for d in range(1, n):
        for r in range(d):
            seq_len = len(range(r, n, d))
            if seq_len < L_min:
                continue
            # All contiguous subsequences of length L_min to L_max
            for length in range(L_min, min(L_max, seq_len) + 1):
                for start in range(seq_len - length + 1):
                    # Convert back to original indices
                    orig_indices = [r + d * (start + i) for i in range(length)]
                    ap_indices.append(orig_indices)
    
    if not ap_indices:
        # Fallback: random signs
        return np.random.choice([-1, 1], size=n)
    
    m = len(ap_indices)
    # Variables: x[0..n-1] are signs, x[n] is D
    # Objective: minimize D
    c = np.zeros(n + 1)
    c[n] = 1.0
    
    # Constraints: -D <= sum(signs[AP]) <= D
    # sum(signs[AP]) - D <= 0
    # -sum(signs[AP]) - D <= 0
    A_ub = np.zeros((2 * m, n + 1))
    b_ub = np.zeros(2 * m)
    
    for i, ap in enumerate(ap_indices):
        for j, idx in enumerate(ap):
            A_ub[2 * i, idx] = 1.0
            A_ub[2 * i + 1, idx] = -1.0
        A_ub[2 * i, n] = -1.0
        A_ub[2 * i + 1, n] = -1.0
    
    # Bounds: -1 <= x_i <= 1 for signs, D >= 0
    bounds = [(-1, 1)] * n + [(0, None)]
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        x = result.x[:n]
        # Round to ±1
        signs = np.sign(x)
        signs[signs == 0] = 1
        return signs
    else:
        # Fallback: random signs
        return np.random.choice([-1, 1], size=n)

def _local_search(signs_list, max_time, rng):
    """
    Local search: flip random positions, accept if it improves max AP discrepancy.
    """
    n = _N
    signs = np.array(signs_list, dtype=np.int32)
    current_disc = _all_ap_sums(signs)
    best_disc = current_disc
    best_signs = signs.copy()
    
    start_time = time.time()
    
    while time.time() - start_time < max_time:
        # Pick a random position to flip
        i = rng.randint(0, n - 1)
        signs[i] = -signs[i]
        
        new_disc = _all_ap_sums(signs)
        
        if new_disc < current_disc:
            current_disc = new_disc
            if current_disc < best_disc:
                best_disc = current_disc
                best_signs = signs.copy()
        else:
            # Revert
            signs[i] = -signs[i]
        
        # Check deadline
        if time.time() > _DEADLINE - 5:
            break
    
    return best_signs.tolist(), best_disc

def construct():
    """
    Construct 256 signs minimizing max |sum| over all arithmetic progressions.
    Strategy: LP-based init + local search.
    """
    start_time = time.time()
    rng = random.Random(42)
    np.random.seed(42)
    
    # Phase 1: LP-based initialization
    lp_signs = _lp_init(L_min=4, L_max=8)
    lp_disc = _all_ap_sums(lp_signs)
    
    # Phase 2: Local search from LP start
    remaining_time = _DEADLINE - 5 - (time.time() - start_time)
    if remaining_time > 1:
        ls_signs, ls_disc = _local_search(lp_signs.tolist(), remaining_time, rng)
        if ls_disc <= lp_disc:
            best_signs = ls_signs
            best_disc = ls_disc
        else:
            best_signs = lp_signs.tolist()
            best_disc = lp_disc
    else:
        best_signs = lp_signs.tolist()
        best_disc = lp_disc
    
    # Check deadline
    if time.time() > _DEADLINE - 2:
        return best_signs
    
    # Phase 3: Try a few random restarts if time permits
    for restart in range(10):
        if time.time() > _DEADLINE - 5:
            break
        
        # Random init
        random_signs = [rng.choice((-1, 1)) for _ in range(_N)]
        
        # Short local search
        short_time = min(2.0, _DEADLINE - 5 - (time.time() - start_time))
        if short_time < 0.5:
            break
        
        ls_signs, ls_disc = _local_search(random_signs, short_time, rng)
        
        if ls_disc < best_disc:
            best_disc = ls_disc
            best_signs = ls_signs
    
    return best_signs
# EVOLVE-BLOCK-END
