"""Direct sign-assignment with efficient AP-sum tracking + simulated annealing."""
import numpy as np
import random
import time

N = 256

def _build_aps():
    """Build list of all arithmetic progressions as arrays of 0-indexed positions."""
    aps = []
    for step in range(1, N + 1):
        for start in range(1, N + 1):
            s0 = start - 1
            length = (N - 1 - s0) // step + 1
            if length >= 1:
                aps.append(np.arange(s0, N, step, dtype=np.int32))
    return aps

_APS = _build_aps()
NUM_APS = len(_APS)
_AP_POSITIONS = np.array(_APS, dtype=object)

# Precompute AP membership: for each position, which APs contain it
_AP_MEMBERSHIP = [[] for _ in range(N)]
for ai, pos in enumerate(_AP_POSITIONS):
    for p in pos:
        _AP_MEMBERSHIP[p].append(ai)

_AP_MEMBERSHIP_ARR = [np.array(m, dtype=np.int32) if m else np.array([], dtype=np.int32) for m in _AP_MEMBERSHIP]

def _compute_all_sums(signs):
    """Compute signed sums for all APs."""
    sums = np.empty(NUM_APS, dtype=np.int32)
    for i, pos in enumerate(_AP_POSITIONS):
        sums[i] = signs[pos].sum()
    return sums

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    rng = random.Random(777)
    
    # Initial: try a few random starts, keep the best
    best_signs = None
    best_score = np.inf
    best_sums = None
    current_signs = None
    current_sums = None
    current_score = np.inf
    
    # Try a few initializations
    for init_seed in range(20):
        if time.time() > deadline:
            break
        rng_init = random.Random(1000 + init_seed)
        signs = np.array([rng_init.choice((-1, 1)) for _ in range(N)], dtype=np.int8)
        sums = _compute_all_sums(signs)
        score = int(np.max(np.abs(sums)))
        if score < best_score:
            best_score = score
            best_signs = signs.copy()
            best_sums = sums.copy()
            current_signs = signs
            current_sums = sums
            current_score = score
    
    T = 10.0
    cooling = 0.99998
    min_T = 0.05
    
    while time.time() < deadline:
        pos = rng.randint(0, N - 1)
        
        ap_idxs = _AP_MEMBERSHIP_ARR[pos]
        if len(ap_idxs) == 0:
            continue
        
        delta = -2 * current_signs[pos]
        old_vals = current_sums[ap_idxs]
        new_vals = old_vals + delta
        
        # Apply flip
        current_signs[pos] = -current_signs[pos]
        current_sums[ap_idxs] = new_vals
        
        new_score = int(np.max(np.abs(current_sums)))
        delta_score = new_score - current_score
        
        if delta_score <= 0 or rng.random() < np.exp(-delta_score / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_signs = current_signs.copy()
                best_sums = current_sums.copy()
        else:
            current_signs[pos] = -current_signs[pos]
            current_sums[ap_idxs] = old_vals
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    return [int(x) for x in best_signs]
