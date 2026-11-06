# EVOLVE-BLOCK-START
import time
import numpy as np

N = 256
DEADLINE = 82.0  # safety margin before 90s kill

def _precompute_aps():
    """Precompute all arithmetic progressions as index arrays and their lengths."""
    aps = []
    lengths = []
    for a in range(N):
        for d in range(1, N):
            # positions: a, a+d, a+2d, ... < N
            seq = list(range(a, N, d))
            if len(seq) >= 1:
                aps.append(np.array(seq, dtype=np.int32))
                lengths.append(len(seq))
    return aps, lengths

def _max_abs_sum(signs):
    """Compute max |sum| over all APs. signs is a numpy array of ±1."""
    max_val = 0
    for ap in _APS:
        s = signs[ap].sum()
        abs_s = abs(s)
        if abs_s > max_val:
            max_val = abs_s
    return max_val

def _max_abs_sum_fast(signs, prefix_sums):
    """Use prefix sums for speed where possible, but APs with step>1 still need direct sum."""
    # For step=1 (contiguous), we can use prefix sums
    # For step>1, direct sum
    max_val = 0
    for ap, length in zip(_APS, _LENGTHS):
        if length == 1:
            s = 1
        else:
            s = int(signs[ap].sum())
        abs_s = abs(s)
        if abs_s > max_val:
            max_val = abs_s
    return max_val

def _identify_critical_aps(signs, top_k=500):
    """Identify the APs with largest |sum| to focus search."""
    sums = []
    for i, ap in enumerate(_APS):
        s = int(signs[ap].sum())
        sums.append((abs(s), i))
    sums.sort(reverse=True)
    return [idx for _, idx in sums[:top_k]]

def _eval_critical(signs, critical_indices):
    """Evaluate max |sum| only on critical APs."""
    max_val = 0
    for idx in critical_indices:
        s = int(signs[_APS[idx]].sum())
        abs_s = abs(s)
        if abs_s > max_val:
            max_val = abs_s
    return max_val

def _local_search(signs, time_limit, use_critical_first=True):
    """Local search: flip single signs to reduce max |AP sum|."""
    signs = signs.copy()
    start = time.time()
    
    # Phase 1: Critical AP local search
    if use_critical_first:
        critical = _identify_critical_aps(signs, top_k=300)
        best_crit = _eval_critical(signs, critical)
        improved = True
        while improved and (time.time() - start) < time_limit:
            improved = False
            for i in range(N):
                if (time.time() - start) > time_limit:
                    break
                signs[i] *= -1
                crit_val = _eval_critical(signs, critical)
                if crit_val < best_crit:
                    best_crit = crit_val
                    improved = True
                else:
                    signs[i] *= -1
            # Refresh critical set occasionally
            if improved and (time.time() - start) < time_limit:
                critical = _identify_critical_aps(signs, top_k=300)
    
    # Phase 2: Full local search
    best_full = _max_abs_sum_fast(signs, None)
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for i in range(N):
            if (time.time() - start) > time_limit:
                break
            signs[i] *= -1
            val = _max_abs_sum_fast(signs, None)
            if val < best_full:
                best_full = val
                improved = True
            else:
                signs[i] *= -1
    
    return signs, best_full

def _spectral_init():
    """
    Spectral approximation: model sign sequence as vector, minimize sum of squares
    of AP sums via subgradient descent on smooth surrogate, then project to ±1.
    """
    rng = np.random.RandomState(42)
    
    # Start with a random ±1 vector
    x = rng.choice([-1.0, 1.0], size=N)
    
    # We'll do a few iterations of: 
    # 1. Compute gradient of sum(s_i^2) where s_i = sum of x over AP i
    # 2. Update x in direction of -gradient
    # 3. Project to ±1
    
    # For efficiency, use only a subset of APs (the "heavy" ones - longer APs)
    # Longer APs have larger potential sums
    
    # Build incidence: for each position j, which APs include it
    # grad[j] = 2 * sum_{i: j in AP_i} s_i
    
    # Use all APs but compute efficiently
    # s_i = sum_{j in AP_i} x[j]
    # grad[j] = 2 * sum_{i: j in AP_i} s_i
    
    # Precompute for each position j, the list of AP indices that contain j
    pos_to_aps = [[] for _ in range(N)]
    for i, ap in enumerate(_APS):
        for j in ap:
            pos_to_aps[j].append(i)
    
    # Subgradient descent
    lr = 0.1
    for it in range(200):
        # Compute all AP sums
        sums = np.array([x[ap].sum() for ap in _APS])
        
        # Compute gradient
        grad = np.zeros(N)
        for j in range(N):
            for i in pos_to_aps[j]:
                grad[j] += sums[i]
        grad *= 2.0
        
        # Update
        x -= lr * grad
        # Project to ±1
        x = np.where(x >= 0, 1.0, -1.0)
    
    return x.astype(np.int8)

def construct():
    global _APS, _LENGTHS
    
    t0 = time.time()
    
    # Precompute APs
    _APS, _LENGTHS = _precompute_aps()
    
    best_signs = None
    best_val = float('inf')
    
    # Try multiple initializations
    n_trials = 0
    while (time.time() - t0) < (DEADLINE - 5.0):
        n_trials += 1
        
        if n_trials == 1:
            # Spectral init
            signs = _spectral_init()
        else:
            # Random init with slight bias
            rng = np.random.RandomState(n_trials * 137)
            signs = rng.choice([-1, 1], size=N).astype(np.int8)
        
        # Local search
        time_left = DEADLINE - 5.0 - (time.time() - t0)
        if time_left < 2.0:
            break
        
        # Use critical-first local search for speed
        signs_ls, val = _local_search(signs, time_limit=time_left * 0.6, use_critical_first=True)
        
        if val < best_val:
            best_val = val
            best_signs = signs_ls.copy()
        
        # If we have time, do a final polish with full local search
        time_left = DEADLINE - 2.0 - (time.time() - t0)
        if time_left > 1.0 and best_signs is not None:
            best_signs, best_val = _local_search(best_signs, time_limit=time_left, use_critical_first=False)
    
    if best_signs is None:
        # Fallback: random
        rng = np.random.RandomState(171)
        best_signs = rng.choice([-1, 1], size=N).astype(np.int8)
    
    return [int(s) for s in best_signs]

# EVOLVE-BLOCK-END
