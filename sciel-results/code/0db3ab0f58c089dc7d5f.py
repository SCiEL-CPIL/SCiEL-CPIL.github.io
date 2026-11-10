"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256
_DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _all_ap_sums(signs):
    """
    Compute all arithmetic progression sums for a given sign vector.
    Returns a dict mapping (start, step, length) -> sum, but we just need max abs.
    For speed, we compute max |sum| directly.
    """
    n = len(signs)
    signs = np.asarray(signs, dtype=np.int32)
    max_disc = 1  # singletons always give |sum| = 1
    
    # For each step d and residue r mod d, compute prefix sums of signs[r::d]
    # The APs with step d and starting at positions congruent to r mod d
    # are contiguous subsequences of signs[r::d].
    for d in range(1, n):
        for r in range(d):
            seq = signs[r::d]
            if len(seq) > 1:
                csum = np.cumsum(seq)
                m = np.max(np.abs(csum))
                if m > max_disc:
                    max_disc = m
    return max_disc

def _greedy_construct(rng):
    """
    Greedy construction: place signs one by one, choosing the sign that
    minimizes the maximum AP discrepancy so far.
    """
    n = _N
    signs = np.zeros(n, dtype=np.int32)
    
    for i in range(n):
        # Try both +1 and -1
        best_sign = 1
        best_disc = float('inf')
        
        for s in (1, -1):
            signs[i] = s
            # Compute current max AP discrepancy
            # Only need to check APs that include position i
            # For speed, compute full discrepancy (n is small at this point)
            disc = _all_ap_sums(signs)
            if disc < best_disc:
                best_disc = disc
                best_sign = s
        
        signs[i] = best_sign
    
    return signs.tolist()

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
    Strategy: greedy constructive init + local search.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # Phase 1: Greedy construction
    greedy_signs = _greedy_construct(rng)
    greedy_disc = _all_ap_sums(greedy_signs)
    
    # Phase 2: Local search from greedy start
    remaining_time = _DEADLINE - 5 - (time.time() - start_time)
    if remaining_time > 1:
        ls_signs, ls_disc = _local_search(greedy_signs, remaining_time, rng)
        if ls_disc <= greedy_disc:
            greedy_signs = ls_signs
            greedy_disc = ls_disc
    
    # Check deadline
    if time.time() > _DEADLINE - 2:
        return greedy_signs
    
    # Phase 3: Try a few random restarts if time permits
    best_signs = greedy_signs
    best_disc = greedy_disc
    
    for restart in range(10):
        if time.time() > _DEADLINE - 5:
            break
        
        # Random init
        random_signs = [rng.choice((-1, 1)) for _ in range(_N)]
        random_disc = _all_ap_sums(random_signs)
        
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
