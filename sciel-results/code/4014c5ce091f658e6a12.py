import time
import random
import numpy as np

def _score_bitset(mask):
    """
    Count distinct d > 0 such that exists a with a, a+d, a+2d all in the set.
    mask: int where bit i is set iff value (i+1) is in the set.
    Uses bit-parallel operations for speed.
    """
    diffs = 0
    # d can range from 1 to 255 (since max value is 512, a+2d <= 512 implies 2d < 512)
    for d in range(1, 256):
        # We need to check if there exists a such that bit a is set, bit a+d is set, bit a+2d is set.
        # Let M be the mask.
        # M & (M >> d) & (M >> 2d) will have bit a set if a, a+d, a+2d are all in the set.
        # We just need to check if this result is non-zero.
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            diffs += 1
    return diffs

def _score_fast(S):
    """Score a set of 32 integers using bitset for fast membership."""
    mask = 0
    for v in S:
        mask |= (1 << (v - 1))
    return _score_bitset(mask)

def _local_search(current, deadline, rng):
    """Perform local search with simulated annealing using fast bitset scoring."""
    N = 32
    MAXV = 512
    current_score = _score_fast(current)
    best = set(current)
    best_score = current_score
    
    # Precompute the mask for fast updates
    current_mask = 0
    for v in current:
        current_mask |= (1 << (v - 1))
    
    T = 15.0
    T_min = 0.1
    cooling = 0.9995
    
    # Precompute list of current elements for fast choice
    # We will maintain a list parallel to the set for O(1) removal/insertion if possible,
    # but since N is small, list(current) is acceptable, or we can just use rng.choice on a list we maintain.
    curr_list = list(current)
    
    while time.time() < deadline:
        # Generate a move: swap one element
        idx = rng.randrange(N)
        to_remove = curr_list[idx]
        
        # Choose to_add from values not in current
        to_add = rng.randint(1, MAXV)
        # Simple retry loop to avoid collision
        while to_add in current:
            to_add = rng.randint(1, MAXV)
        
        new_mask = (current_mask & ~(1 << (to_remove - 1))) | (1 << (to_add - 1))
        new_score = _score_bitset(new_mask)
        
        if new_score > current_score:
            current.discard(to_remove)
            current.add(to_add)
            curr_list[idx] = to_add
            current_mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = set(current)
        else:
            delta = new_score - current_score
            if delta < 0 and rng.random() < np.exp(delta / T):
                current.discard(to_remove)
                current.add(to_add)
                curr_list[idx] = to_add
                current_mask = new_mask
                current_score = new_score
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best, best_score

def _greedy_improve(S, deadline, rng):
    """Try greedy swaps that improve the score."""
    current = set(S)
    current_score = _score_fast(current)
    best = set(current)
    best_score = current_score
    
    current_mask = 0
    for v in current:
        current_mask |= (1 << (v - 1))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for to_remove in list(current):
            if time.time() >= deadline:
                break
            # Try a few random additions
            for _ in range(10):
                to_add = rng.randint(1, 512)
                if to_add in current:
                    continue
                new_mask = current_mask & ~(1 << (to_remove - 1)) | (1 << (to_add - 1))
                new_score = _score_bitset(new_mask)
                if new_score > current_score:
                    current.discard(to_remove)
                    current.add(to_add)
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = set(current)
                    improved = True
                    break
    return best, best_score

def construct():
    start_time = time.time()
    # Set a conservative deadline to ensure we finish well within 120s
    deadline = start_time + 110.0
    
    N = 32
    MAXV = 512
    rng = random.Random(42)
    
    # Start with a structured initial set that has many progressions
    # The set {1,2,...,32} has many 3-term progressions
    best_overall = set(range(1, N + 1))
    best_overall_score = _score_fast(best_overall)
    
    # Also try some other structured starts
    structured_starts = [
        set(range(1, N + 1)),           # consecutive
        set(range(1, 2*N+1, 2)),        # evens up to 62
        set(range(2, 2*N+2, 2)),        # evens 2..64
        set(range(1, N*2+1, 2)),        # odds 1..63
    ]
    
    for start_set in structured_starts:
        if len(start_set) == N:
            s = _score_fast(start_set)
            if s > best_overall_score:
                best_overall_score = s
                best_overall = set(start_set)
    
    # Multiple restarts with local search
    # We use shorter slices to allow more restarts if the search is fast
    slice_duration = 2.0
    
    while time.time() < deadline:
        # Random initial set
        current = set(rng.sample(range(1, MAXV + 1), N))
        current_score = _score_fast(current)
        
        if current_score > best_overall_score:
            best_overall = set(current)
            best_overall_score = current_score
        
        # Run local search with a time slice
        slice_deadline = min(deadline, time.time() + slice_duration)
        best, best_score = _local_search(current, slice_deadline, rng)
        
        if best_score > best_overall_score:
            best_overall = best
            best_overall_score = best_score
        
        # If we have plenty of time left, do a greedy improvement pass on the best
        if time.time() < deadline - 10.0:
            g_best, g_score = _greedy_improve(best_overall, deadline - 5.0, rng)
            if g_score > best_overall_score:
                best_overall = g_best
                best_overall_score = g_score
    
    return sorted(list(best_overall))
