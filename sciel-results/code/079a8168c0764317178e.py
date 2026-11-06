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
    for d in range(1, 256):
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
    
    current_mask = 0
    for v in current:
        current_mask |= (1 << (v - 1))
    
    T = 20.0
    T_min = 0.1
    cooling = 0.999
    
    curr_list = list(current)
    
    while time.time() < deadline:
        idx = rng.randrange(N)
        to_remove = curr_list[idx]
        
        to_add = rng.randint(1, MAXV)
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

def _polish(S, deadline, rng):
    """Final high-precision polish: try all single-swap moves exhaustively."""
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
        curr_list = list(current)
        for i, to_remove in enumerate(curr_list):
            if time.time() >= deadline:
                break
            # Try all possible additions (skip if too many left)
            for to_add in range(1, 513):
                if to_add in current:
                    continue
                new_mask = (current_mask & ~(1 << (to_remove - 1))) | (1 << (to_add - 1))
                new_score = _score_bitset(new_mask)
                if new_score > current_score:
                    current.discard(to_remove)
                    current.add(to_add)
                    curr_list[i] = to_add
                    current_mask = new_mask
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best = set(current)
                    improved = True
                    break  # restart scan after improvement
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    N = 32
    MAXV = 512
    rng = random.Random(42)
    
    # Structured initial sets with many progressions
    best_overall = set(range(1, N + 1))
    best_overall_score = _score_fast(best_overall)
    
    structured_starts = [
        set(range(1, N + 1)),                    # 1..32
        set(range(1, 2*N+1, 2)),                 # evens 2..62
        set(range(2, 2*N+2, 2)),                 # evens 2..64
        set(range(1, N*2+1, 2)),                 # odds 1..63
        set(range(1, 3*N+1, 3)),                 # multiples of 3 up to 96
        set(range(1, 4*N+1, 4)),                 # multiples of 4 up to 128
        set(range(1, 8*N+1, 8)),                 # multiples of 8 up to 256
        set(range(1, 16*N+1, 16)),               # multiples of 16 up to 512
    ]
    
    for start_set in structured_starts:
        if len(start_set) == N:
            s = _score_fast(start_set)
            if s > best_overall_score:
                best_overall_score = s
                best_overall = set(start_set)
    
    # Multiple restarts with local search
    # Use shorter slices to allow more restarts
    slice_duration = 3.0
    
    while time.time() < deadline - 15.0:  # reserve 15s for polish
        # Random initial set
        current = set(rng.sample(range(1, MAXV + 1), N))
        current_score = _score_fast(current)
        
        if current_score > best_overall_score:
            best_overall = set(current)
            best_overall_score = current_score
        
        # Run local search with a time slice
        slice_deadline = min(deadline - 15.0, time.time() + slice_duration)
        best, best_score = _local_search(current, slice_deadline, rng)
        
        if best_score > best_overall_score:
            best_overall = best
            best_overall_score = best_score
    
    # Final polish pass on the best solution found
    if time.time() < deadline - 5.0:
        polish_deadline = deadline - 5.0
        g_best, g_score = _polish(best_overall, polish_deadline, rng)
        if g_score > best_overall_score:
            best_overall = g_best
            best_overall_score = g_score
    
    return sorted(list(best_overall))
