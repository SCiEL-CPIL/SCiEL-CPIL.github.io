"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _count_progression_diffs(S):
    """Count distinct d > 0 such that there exists a with a, a+d, a+2d in S."""
    S_set = set(S)
    diffs = set()
    for a in S:
        for b in S:
            d = b - a
            if d > 0 and (a + 2 * d) in S_set:
                diffs.add(d)
    return len(diffs)

def _local_search(S, time_limit=140.0):
    """
    Local search: try swapping one element of S with a new element from 1..512.
    Return best S found.
    """
    S = list(S)
    S_set = set(S)
    best_score = _count_progression_diffs(S)
    best_S = list(S)
    
    start_time = time.time()
    deadline = start_time + time_limit
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        if no_improve_count >= max_no_improve:
            break
        
        # Try random swaps
        i = random.randint(0, len(S) - 1)
        old_val = S[i]
        
        # Generate candidate new values
        candidates = []
        # Values that might create new progressions
        for a in S:
            if a != old_val:
                # If we add x, check if it completes a progression with existing pairs
                # x = a + d where a+d is in S, or x = a + 2d where a+d is in S
                pass
        
        # Simpler: just try random values
        for _ in range(50):
            new_val = random.randint(1, 512)
            if new_val in S_set or new_val == old_val:
                continue
            
            # Test the swap
            S_set.remove(old_val)
            S_set.add(new_val)
            
            score = _count_progression_diffs(S_set)
            
            S_set.remove(new_val)
            S_set.add(old_val)
            
            if score > best_score:
                best_score = score
                best_S = S_set.copy()
                S_set = S_set.copy()
                no_improve_count = 0
                break
            else:
                no_improve_count += 1
    
    return list(best_S)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with a structured initial set
    # Try different initial patterns
    best_S = list(range(1, 33))
    best_score = _count_progression_diffs(best_S)
    
    # Try several random restarts
    random.seed(42)
    
    while time.time() < deadline:
        # Generate a random initial set
        S = random.sample(range(1, 513), 32)
        S_set = set(S)
        score = _count_progression_diffs(S_set)
        
        if score > best_score:
            best_score = score
            best_S = list(S_set)
        
        # Run local search from this starting point
        time_remaining = deadline - time.time() - 5.0  # 5s safety margin
        if time_remaining > 10.0:
            S_result = _local_search(list(S_set), time_limit=min(time_remaining, 30.0))
            score_result = _count_progression_diffs(S_result)
            if score_result > best_score:
                best_score = score_result
                best_S = S_result
    
    return best_S
# EVOLVE-BLOCK-END
