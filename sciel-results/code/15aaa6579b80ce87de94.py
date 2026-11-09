import itertools
import random
import time
import math

N = 16

def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_isosceles(a, b, c):
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _can_add(p, chosen):
    """Check if point p can be added to chosen set without creating an isosceles triangle."""
    for a in chosen:
        for b in chosen:
            if a == b:
                continue
            if _is_isosceles(p, a, b):
                return False
    return True

def _is_maximal(s_set, all_points):
    """Check if s_set is a maximal no-isosceles set."""
    s_list = list(s_set)
    for q in all_points:
        if q in s_set:
            continue
        # Check if q can be added to s_set
        if _can_add(q, s_list):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    all_points = [(x, y) for x in range(N) for y in range(N)]
    n_points = len(all_points)
    
    best_set = None
    best_size = float('inf')
    
    # Strategy: Randomized greedy with local search (remove points)
    num_trials = 2000  # We will break early if time runs out
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        random.seed(trial)
        
        chosen = []
        chosen_set = set()
        
        # Greedy construction: pick random valid points
        while True:
            if time.time() > deadline:
                break
                
            # Find all valid points
            valid_points = []
            for p in all_points:
                if p in chosen_set:
                    continue
                if _can_add(p, chosen):
                    valid_points.append(p)
            
            if not valid_points:
                break
            
            # Pick a random valid point (simple strategy, diversity comes from randomness)
            p = random.choice(valid_points)
            chosen.append(p)
            chosen_set.add(p)
        
        # Local search: try to remove points while maintaining maximality
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            s_copy = chosen.copy()
            s_set = set(s_copy)
            
            # Shuffle order for removal attempts
            random.shuffle(s_copy)
            
            for p in s_copy:
                if time.time() > deadline - 5:
                    break
                # Try removing p
                s_temp_set = s_set - {p}
                s_temp = list(s_temp_set)
                
                # Check if s_temp is maximal
                if _is_maximal(s_temp_set, all_points):
                    # s_temp is a smaller maximal set
                    chosen = s_temp
                    chosen_set = s_temp_set
                    improved = True
                    break # Restart the while loop with the new set
        
        # Update best if improved
        if chosen and len(chosen) < best_size:
            best_size = len(chosen)
            best_set = chosen.copy()
    
    if best_set is None:
        # Fallback: Return a simple greedy set
        chosen = []
        for p in all_points:
            if all(not _is_isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
                chosen.append(p)
        return chosen
        
    return best_set

# EVOLVE-BLOCK-END
