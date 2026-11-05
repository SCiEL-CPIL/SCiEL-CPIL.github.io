import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Keep 5s margin for safety
    
    best_set = None
    best_size = float('inf')
    
    # 1. Try deterministic strategies first
    # Ascending greedy (original)
    set_asc = _greedy_order(range(1, N + 1))
    if len(set_asc) < best_size:
        best_size = len(set_asc)
        best_set = set_asc
        
    # Descending greedy
    set_desc = _greedy_order(range(N, 0, -1))
    if len(set_desc) < best_size:
        best_size = len(set_desc)
        best_set = set_desc

    # 2. Randomized Greedy with restarts
    # We will run this until the deadline
    while time.time() < deadline:
        # Random permutation of candidates?
        # Or just random choice from valid candidates at each step?
        # Random choice from valid candidates is better because it adapts to the current state.
        current_set = _randomized_greedy()
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set
            # If we find a very small set, we might want to continue to see if we can do better
            # Lower bound is ~sqrt(N) = 16.
            # If we hit 16, we can stop.
            if best_size <= 16:
                break

    return best_set

def _greedy_order(order):
    selected = []
    used_sums = set()
    for x in order:
        # Check if x can be added
        # Sums: 2x and x+y for all y in selected
        new_sums = [2 * x]
        valid = True
        for y in selected:
            s = x + y
            if s in used_sums:
                valid = False
                break
            new_sums.append(s)
        
        if valid:
            # Check 2x
            if 2 * x in used_sums:
                valid = False
            
        if valid:
            selected.append(x)
            used_sums.update(new_sums)
            
    return selected

def _randomized_greedy():
    selected = []
    used_sums = set()
    available = list(range(1, N + 1))
    
    while available:
        valid_candidates = []
        for x in available:
            # Quick check: 2x not in used_sums
            if 2 * x in used_sums:
                continue
            
            # Check x+y for all y in selected
            ok = True
            for y in selected:
                if x + y in used_sums:
                    ok = False
                    break
            
            if ok:
                valid_candidates.append(x)
        
        if not valid_candidates:
            break
        
        # Pick a random valid candidate
        # Heuristic: Maybe pick one that is "central"? 
        # For now, uniform random.
        x = random.choice(valid_candidates)
        selected.append(x)
        
        # Update used_sums
        used_sums.add(2 * x)
        for y in selected[:-1]:
            used_sums.add(x + y)
        
        # Remove x from available
        # Linear search is O(N), but N is small.
        # Alternatively, we can just leave it and skip it in next iteration?
        # No, we must remove it to avoid re-checking.
        # Using list remove is O(N).
        available.remove(x)
        
    return selected
# EVOLVE-BLOCK-END
