"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def _can_add(selected, x, used_sums):
    """Check if adding x to selected keeps all triple sums unique.
    Returns True if valid, and updates used_sums in place if valid."""
    n = len(selected)
    # Triple sums involving x:
    # (x, x, x)
    # (x, x, y) for y in selected
    # (x, y, z) for y <= z in selected
    new_sums = set()
    
    # (x, x, x)
    s = 3 * x
    if s in used_sums or s in new_sums:
        return False
    new_sums.add(s)
    
    # (x, x, y) for y in selected
    for y in selected:
        s = 2 * x + y
        if s in used_sums or s in new_sums:
            return False
        new_sums.add(s)
    
    # (x, y, z) for y <= z in selected
    for i in range(n):
        y = selected[i]
        for j in range(i, n):
            z = selected[j]
            s = x + y + z
            if s in used_sums or s in new_sums:
                return False
            new_sums.add(s)
    
    # All good, commit
    used_sums.update(new_sums)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    best_selected = []
    best_used_sums = set()
    
    # Try multiple random seeds and strategies
    for seed in range(50):
        if time.time() > deadline:
            break
        
        random.seed(seed)
        
        # Strategy: geometric skeleton + gap filling
        # Start with a sparse geometric progression to guarantee uniqueness
        # Then try to fill in gaps with smaller numbers
        
        # Approach 1: Greedy from small numbers with randomization
        selected = []
        used_sums = set()
        
        # Pre-compute candidates: mix of small numbers and geometric
        # Powers of 4: 1, 4, 16, 64, 256, 1024, 4096 (but 4096 is the max)
        # Actually let's use a denser approach
        
        # Try greedy with random restarts and shuffling of candidate order
        candidates = list(range(1, 4097))
        
        # Shuffle to avoid getting stuck in local optima
        random.shuffle(candidates)
        
        # But we want to bias towards smaller numbers for density
        # Use a weighted approach: try small numbers first with some randomization
        for x in candidates:
            if time.time() > deadline:
                break
            if _can_add(selected, x, used_sums):
                selected.append(x)
                if len(selected) >= 80:
                    break
        
        # Try to improve by local search: swap out larger elements for smaller
        # This is expensive, so only do it if we have time
        if len(selected) < 80 and time.time() < deadline - 5.0:
            # Try replacing the largest element with a smaller unused one
            selected_sorted = sorted(selected)
            for i in range(len(selected_sorted) - 1, -1, -1):
                if time.time() > deadline:
                    break
                if len(selected) >= 80:
                    break
                # Try to find a smaller number that can replace selected_sorted[i]
                # This is complex; skip for now and rely on multiple restarts
        
        if len(selected) > len(best_selected):
            best_selected = list(selected)
            best_used_sums = set(used_sums)
        
        # Early exit if we found 80
        if len(best_selected) >= 80:
            break
    
    # If we didn't find a good solution, fall back to simple greedy
    if len(best_selected) < 10:
        best_selected = []
        best_used_sums = set()
        for x in range(1, 4097):
            if _can_add(best_selected, x, best_used_sums):
                best_selected.append(x)
                if len(best_selected) >= 80:
                    break
    
    return best_selected
# EVOLVE-BLOCK-END
