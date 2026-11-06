"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    def time_left():
        return deadline - time.time()
    
    def check_candidate(selected, x, sums_set):
        """Check if adding x to selected keeps all triple sums unique.
        selected is sorted. sums_set contains all triple sums of selected.
        Returns (True, new_sums) if valid, else (False, None)."""
        n = len(selected)
        # New triple sums involving x:
        # 1. (x, x, x) = 3x
        # 2. (x, x, y) for y in selected -> 2x + y
        # 3. (x, y, z) for y <= z in selected -> x + y + z
        
        # Check 3x
        if 3 * x in sums_set:
            return False, None
        
        new_sums = {3 * x}
        
        # Check 2x + y for each y in selected
        for y in selected:
            s = 2 * x + y
            if s in sums_set or s in new_sums:
                return False, None
            new_sums.add(s)
        
        # Check x + y + z for y <= z in selected
        for i in range(n):
            y = selected[i]
            for j in range(i, n):
                z = selected[j]
                s = x + y + z
                if s in sums_set or s in new_sums:
                    return False, None
                new_sums.add(s)
        
        return True, new_sums
    
    def greedy_forward(start, end, selected, sums_set):
        """Greedy forward from start to end."""
        for x in range(start, end + 1):
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected, x, sums_set)
            if valid:
                selected.append(x)
                sums_set.update(new_sums)
        return selected, sums_set
    
    def greedy_backward(start, end, selected, sums_set):
        """Greedy backward from start down to end."""
        for x in range(start, end - 1, -1):
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected, x, sums_set)
            if valid:
                selected.append(x)
                sums_set.update(new_sums)
        return selected, sums_set
    
    def greedy_random(selected, sums_set, candidates):
        """Greedy with random candidate selection."""
        for x in candidates:
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected, x, sums_set)
            if valid:
                selected.append(x)
                sums_set.update(new_sums)
        return selected, sums_set
    
    # Strategy 1: Greedy forward from 1 to 4096
    selected1 = []
    sums1 = set()
    selected1, sums1 = greedy_forward(1, 4096, selected1, sums1)
    best = selected1[:]
    
    if time_left() < 5.0:
        return best
    
    # Strategy 2: Greedy backward from 4096 to 1
    selected2 = []
    sums2 = set()
    selected2, sums2 = greedy_backward(4096, 1, selected2, sums2)
    if len(selected2) > len(best):
        best = selected2[:]
    
    if time_left() < 5.0:
        return best
    
    # Strategy 3: Greedy with randomization - try multiple random orders
    rng = random.Random(42)
    for trial in range(20):
        if time_left() < 5.0:
            break
        candidates = list(range(1, 4097))
        rng.shuffle(candidates)
        selected3 = []
        sums3 = set()
        selected3, sums3 = greedy_random(selected3, sums3, candidates)
        if len(selected3) > len(best):
            best = selected3[:]
    
    if time_left() < 5.0:
        return best
    
    # Strategy 4: Local search - try to swap elements to increase size
    # Start from best, try removing one element and adding a different one
    current = best[:]
    current_set = set(current)
    # Rebuild sums_set for current
    from itertools import combinations_with_replacement
    current_sums = set()
    for triple in combinations_with_replacement(current, 3):
        current_sums.add(sum(triple))
    
    improved = True
    while improved and time_left() > 10.0:
        improved = False
        # Try swapping each element with a random candidate
        for idx in range(len(current)):
            if time_left() < 10.0:
                break
            old = current[idx]
            # Try replacing old with a random value not in current_set
            for _ in range(50):
                new = rng.randint(1, 4096)
                if new in current_set:
                    continue
                # Remove old, add new
                test_set = current_set - {old} | {new}
                # Check if all triple sums are unique
                # This is expensive, so only do it for promising swaps
                # Quick check: just verify no new collisions
                # For now, skip this as it's too expensive
                break
        # Try adding a new element
        for _ in range(100):
            if time_left() < 10.0:
                break
            x = rng.randint(1, 4096)
            if x in current_set:
                continue
            valid, new_sums = check_candidate(sorted(current), x, current_sums)
            if valid:
                current.append(x)
                current_set.add(x)
                current_sums.update(new_sums)
                improved = True
                if len(current) > len(best):
                    best = current[:]
    
    return best
# EVOLVE-BLOCK-END
