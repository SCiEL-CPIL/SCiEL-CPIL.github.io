import time
import numpy as np
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    N = 4096
    MAX_K = 80
    MAX_SUM = 3 * N  # 12288
    
    # Use a numpy boolean array for fast membership checks
    used_sums = np.zeros(MAX_SUM + 1, dtype=bool)
    
    def can_add(x, selected, used_sums):
        """Check if x can be added to selected without creating duplicate triple sums."""
        n = len(selected)
        # Check (x, x, x)
        s = 3 * x
        if s > MAX_SUM or used_sums[s]:
            return False
        # Check (x, x, y) for y in selected
        for y in selected:
            s = 2 * x + y
            if s > MAX_SUM or used_sums[s]:
                return False
        # Check (x, y, z) for y <= z in selected
        for i in range(n):
            y = selected[i]
            for j in range(i, n):
                z = selected[j]
                s = x + y + z
                if s > MAX_SUM or used_sums[s]:
                    return False
        return True
    
    def add_element(x, selected, used_sums):
        """Add x to selected and update used_sums."""
        selected.append(x)
        selected.sort()
        n = len(selected)
        used_sums[3 * x] = True
        for y in selected:
            if y != x:
                used_sums[2 * x + y] = True
        for i in range(n):
            y = selected[i]
            for j in range(i, n):
                z = selected[j]
                used_sums[x + y + z] = True
    
    def greedy_build(max_x):
        """Greedy build from 1 to max_x."""
        selected = []
        used_sums_local = np.zeros(MAX_SUM + 1, dtype=bool)
        for x in range(1, max_x + 1):
            if time.time() > deadline:
                break
            if len(selected) >= MAX_K:
                break
            if can_add(x, selected, used_sums_local):
                add_element(x, selected, used_sums_local)
        return selected, used_sums_local
    
    def randomized_greedy(max_x, seed):
        """Randomized greedy with shuffled candidates."""
        selected = []
        used_sums_local = np.zeros(MAX_SUM + 1, dtype=bool)
        rng = random.Random(seed)
        candidates = list(range(1, max_x + 1))
        rng.shuffle(candidates)
        
        for x in candidates:
            if len(selected) >= MAX_K:
                break
            if time.time() > deadline:
                break
            if can_add(x, selected, used_sums_local):
                add_element(x, selected, used_sums_local)
        return selected, used_sums_local
    
    def try_extend(selected, used_sums_local):
        """Try to extend a set by adding more elements."""
        selected = list(selected)
        used_sums_local = used_sums_local.copy()
        added = True
        while added and len(selected) < MAX_K:
            added = False
            for x in range(1, N + 1):
                if time.time() > deadline:
                    break
                if x in selected:
                    continue
                if can_add(x, selected, used_sums_local):
                    add_element(x, selected, used_sums_local)
                    added = True
                    if len(selected) >= MAX_K:
                        break
        return selected, used_sums_local
    
    # Initial greedy build
    selected, used_sums = greedy_build(513)
    best_k = len(selected)
    
    # Try many randomized attempts with different seeds and ranges
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        
        # Vary the max_x for diversity
        if attempt % 10 == 1:
            max_x = N
        elif attempt % 10 == 2:
            max_x = 2048
        elif attempt % 10 == 3:
            max_x = 1024
        elif attempt % 10 == 4:
            max_x = 512
        elif attempt % 10 == 5:
            max_x = 3000
        elif attempt % 10 == 6:
            max_x = 1500
        elif attempt % 10 == 7:
            max_x = 700
        elif attempt % 10 == 8:
            max_x = 800
        elif attempt % 10 == 9:
            max_x = 1200
        else:
            max_x = 400
        
        max_x = min(max_x, N)
        
        selected_new, used_sums_new = randomized_greedy(max_x, seed=attempt)
        
        if len(selected_new) > best_k:
            best_k = len(selected_new)
            selected = selected_new
            used_sums = used_sums_new
            # Try to extend
            if best_k < MAX_K:
                selected, used_sums = try_extend(selected, used_sums)
                best_k = len(selected)
    
    # Final check: ensure all elements are <= 4096 and distinct
    selected = sorted(set([x for x in selected if 1 <= x <= N]))
    
    # Verify the result is valid
    sums = [sum(t) for t in combinations_with_replacement(selected, 3)]
    if len(sums) != len(set(sums)):
        # Should not happen, but if it does, fall back to greedy
        selected, _ = greedy_build(513)
    
    return selected
