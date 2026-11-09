<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    selected=set()
    for x in range(1,257):
        if all(not all(x-j*d in selected for j in (1,2,3)) for d in range(1,(x-1)//3+1)):
            selected.add(x)
    return sorted(selected)
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0
    
    def would_create_4ap(s, x):
        for d in range(1, (256 - x) // 3 + 1):
            if (x + d) in s and (x + 2*d) in s and (x + 3*d) in s:
                return True
        for d in range(1, (256 - x) // 2 + 1):
            if x - d >= 1 and (x - d) in s and (x + d) in s and (x + 2*d) in s:
                return True
        for d in range(1, (x - 1) // 2 + 1):
            if (x - 2*d) in s and (x - d) in s and (x + d) in s:
                return True
        for d in range(1, (x - 1) // 3 + 1):
            if (x - 3*d) in s and (x - 2*d) in s and (x - d) in s:
                return True
        return False
    
    def count_3ap(s):
        count = 0
        for a in s:
            for d in range(1, (256 - a) // 2 + 1):
                if (a + d) in s and (a + 2*d) in s:
                    count += 1
        return count
    
    # Try multiple greedy strategies
    best_set = set()
    best_score = 0
    
    # Strategy 1: Natural order greedy
    selected = set()
    for x in range(1, 257):
        if not would_create_4ap(selected, x):
            selected.add(x)
    score = count_3ap(selected)
    if score > best_score:
        best_set = set(selected)
        best_score = score
    
    # Strategy 2: Randomized greedy
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        order = list(range(1, 257))
        random.shuffle(order)
        selected = set()
        for x in order:
            if not would_create_4ap(selected, x):
                selected.add(x)
        score = count_3ap(selected)
        if score > best_score:
            best_set = set(selected)
            best_score = score
    
    # Strategy 3: Local search
    all_elements = set(range(1, 257))
    while time.time() < deadline - 1:
        improved = False
        # Try swaps
        for remove_elem in list(best_set):
            if time.time() > deadline - 1:
                break
            temp_set = best_set - {remove_elem}
            for add_elem in all_elements - temp_set:
                if time.time() > deadline - 1:
                    break
                if would_create_4ap(temp_set, add_elem):
                    continue
                new_set = temp_set | {add_elem}
                new_score = count_3ap(new_set)
                if new_score > best_score:
                    best_set = new_set
                    best_score = new_score
                    improved = True
                    break
            if time.time() > deadline - 1:
                break
        if not improved:
            break
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

This is still potentially too slow. The local search loop with O(n^2) swaps, each requiring O(n) work for would_create_4ap and O(n) for count_3ap, will be very slow for n=256.

Let me think more carefully. The number of elements in the set is probably around 100-150. So the local search has ~120*136 ≈ 16000 swap candidates per iteration, each taking ~200 operations = 3.2M operations per iteration. That's too slow if we do many iterations.

I need a much more efficient approach. Let me:
1. Skip the local search or make it very limited
2. Focus on getting a good initial solution via multiple randomized greedy runs
3. Maybe use a smarter heuristic

Actually, let me just do multiple randomized greedy runs and pick the best. That should be fast enough and might find a better solution than the deterministic greedy.

