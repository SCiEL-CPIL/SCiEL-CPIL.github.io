import time
import random
import itertools

def is_sidon(subset):
    # Check if subset has all unordered pair sums unique INCLUDING diagonals
    # i.e., a[i] + a[j] == a[k] + a[l] implies {i,j} == {k,l}
    # Since we iterate i <= j, we just need to check if any sum repeats.
    s = sorted(subset)
    n = len(s)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def max_sidon_size(S):
    # Find largest subset of S that is Sidon
    # |S| = 12. C(12, k) is small.
    # k=12: 1
    # k=11: 12
    # k=10: 66
    # k=9: 220
    # k=8: 495
    # k=7: 792
    # k=6: 924
    # Total ~ 2500 checks. Each check is O(k^2) ~ 144 ops. Total ~ 360k ops. Very fast.
    for k in range(len(S), 0, -1):
        for comb in itertools.combinations(S, k):
            if is_sidon(list(comb)):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    # Start with contiguous
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    # Try simple variations
    candidates = [
        list(range(12)),
        list(range(11)) + [12],
        list(range(10)) + [11, 12],
        # A set with more collisions?
        # e.g. 0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14
    ]
    
    current_best = list(range(12))
    current_best_val = max_sidon_size(current_best)
    
    for c in candidates:
        if time.time() > deadline: break
        val = max_sidon_size(c)
        if val < current_best_val:
            current_best_val = val
            current_best = c
            
    # Local search: Swap elements to try to reduce max Sidon size
    # We want to minimize the value.
    # If we swap an element with a value that creates more collisions,
    # the max Sidon size might drop.
    
    if time.time() < deadline - 5:
        # Random restarts / hill climbing
        for _ in range(500):
            if time.time() > deadline: break
            
            # Pick a random element to change
            idx = random.randint(0, 11)
            old_val = current_best[idx]
            
            # Try a few random replacements
            for _ in range(20):
                if time.time() > deadline: break
                new_val = random.randint(0, 128)
                if new_val in current_best and new_val != old_val:
                    continue
                if new_val == old_val:
                    continue
                
                candidate = current_best.copy()
                candidate[idx] = new_val
                candidate = sorted(candidate)
                
                # Quick check: if the set is not distinct, skip
                if len(set(candidate)) != 12:
                    continue
                    
                val = max_sidon_size(candidate)
                if val < current_best_val:
                    current_best_val = val
                    current_best = candidate
                    # Accept and continue hill climbing from here
                    break # Move to next iteration of outer loop to try another swap?
                    # Or continue trying other swaps on this new best?
                    # Let's accept and restart the inner loop logic implicitly by 
                    # just continuing the outer loop which picks a new idx.
                    
    return current_best
