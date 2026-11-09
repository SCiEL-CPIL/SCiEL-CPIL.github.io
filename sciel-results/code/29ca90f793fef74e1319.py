"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _max_sidon_size(arr):
    """Return the largest subset size with all unordered pair sums unique (including diagonals).
    
    A subset is Sidon (with diagonals) if for all distinct pairs (a,b) and (c,d) in the subset
    (where a<=b, c<=d), we have a+b != c+d. This includes the case where pairs share an element.
    """
    n = len(arr)
    # For each element, we want to find the maximum subset size
    # We can use a greedy/backtracking approach for small sets
    
    # First, let's try to find if there's a Sidon subset of size k
    # Start from large k and go down
    
    def is_sidon_subset(subset):
        """Check if a subset is Sidon (all pair sums unique)."""
        if len(subset) <= 2:
            return True
        sums = set()
        for i in range(len(subset)):
            for j in range(i, len(subset)):
                s = subset[i] + subset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    # Try decreasing sizes from n down to 1
    for size in range(n, 0, -1):
        # Check all combinations of this size
        for combo in combinations(range(n), size):
            subset = [arr[i] for i in combo]
            if is_sidon_subset(subset):
                return size
    return 1

def _fast_max_sidon_size(arr):
    """Faster version using early termination and heuristics."""
    n = len(arr)
    if n <= 2:
        return n
    
    # Quick check: is the whole set Sidon?
    if is_sidon_subset(arr):
        return n
    
    # Try size n-1
    for i in range(n):
        subset = arr[:i] + arr[i+1:]
        if is_sidon_subset(subset):
            return n - 1
    
    # Try size n-2
    for i in range(n):
        for j in range(i+1, n):
            subset = arr[:i] + arr[i+1:j] + arr[j+1:]
            if is_sidon_subset(subset):
                return n - 2
    
    # For smaller sizes, do a more systematic search
    # Try size 5 (which is likely the minimum for 12 elements)
    for combo in combinations(range(n), 5):
        subset = [arr[i] for i in combo]
        if is_sidon_subset(subset):
            return 5
    
    return 4  # Shouldn't happen, but safe fallback

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Start with a reasonable initial set
    # We want a set where the maximum Sidon subset is as small as possible
    # The minimum possible is at least 5 (since any 12-element set in [0,128] 
    # likely has a Sidon subset of size >= 5)
    
    best_arr = list(range(12))
    best_score = _fast_max_sidon_size(best_arr)
    
    # Local search: try to minimize the maximum Sidon subset size
    random.seed(42)
    
    while time.time() < deadline:
        attempts = 0
        max_attempts = 50000
        while time.time() < deadline and attempts < max_attempts:
            attempts += 1
            
            # Generate a random 12-element set
            arr = random.sample(range(129), 12)
            arr.sort()
            
            score = _fast_max_sidon_size(arr)
            if score < best_score:
                best_score = score
                best_arr = arr[:]
                if best_score <= 4:
                    break
            
            # Local search: try swapping one element
            if score <= best_score + 1:
                for i in range(12):
                    if time.time() > deadline:
                        break
                    old_val = arr[i]
                    # Try a few random replacements
                    for _ in range(50):
                        if time.time() > deadline:
                            break
                        new_val = random.randint(0, 128)
                        if new_val == old_val or new_val in arr:
                            continue
                        trial = arr[:]
                        trial[i] = new_val
                        trial.sort()
                        trial_score = _fast_max_sidon_size(trial)
                        if trial_score < best_score:
                            best_score = trial_score
                            best_arr = trial[:]
                            if best_score <= 4:
                                break
                        if best_score <= 4:
                            break
                    if best_score <= 4:
                        break
                if best_score <= 4:
                    break
        
        if best_score <= 4:
            break
    
    # Ensure exactly 12 distinct integers in [0, 128]
    assert len(best_arr) == 12
    assert len(set(best_arr)) == 12
    assert all(0 <= x <= 128 for x in best_arr)
    
    return best_arr
# EVOLVE-BLOCK-END
