import time
import random
from itertools import combinations

def _is_sidon_fast(subset):
    """Check Sidon property. subset must be sorted."""
    n = len(subset)
    if n <= 2:
        return True
    # Use a set for sums
    seen = set()
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if s in seen:
                return False
            seen.add(s)
    return True

def _max_sidon_size(arr):
    """arr is sorted list of 12 distinct ints. Return largest Sidon subset size."""
    n = len(arr)
    # Check full set
    if _is_sidon_fast(arr):
        return n
    # Check subsets of size n-1
    for skip in range(n):
        subset = [arr[i] for i in range(n) if i != skip]
        if _is_sidon_fast(subset):
            return n - 1
    # Check subsets of size n-2
    for i in range(n):
        for j in range(i + 1, n):
            subset = [arr[k] for k in range(n) if k != i and k != j]
            if _is_sidon_fast(subset):
                return n - 2
    # For size 8 and below, use a different approach
    # Actually let's just check all combinations for sizes 10, 9, 8, 7, 6, 5
    for k in range(n - 3, 4, -1):
        for comb in combinations(range(n), k):
            subset = [arr[i] for i in comb]
            if _is_sidon_fast(subset):
                return k
    # Check size 5
    for comb in combinations(range(n), 5):
        subset = [arr[i] for i in comb]
        if _is_sidon_fast(subset):
            return 5
    return 4

def construct():
    start = time.time()
    deadline = start + 80
    
    best = list(range(12))
    best_score = _max_sidon_size(best)
    
    random.seed(12345)
    
    # Local search with restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        if restart == 0:
            current = best[:]
        else:
            # Perturb best
            current = best[:]
            for _ in range(random.randint(2, 5)):
                i = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val not in current:
                    current[i] = new_val
            current.sort()
        
        current_score = _max_sidon_size(current)
        
        for step in range(5000):
            if time.time() > deadline:
                break
            
            # Try replacing one element
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in current:
                continue
            
            trial = current[:]
            trial[i] = new_val
            trial.sort()
            
            trial_score = _max_sidon_size(trial)
            
            if trial_score < current_score:
                current = trial
                current_score = trial_score
                if current_score < best_score:
                    best = current[:]
                    best_score = current_score
                    if best_score <= 4:
                        return best
            elif trial_score == current_score:
                if random.random() < 0.2:
                    current = trial
    
    return best
