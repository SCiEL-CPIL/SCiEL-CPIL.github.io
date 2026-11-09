import random
import time
import itertools
from collections import defaultdict

def is_sidon(subset):
    """Check if a subset is Sidon (all pair sums including diagonals are unique)."""
    n = len(subset)
    if n <= 1:
        return True
    sums = []
    for i in range(n):
        for j in range(i, n):
            sums.append(subset[i] + subset[j])
    return len(set(sums)) == len(sums)

def largest_sidon_subset_size(s):
    """Find the size of the largest Sidon subset of s."""
    s = sorted(s)
    n = len(s)
    best = 0
    # Try all subsets in descending order of size
    for size in range(n, best, -1):
        for combo in itertools.combinations(range(n), size):
            subset = [s[i] for i in combo]
            if is_sidon(subset):
                return size
    return best

def count_sidon_4_subsets(s):
    """Count the number of 4-element Sidon subsets as a fast surrogate."""
    s = sorted(s)
    n = len(s)
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    subset = [s[i], s[j], s[k], s[l]]
                    if is_sidon(subset):
                        count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    best_set = None
    best_score = float('inf')
    
    def update_best(s, score):
        nonlocal best_set, best_score
        if score < best_score:
            best_score = score
            best_set = list(s)
            if best_score <= 3:  # Can't do much better than 3
                return True
        return False

    random.seed(42)
    
    # Try structured initializations
    candidates = [
        list(range(12)),  # AP
        [0, 1, 3, 7, 15, 31, 63, 127, 2, 4, 8, 16],  # Mixed
        [0, 1, 2, 4, 8, 16, 32, 64, 5, 10, 20, 40],  # Powers + others
        [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66],  # Triangular
        [0, 1, 2, 5, 7, 13, 18, 26, 35, 46, 58, 71],  # Fibonacci-like
    ]
    
    for cand in candidates:
        if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            score = largest_sidon_subset_size(cand)
            if update_best(cand, score):
                return best_set
    
    # Random search with local improvement using surrogate for speed
    T = 1.0
    cooling_rate = 0.995
    min_T = 0.001
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        # Random initialization
        current_set = sorted(random.sample(range(129), 12))
        current_score = largest_sidon_subset_size(current_set)
        current_surrogate = count_sidon_4_subsets(current_set)
        
        if update_best(current_set, current_score):
            return best_set
        
        # Local search
        T = 1.0
        no_improve = 0
        while T > min_T and time.time() < deadline and no_improve < 50:
            # Generate neighbor: replace one or two elements
            new_set = current_set.copy()
            num_replacements = random.choice([1, 1, 2])
            
            for _ in range(num_replacements):
                idx = random.randint(0, 11)
                old_val = new_set[idx]
                # Try nearby values first for better exploration
                if random.random() < 0.5:
                    offset = random.randint(-5, 5)
                    new_val = max(0, min(128, old_val + offset))
                else:
                    new_val = random.randint(0, 128)
                
                if new_val != old_val and new_val not in new_set:
                    new_set[idx] = new_val
                    new_set.sort()
            
            if len(set(new_set)) != 12:
                no_improve += 1
                continue
            
            # Use surrogate for quick comparison
            new_surrogate = count_sidon_4_subsets(new_set)
            
            if new_surrogate < current_surrogate or random.random() < pow(2.718, -(new_surrogate - current_surrogate) / max(T, 1e-9)):
                # Verify with exact score before accepting
                new_score = largest_sidon_subset_size(new_set)
                delta = new_score - current_score
                
                if delta < 0 or random.random() < pow(2.718, -delta / max(T, 1e-9)):
                    current_set = new_set
                    current_score = new_score
                    current_surrogate = new_surrogate
                    no_improve = 0
                    
                    if update_best(current_set, current_score):
                        return best_set
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
            T *= cooling_rate
    
    return best_set
