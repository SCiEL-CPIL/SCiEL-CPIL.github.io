import random
import time
import itertools

def is_sidon_4(subset):
    """Check if a 4-element subset is Sidon (all 10 pair sums including diagonals are unique)."""
    s0, s1, s2, s3 = subset
    sums = [
        2*s0, s0+s1, s0+s2, s0+s3,
        2*s1, s1+s2, s1+s3,
        2*s2, s2+s3,
        2*s3
    ]
    return len(set(sums)) == 10

def count_sidon_4_subsets(s):
    """Count the number of Sidon 4-subsets in set s."""
    count = 0
    for combo in itertools.combinations(s, 4):
        if is_sidon_4(combo):
            count += 1
    return count

def largest_sidon_subset_size(s):
    """Find the size of the largest Sidon subset of s."""
    s = sorted(s)
    n = len(s)
    # Check from size 5 down to 1
    for size in range(min(n, 5), 0, -1):
        for combo in itertools.combinations(s, size):
            if is_sidon_4(combo) if size == 4 else _is_sidon_general(combo):
                return size
    return 1

def _is_sidon_general(subset):
    """Check if a subset of any size is Sidon."""
    n = len(subset)
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_set = None
    best_score = float('inf')
    
    def update_best(s, score):
        nonlocal best_set, best_score
        if score < best_score:
            best_score = score
            best_set = list(s)
            if best_score == 0:  # No Sidon 4-subsets, so max Sidon size <= 3
                return True
        return False
    
    random.seed(42)
    
    # Try structured initializations that maximize collisions
    candidates = [
        # AP with step 2
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        # Two interleaved APs
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        # Dense small values
        list(range(12)),
        # AP with step 3
        [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33],
        # Mixed dense
        [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13],
    ]
    
    for cand in candidates:
        if len(set(cand)) == 12 and all(0 <= x <= 128 for x in cand):
            score = count_sidon_4_subsets(cand)
            if update_best(cand, score):
                return best_set
    
    # Local search to minimize Sidon 4-subsets
    current_set = sorted(random.sample(range(129), 12))
    current_score = count_sidon_4_subsets(current_set)
    
    if update_best(current_set, current_score):
        return best_set
    
    T = 1.0
    cooling_rate = 0.995
    min_T = 0.01
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Random initialization
        current_set = sorted(random.sample(range(129), 12))
        current_score = count_sidon_4_subsets(current_set)
        
        if update_best(current_set, current_score):
            return best_set
        
        # Local search
        T = 1.0
        no_improve = 0
        while T > min_T and time.time() < deadline and no_improve < 100:
            # Generate neighbor: replace one element
            new_set = current_set.copy()
            idx = random.randint(0, 11)
            old_val = new_set[idx]
            
            # Prefer values that create more collisions
            if random.random() < 0.7:
                # Pick a value close to existing elements to create collisions
                base = random.choice(current_set)
                offset = random.randint(-3, 3)
                new_val = max(0, min(128, base + offset))
            else:
                new_val = random.randint(0, 128)
            
            if new_val != old_val and new_val not in new_set:
                new_set[idx] = new_val
                new_set.sort()
                
                if len(set(new_set)) == 12:
                    new_score = count_sidon_4_subsets(new_set)
                    delta = new_score - current_score
                    
                    if delta < 0 or random.random() < pow(2.718, -delta / max(T, 1e-9)):
                        current_set = new_set
                        current_score = new_score
                        no_improve = 0
                        
                        if update_best(current_set, current_score):
                            return best_set
                    else:
                        no_improve += 1
                    
                    T *= cooling_rate
            else:
                no_improve += 1
                T *= cooling_rate
    
    return best_set
