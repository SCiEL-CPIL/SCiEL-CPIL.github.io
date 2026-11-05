import random
import time
from itertools import combinations

def is_sidon(subset):
    """Check if a given subset has all unique pair sums including diagonals."""
    n = len(subset)
    sums = set()
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size_fast(candidate_set):
    """Find the size of the largest Sidon subset with early pruning."""
    n = len(candidate_set)
    cs = sorted(candidate_set)
    for k in range(n, 0, -1):
        for subset in combinations(cs, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_set = list(range(12))
    best_size = max_sidon_size_fast(best_set)
    
    # Structured candidates designed to force pair-sum collisions
    # Key insight: arithmetic progressions and dense clusters create many sum collisions
    candidates = [
        list(range(12)),  # Consecutive: many collisions
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32],  # Block structure
        [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],  # Tighter blocks
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],  # Two blocks
        [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13],  # Two blocks
        [0, 1, 3, 7, 12, 18, 25, 33, 42, 52, 63, 75],
        [0, 1, 4, 10, 20, 40, 80, 2, 5, 15, 35, 75],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],  # Even spaced
        [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17],  # Periodic blocks
        # Additional structured candidates
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],  # One outlier
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # Same as range(12)
        [0, 0+1, 0+2, 1+1, 1+2, 2+1, 2+2, 3+1, 3+2, 4+1, 4+2, 5+1],  # Grid-like
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
        [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12],
        [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
        [0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15],
        [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15],
    ]
    
    for c in candidates:
        if len(set(c)) != 12: continue
        if any(x < 0 or x > 128 for x in c): continue
        sz = max_sidon_size_fast(c)
        if sz < best_size:
            best_size = sz
            best_set = list(c)
    
    # Local search with improved mutation strategies
    random.seed(42)
    current_set = list(best_set)
    current_size = best_size
    
    def try_swap(s):
        """Try swapping two elements for a better value."""
        best_s = list(s)
        best_sz = max_sidon_size_fast(s)
        for _ in range(30):
            new_s = list(s)
            i, j = random.sample(range(12), 2)
            # Try to find a value that creates more collisions
            new_val = random.randint(0, 128)
            if new_val in new_s:
                continue
            new_s[j] = new_val
            new_sz = max_sidon_size_fast(new_s)
            if new_sz < best_sz:
                best_sz = new_sz
                best_s = new_s
        return best_s, best_sz
    
    def try_targeted_collision(s):
        """Try to create additional sum collisions by targeting specific pairs."""
        best_s = list(s)
        best_sz = max_sidon_size_fast(s)
        s_sorted = sorted(s)
        # Try to make sums collide by adjusting elements
        for _ in range(20):
            new_s = list(s)
            idx = random.randint(0, 11)
            # Try values near existing elements to create collisions
            if idx > 0:
                # Try values that would create sum collisions with other pairs
                other_idx = random.randint(0, 11)
                if other_idx == idx:
                    continue
                target_sum = s[other_idx] + s[other_idx]  # diagonal sum
                # Try to make s[idx] + s[other_idx] = target_sum
                candidate = target_sum - s[other_idx]
                if candidate != s[idx] and 0 <= candidate <= 128 and candidate not in new_s:
                    new_s[idx] = candidate
                    new_sz = max_sidon_size_fast(new_s)
                    if new_sz < best_sz:
                        best_sz = new_sz
                        best_s = new_s
        return best_s, best_sz
    
    # Simulated annealing style local search
    while time.time() < deadline:
        # Try multiple mutation rounds
        for _ in range(15):
            if time.time() >= deadline:
                break
            
            # Alternate between different mutation strategies
            strategy = random.random()
            if strategy < 0.4:
                mutated, mut_sz = try_swap(current_set)
            elif strategy < 0.7:
                mutated, mut_sz = try_targeted_collision(current_set)
            else:
                # Random restart from current
                mutated = random.sample(range(129), 12)
                mut_sz = max_sidon_size_fast(mutated)
            
            if mut_sz < current_size:
                current_set = mutated
                current_size = mut_sz
                if current_size < best_size:
                    best_size = current_size
                    best_set = list(current_set)
                    if best_size <= 3:
                        break
        if best_size <= 3:
            break
    
    # Final random restarts if time remains
    if time.time() < deadline - 5:
        while time.time() < deadline - 2:
            s = random.sample(range(129), 12)
            sz = max_sidon_size_fast(s)
            if sz < best_size:
                best_size = sz
                best_set = s
                if best_size <= 3:
                    break
    
    return best_set
