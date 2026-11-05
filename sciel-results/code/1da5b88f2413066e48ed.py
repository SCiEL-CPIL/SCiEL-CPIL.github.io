import random
import time
from itertools import combinations

def is_sidon(subset):
    """Check if a given subset has all unique pair sums including diagonals."""
    sums = set()
    n = len(subset)
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(candidate_set):
    """Find the size of the largest Sidon subset of the given candidate set."""
    n = len(candidate_set)
    for k in range(n, 0, -1):
        for subset in combinations(candidate_set, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best_set = list(range(12))
    best_size = max_sidon_size(best_set)
    
    # Structured candidates designed to force pair-sum collisions
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
    ]
    
    for c in candidates:
        if len(set(c)) != 12: continue
        if any(x < 0 or x > 128 for x in c): continue
        sz = max_sidon_size(c)
        if sz < best_size:
            best_size = sz
            best_set = list(c)
    
    # Local search: mutate one element at a time
    random.seed(42)
    current_set = list(best_set)
    current_size = best_size
    
    def try_mutate(s, sz):
        """Try random mutations and return best result."""
        best_s = list(s)
        best_sz = sz
        for _ in range(50):
            new_s = list(s)
            idx = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in new_s:
                continue
            new_s[idx] = new_val
            new_sz = max_sidon_size(new_s)
            if new_sz < best_sz:
                best_sz = new_sz
                best_s = new_s
        return best_s, best_sz
    
    # Simulated annealing style local search
    while time.time() < deadline:
        # Try multiple mutation rounds
        for _ in range(10):
            if time.time() >= deadline:
                break
            mutated, mut_sz = try_mutate(current_set, current_size)
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
            sz = max_sidon_size(s)
            if sz < best_size:
                best_size = sz
                best_set = s
                if best_size <= 3:
                    break
    
    return best_set
