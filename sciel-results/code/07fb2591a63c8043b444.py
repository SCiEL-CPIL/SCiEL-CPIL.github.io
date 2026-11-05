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

def has_sidon_size(candidate_set, k):
    """Check if candidate_set contains a Sidon subset of size k using backtracking with pruning."""
    s = sorted(candidate_set)
    n = len(s)
    
    def backtrack(start, current_subset, used_sums):
        if len(current_subset) == k:
            return True
        if n - start < k - len(current_subset):
            return False
        
        for i in range(start, n):
            x = s[i]
            new_sums = set()
            
            # Check diagonal
            if x + x in used_sums:
                continue
            new_sums.add(x + x)
            
            # Check off-diagonal
            conflict = False
            for y in current_subset:
                val = x + y
                if val in used_sums or val in new_sums:
                    conflict = True
                    break
                new_sums.add(val)
            
            if conflict:
                continue
            
            if backtrack(i + 1, current_subset + [x], used_sums | new_sums):
                return True
        
        return False
    
    return backtrack(0, [], set())

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Initial candidates with structured designs
    # Focus on dense additive collision structures
    candidates = [
        list(range(12)),
        [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32],
        [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23],
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13],
        [0, 1, 3, 7, 12, 18, 25, 33, 42, 52, 63, 75],
        [0, 1, 4, 10, 20, 40, 80, 2, 5, 15, 35, 75],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17],
        list(range(0, 24, 2)),
        [0, 1, 2, 4, 8, 16, 32, 64, 2, 3, 5, 9],
        [0, 1, 3, 5, 8, 13, 21, 34, 55, 89, 120, 128],
        # Additional dense collision candidates
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 18, 19],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 17, 18],
        [0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16, 17],
        [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 16],
        [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],
        [0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15],
        [0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15],
        [0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    ]
    
    best_set = None
    best_size = 13
    
    # Evaluate initial candidates
    for c in candidates:
        if time.time() > deadline:
            break
        if len(set(c)) != 12:
            continue
        if any(x < 0 or x > 128 for x in c):
            continue
        
        sz = max_sidon_size(c)
        if sz < best_size:
            best_size = sz
            best_set = list(c)
            if best_size <= 3:
                break
    
    if best_set is None:
        best_set = list(range(12))
        best_size = max_sidon_size(best_set)
    
    # Local search with simulated annealing
    random.seed(42)
    current_set = list(best_set)
    current_size = best_size
    temperature = 1.0
    
    def try_mutate(s, sz):
        """Try a random mutation and return the new set and size."""
        new_s = list(s)
        idx = random.randint(0, 11)
        new_val = random.randint(0, 128)
        if new_val in new_s:
            return s, sz
        new_s[idx] = new_val
        new_sz = max_sidon_size(new_s)
        return new_s, new_sz
    
    # Simulated annealing loop
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        new_s, new_sz = try_mutate(current_set, current_size)
        
        # Accept if better, or with probability based on temperature
        if new_sz < current_size:
            current_set = new_s
            current_size = new_sz
            if current_size < best_size:
                best_size = current_size
                best_set = list(current_set)
                if best_size <= 3:
                    break
        elif random.random() < pow(2.718281828, -(new_sz - current_size) / temperature):
            current_set = new_s
            current_size = new_sz
        
        # Cool down
        temperature *= 0.999
        if temperature < 0.01:
            temperature = 0.01
    
    # Final random restarts if time remains
    if time.time() < deadline - 5:
        while time.time() < deadline - 2:
            s = random.sample(range(129), 12)
            # Use fast check: does it have a Sidon subset of size 5?
            if has_sidon_size(s, 5):
                continue
            # Check size 4
            if has_sidon_size(s, 4):
                sz = 4
            else:
                sz = 3
            
            if sz < best_size:
                best_size = sz
                best_set = s
                if best_size <= 3:
                    break
    
    return best_set
