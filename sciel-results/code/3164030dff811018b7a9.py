<<<<<<< SEARCH
import time
import random

def is_sidon(subset):
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def has_sidon_4(A):
    # Check if any 4-element subset is Sidon
    from itertools import combinations
    for comb in combinations(A, 4):
        if is_sidon(list(comb)):
            return True
    return False

def max_sidon_size(A):
    # Assume max is 3 or 4
    if has_sidon_4(A):
        return 4
    else:
        return 3

def construct():
    start_time = time.time()
    deadline = start_time + 50 # Safety margin
    
    # Initial set: dense interval
    A = list(range(12))
    best_A = A[:]
    best_size = max_sidon_size(best_A)
    
    # Local Search
    while time.time() < deadline:
        # Randomly pick an index to replace
        idx = random.randint(0, 11)
        old_val = A[idx]
        
        # Pick a new value
        new_val = random.randint(0, 128)
        if new_val in A:
            continue
            
        A[idx] = new_val
        
        current_size = max_sidon_size(A)
        
        if current_size < best_size:
            best_size = current_size
            best_A = A[:]
            if best_size == 3:
                break # Can't get better than 3
        else:
            # Revert
            A[idx] = old_val
            
    return best_A
=======
import time
import random
from itertools import combinations
from collections import defaultdict

def is_sidon(subset):
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def count_sum_collisions(A):
    """Count excess sum occurrences (pairs beyond first for each sum value)."""
    sum_count = defaultdict(int)
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            sum_count[s] += 1
    collisions = 0
    for cnt in sum_count.values():
        if cnt > 1:
            collisions += cnt - 1
    return collisions

def max_sidon_size(A):
    """Find the largest Sidon subset size in A by checking all k from n down to 2."""
    n = len(A)
    for k in range(n, 1, -1):
        for comb in combinations(A, k):
            if is_sidon(list(comb)):
                return k
    return 2

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Initial set: dense interval [0, 11]
    A = list(range(12))
    best_A = A[:]
    best_size = max_sidon_size(best_A)
    best_collisions = count_sum_collisions(best_A)
    
    while time.time() < deadline:
        # Try swap-based moves
        i = random.randint(0, 11)
        j = random.randint(0, 11)
        if i == j:
            continue
        
        A[i], A[j] = A[j], A[i]
        new_collisions = count_sum_collisions(A)
        
        if new_collisions <= best_collisions:
            new_size = max_sidon_size(A)
            if new_size < best_size or (new_size == best_size and new_collisions < best_collisions):
                best_size = new_size
                best_A = A[:]
                best_collisions = new_collisions
                if best_size <= 2:
                    break
            else:
                A[i], A[j] = A[j], A[i]
        else:
            A[i], A[j] = A[j], A[i]
        
        # Try single-element replacement
        idx = random.randint(0, 11)
        old_val = A[idx]
        new_val = random.randint(0, 128)
        if new_val in A:
            continue
        
        A[idx] = new_val
        new_collisions = count_sum_collisions(A)
        
        if new_collisions <= best_collisions:
            new_size = max_sidon_size(A)
            if new_size < best_size or (new_size == best_size and new_collisions < best_collisions):
                best_size = new_size
                best_A = A[:]
                best_collisions = new_collisions
                if best_size <= 2:
                    break
            else:
                A[idx] = old_val
        else:
            A[idx] = old_val
    
    return best_A
>>>>>>> REPLACE
