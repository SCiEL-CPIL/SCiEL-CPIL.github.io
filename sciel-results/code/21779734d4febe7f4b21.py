import time
import random

def is_sidon(S):
    n = len(S)
    if n <= 1:
        return True
    sums = set()
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(A):
    n = len(A)
    max_size = 0
    for mask in range(1 << n):
        # Optimization: if bit count < max_size, skip
        # Using bin(mask).count('1') is slow, manual count is better or precomputed
        # For n=12, 4096 masks, it's fine.
        bit_count = bin(mask).count('1')
        if bit_count <= max_size:
            continue
        subset = [A[i] for i in range(n) if mask & (1 << i)]
        if is_sidon(subset):
            max_size = bit_count
    return max_size

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Heuristic Initial Candidates
    candidates = []
    # 1. Seed
    candidates.append(list(range(12)))
    # 2. Union of 3-APs
    candidates.append([0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32])
    # 3. Another 3-AP union, spread out
    candidates.append([0, 1, 2, 20, 21, 22, 40, 41, 42, 60, 61, 62])
    # 4. Dense cluster
    candidates.append([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    best_set = None
    best_val = float('inf')
    
    for c in candidates:
        v = max_sidon_size(c)
        if v < best_val:
            best_val = v
            best_set = c[:]
            
    # Local Search
    current_set = best_set[:]
    current_val = best_val
    
    random.seed(42)
    
    while time.time() < deadline:
        i = random.randint(0, 11)
        v = random.randint(0, 128)
        
        if v == current_set[i] or v in current_set:
            continue
            
        new_set = current_set[:]
        new_set[i] = v
        
        new_val = max_sidon_size(new_set)
        
        if new_val < best_val:
            best_val = new_val
            best_set = new_set[:]
            current_set = new_set[:]
            current_val = new_val
        elif new_val == current_val:
            if random.random() < 0.7:
                current_set = new_set[:]
                current_val = new_val
        else:
            # If we got worse, we might want to restart from best_set
            if random.random() < 0.1:
                current_set = best_set[:]
                current_val = best_val

    return best_set
