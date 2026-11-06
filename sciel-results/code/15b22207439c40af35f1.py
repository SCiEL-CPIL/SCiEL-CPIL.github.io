import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    MAX_SUM = 19999
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    fact_cache = {}
    def get_factors(n):
        if n not in fact_cache:
            if n <= 1:
                fact_cache[n] = frozenset()
            else:
                factors = set()
                nn = n
                while nn > 1:
                    p = spf[nn]
                    factors.add(p)
                    while nn % p == 0:
                        nn //= p
                fact_cache[n] = frozenset(factors)
        return fact_cache[n]
    
    n = 12
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def full_objective(arr):
        prime_set = set()
        for i, j in pairs:
            prime_set.update(get_factors(arr[i] + arr[j]))
        return len(prime_set), prime_set
    
    # Generate candidates
    candidates = []
    for base in [2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60, 100, 200, 500]:
        cand = [base * i for i in range(1, 13)]
        if max(cand) <= 10000:
            candidates.append(cand)
    candidates.append([2**i for i in range(12)])
    for start in [1, 10, 100, 1000, 5000, 9000]:
        if start + 11 <= 10000:
            candidates.append([start + i for i in range(12)])
    
    random.seed(42)
    for _ in range(100):
        candidates.append(random.sample(range(1, 10001), 12))
    
    best_arr = None
    best_score = float('inf')
    best_prime_set = None
    
    for cand in candidates:
        if time.time() > deadline - 15:
            break
        s, ps = full_objective(cand)
        if s < best_score:
            best_score = s
            best_arr = cand[:]
            best_prime_set = ps
    
    if time.time() > deadline - 10:
        return best_arr
    
    # Local search with incremental updates
    current_arr = best_arr[:]
    current_score = best_score
    
    # For incremental: when changing arr[idx], only sums involving idx change
    def get_sums_involving(arr, idx):
        """Return list of (other_idx, sum_value) for pairs involving idx."""
        result = []
        for j in range(n):
            if j != idx:
                result.append((j, arr[idx] + arr[j]))
        return result
    
    # Simpler: just do full recompute but be smart about search
    # Try each position, try a range of values
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        improved = False
        for idx in range(n):
            if time.time() > deadline - 5:
                break
            old_val = current_arr[idx]
            # Try values: old_val +/- small range, and some random
            candidates_to_try = set()
            for d in range(-50, 51):
                v = old_val + d
                if 1 <= v <= 10000 and v != old_val:
                    candidates_to_try.add(v)
            # Add some random
            for _ in range(20):
                v = random.randint(1, 10000)
                candidates_to_try.add(v)
            
            best_new_val = None
            best_new_score = current_score
            
            for v in candidates_to_try:
                if v in current_arr and v != old_val:
                    continue
                current_arr[idx] = v
                new_score, _ = full_objective(current_arr)
                if new_score < best_new_score:
                    best_new_score = new_score
                    best_new_val = v
            
            if best_new_val is not None:
                current_arr[idx] = best_new_val
                current_score = best_new_score
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_arr = current_arr[:]
            else:
                current_arr[idx] = old_val
        
        if not improved:
            # Random restart perturbation
            for _ in range(3):
                idx = random.randint(0, n-1)
                current_arr[idx] = random.randint(1, 10000)
            # Ensure distinct
            while len(set(current_arr)) < n:
                idx = random.randint(0, n-1)
                current_arr[idx] = random.randint(1, 10000)
            current_score, _ = full_objective(current_arr)
    
    return best_arr
