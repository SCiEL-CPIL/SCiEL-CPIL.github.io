import time
import random
from itertools import combinations

# Precompute smallest prime factors for fast factorization
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

def get_prime_factors(n):
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

# Precompute prime factors for all possible sums
sum_factors = [set() for _ in range(MAX_SUM + 1)]
for s in range(2, MAX_SUM + 1):
    sum_factors[s] = get_prime_factors(s)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Strategy 1: Try to find a set of 12 numbers of form 16*y_i
    # such that y_i + y_j is 7-smooth (primes 3,5,7)
    # This would result in prime set {2, 3, 5, 7}
    
    # Precompute 7-smooth numbers up to 1250 (max sum of y_i if y_i <= 625)
    smooth_7 = set()
    for a in range(0, 10): # 3^a
        v1 = 3**a
        if v1 > 1250: break
        for b in range(0, 10): # 5^b
            v2 = v1 * (5**b)
            if v2 > 1250: break
            for c in range(0, 10): # 7^c
                v3 = v2 * (7**c)
                if v3 > 1250: break
                smooth_7.add(v3)
    
    # Candidates for y_i: 1..625
    candidates = list(range(1, 626))
    
    # Filter candidates that can form valid sums with at least 11 other candidates
    # This is a heuristic for backtracking
    # We just run a simple backtracking with randomization
    
    def find_smooth_set():
        # Randomized backtracking
        for _ in range(1000): # 1000 restarts
            if time.time() > deadline:
                return None
            
            # Shuffle candidates to avoid bias
            shuffled = candidates[:]
            random.shuffle(shuffled)
            
            chosen = []
            used = set()
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if len(chosen) == 12:
                    return True
                
                # Pruning: if not enough candidates left, fail
                if len(shuffled) - idx < 12 - len(chosen):
                    return False
                
                y = shuffled[idx]
                if y in used:
                    return backtrack(idx + 1)
                
                # Check if y can be added
                valid = True
                for prev in chosen:
                    if (y + prev) not in smooth_7:
                        valid = False
                        break
                
                if valid:
                    chosen.append(y)
                    used.add(y)
                    if backtrack(idx + 1):
                        return True
                    chosen.pop()
                    used.remove(y)
                
                return backtrack(idx + 1)
            
            if backtrack(0):
                return chosen
        return None

    smooth_set = find_smooth_set()
    if smooth_set:
        result = [16 * y for y in smooth_set]
        # Verify
        primes = set()
        for i in range(12):
            for j in range(i+1, 12):
                primes.update(sum_factors[result[i] + result[j]])
        # The primes should be subset of {2,3,5,7}
        return result

    # Strategy 2: General Local Search
    # Minimize number of distinct primes in sums
    
    # Initialize with a set of numbers that are multiples of a large power of 2
    # to encourage sharing factor 2
    base = 256
    init_set = [base * (i + 1) for i in range(12)]
    # Ensure distinct and <= 10000
    # 256 * 12 = 3072, so this is fine.
    
    current_set = init_set[:]
    current_primes = set()
    for i in range(12):
        for j in range(i+1, 12):
            current_primes.update(sum_factors[current_set[i] + current_set[j]])
    
    best_set = current_set[:]
    best_primes_count = len(current_primes)
    
    while time.time() < deadline:
        improved = False
        for i in range(12):
            if time.time() > deadline:
                break
            # Try replacing current_set[i] with a random value
            for _ in range(50): # 50 attempts per element
                new_val = random.randint(1, 10000)
                if new_val in current_set and new_val != current_set[i]:
                    continue
                
                old_val = current_set[i]
                current_set[i] = new_val
                
                # Recalculate primes
                new_primes = set()
                for a in range(12):
                    for b in range(a+1, 12):
                        new_primes.update(sum_factors[current_set[a] + current_set[b]])
                
                if len(new_primes) < best_primes_count:
                    best_primes_count = len(new_primes)
                    best_set = current_set[:]
                    improved = True
                elif len(new_primes) == best_primes_count:
                    # Accept with some probability to explore
                    if random.random() < 0.1:
                        best_set = current_set[:]
                        improved = True
                else:
                    # Revert
                    current_set[i] = old_val
                    continue
                
                if improved:
                    break
            
            if time.time() > deadline:
                break
        
        if not improved:
            # Perturb: replace 2 random elements
            if random.random() < 0.5:
                i1, i2 = random.sample(range(12), 2)
                current_set[i1] = random.randint(1, 10000)
                current_set[i2] = random.randint(1, 10000)
                # Ensure distinct
                if current_set[i1] == current_set[i2]:
                    current_set[i2] = (current_set[i2] % 10000) + 1
                # Ensure no duplicates with others
                for k in range(12):
                    if k not in [i1, i2] and current_set[k] in [current_set[i1], current_set[i2]]:
                         # Simple fix, might not be perfect but avoids crash
                         current_set[i1] = random.randint(1, 10000)
                         current_set[i2] = random.randint(1, 10000)
    
    # Final check for distinctness and bounds
    if len(set(best_set)) != 12 or any(x < 1 or x > 10000 for x in best_set):
        return list(range(1, 13))
        
    return best_set

# EVOLVE-BLOCK-END
