import time
import random

# Precompute smallest prime factor for numbers up to 20000 (max sum = 2*10000)
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

# Precompute prime factors for all numbers up to MAX_SUM
# factors[n] is a set of primes
factors = [set() for _ in range(MAX_SUM + 1)]
for i in range(2, MAX_SUM + 1):
    n = i
    while n > 1:
        p = spf[n]
        factors[i].add(p)
        while n % p == 0:
            n //= p

def get_prime_counts(arr):
    """
    Returns a dictionary mapping prime -> count of sums containing that prime.
    """
    counts = {}
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            for p in factors[s]:
                counts[p] = counts.get(p, 0) + 1
    return counts

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    # Initial solution: Powers of 2
    # 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    arr = [1 << i for i in range(12)]
    counts = get_prime_counts(arr)
    best_score = len(counts)
    
    # Candidate pool: Smooth numbers (2,3,5) and randoms
    smooth_candidates = set()
    for i in range(15):
        for j in range(10):
            for k in range(8):
                val = (1 << i) * (3 ** j) * (5 ** k)
                if val <= 10000:
                    smooth_candidates.add(val)
    
    random.seed(42)
    for _ in range(200):
        smooth_candidates.add(random.randint(1, 10000))
        
    candidates = list(smooth_candidates)
    
    # Hill Climbing
    while time.time() < deadline:
        improved = False
        random.shuffle(candidates)
        
        for i in range(12):
            old_val = arr[i]
            
            # Calculate primes to remove (from sums involving old_val)
            primes_to_remove = {}
            for j in range(12):
                if i == j: 
                    continue
                s = old_val + arr[j]
                for p in factors[s]:
                    primes_to_remove[p] = primes_to_remove.get(p, 0) + 1
            
            for c in candidates:
                if c == old_val:
                    continue
                # Check distinctness
                if c in arr:
                    if arr.index(c) != i:
                        continue
                    else:
                        # c is at i, which is old_val, already checked c == old_val
                        pass 
                
                # Calculate primes to add (from sums involving c)
                primes_to_add = {}
                for j in range(12):
                    if i == j: 
                        continue
                    s = c + arr[j]
                    for p in factors[s]:
                        primes_to_add[p] = primes_to_add.get(p, 0) + 1
                
                # Update counts
                new_counts = counts.copy()
                
                # Remove old contributions
                for p, cnt in primes_to_remove.items():
                    new_counts[p] -= cnt
                    if new_counts[p] == 0:
                        del new_counts[p]
                
                # Add new contributions
                for p, cnt in primes_to_add.items():
                    if p in new_counts:
                        new_counts[p] += cnt
                    else:
                        new_counts[p] = cnt
                
                new_score = len(new_counts)
                
                if new_score < best_score:
                    best_score = new_score
                    counts = new_counts
                    arr[i] = c
                    improved = True
        
        if not improved:
            # Add new random candidates to explore
            for _ in range(20):
                candidates.append(random.randint(1, 10000))
            
    return arr
