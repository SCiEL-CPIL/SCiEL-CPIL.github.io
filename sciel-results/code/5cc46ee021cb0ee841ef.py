import time
import random
from collections import defaultdict

# Precompute prime factors for numbers up to 20000 (max sum is 10000+10000)
MAX_VAL = 10000
MAX_SUM = 20000

def get_prime_factors(n):
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

# Sieve of Eratosthenes variant for fast factorization
def precompute_factors(limit):
    pf = [[] for _ in range(limit + 1)]
    for i in range(2, limit + 1):
        if not pf[i]:
            for j in range(i, limit + 1, i):
                pf[j].append(i)
    return pf

# Precompute at module load time to avoid re-doing it if construct is called multiple times
# But the task says construct() is the entry point. We can use a global cache.
_pf_cache = None

def get_pf(n):
    global _pf_cache
    if _pf_cache is None:
        _pf_cache = precompute_factors(20000)
    return _pf_cache[n]

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s budget, 10s safety margin
    
    n = 12
    max_val = 10000
    
    # Initialize with a random set or small integers
    # Small integers often have smaller sums, which might have fewer/smaller primes
    # But we want to minimize the COUNT of distinct primes.
    # Let's start with range(1, 13)
    current_set = list(range(1, 13))
    
    # Calculate initial counts
    # counts[p] = number of pairs (i, j) in current_set such that p divides sum[i] + sum[j]
    counts = defaultdict(int)
    
    def update_counts_remove(idx, arr, counts):
        a = arr[idx]
        for i in range(n):
            if i != idx:
                s = a + arr[i]
                for p in get_pf(s):
                    counts[p] -= 1
                    if counts[p] == 0:
                        # We don't delete key to keep structure, but active count logic needs care
                        pass
    
    def update_counts_add(idx, arr, counts, new_val):
        for i in range(n):
            if i != idx:
                s = new_val + arr[i]
                for p in get_pf(s):
                    if counts[p] == 0:
                        # This prime was inactive, now becomes active
                        pass
                    counts[p] += 1

    # Better approach: 
    # Keep a set of active primes? No, counts is better.
    # Let's just calculate the number of active primes from counts.
    
    def get_active_prime_count(counts):
        return sum(1 for v in counts.values() if v > 0)

    # Initialize counts for current_set
    for i in range(n):
        for j in range(i + 1, n):
            s = current_set[i] + current_set[j]
            for p in get_pf(s):
                counts[p] += 1
                
    best_count = get_active_prime_count(counts)
    best_set = current_set[:]
    
    # Local Search
    # We will try to swap elements.
    # To speed up, we precompute a list of candidate values.
    # Candidates: 1..10000
    
    # Optimization: 
    # When swapping arr[k] with new_val:
    # 1. Decrement counts for sums involving arr[k].
    # 2. Increment counts for sums involving new_val.
    # 3. Check if active count decreased.
    
    # We need to be careful with the counts dict modification.
    # We will create a copy of counts for each trial? 
    # No, that's slow. 
    # We will modify counts, check, and revert if not better.
    
    # Reverting requires knowing which primes changed.
    # We can store the old values of the affected primes.
    
    # Let's iterate.
    while time.time() < deadline:
        improved = False
        # Shuffle indices to try different orders
        indices = list(range(n))
        random.shuffle(indices)
        
        for k in indices:
            old_val = current_set[k]
            # Try a few random new values
            # To find better solutions, we should try values that might share factors.
            # But random search with many iterations usually works.
            # Let's try 100 random candidates per index per pass.
            for _ in range(50):
                new_val = random.randint(1, max_val)
                if new_val == old_val:
                    continue
                if new_val in current_set:
                    continue
                
                # Calculate impact
                affected_primes = set()
                
                # Remove old sums
                for i in range(n):
                    if i != k:
                        s = old_val + current_set[i]
                        for p in get_pf(s):
                            affected_primes.add(p)
                            counts[p] -= 1
                
                # Add new sums
                for i in range(n):
                    if i != k:
                        s = new_val + current_set[i]
                        for p in get_pf(s):
                            affected_primes.add(p)
                            counts[p] += 1
                            
                new_count = get_active_prime_count(counts)
                
                if new_count < best_count:
                    # Accept
                    current_set[k] = new_val
                    best_count = new_count
                    best_set = current_set[:]
                    improved = True
                else:
                    # Revert
                    for i in range(n):
                        if i != k:
                            # Re-add old sums
                            s = old_val + current_set[i]
                            for p in get_pf(s):
                                counts[p] += 1
                            # Re-remove new sums
                            s = new_val + current_set[i]
                            for p in get_pf(s):
                                counts[p] -= 1

        if not improved:
            # Maybe restart with a different initialization if stuck?
            # Or just continue random searching.
            # If we are stuck at a local minimum, we might need a restart.
            # Let's try a restart every 5 seconds if no improvement.
            pass

    return best_set
