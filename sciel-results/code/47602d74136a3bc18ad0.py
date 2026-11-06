import math
import random
import time

def _prime_factors(n):
    """Return set of prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _count_primes(k_list):
    """Count distinct primes in sums k_i + k_j + 1."""
    primes = set()
    n = len(k_list)
    for i in range(n):
        for j in range(i + 1, n):
            s = k_list[i] + k_list[j] + 1
            primes.update(_prime_factors(s))
    return len(primes)

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 45  # 45s safety margin
    
    # Initial candidate: k_i = i
    k = list(range(12))
    best_score = _count_primes(k)
    best_k = k[:]
    
    # Local search
    # Try to replace k_i with a new value to reduce prime count
    # We want to keep k_i distinct and small (to keep a_i <= 10000)
    # a_i = 30 * k_i + 15 <= 10000 => k_i <= 332
    
    # Precompute prime factors for small numbers to speed up
    # Max sum of k_i is roughly 332+331+1 = 664. 
    # Let's precompute up to 1000.
    max_k = 400
    pf_cache = {}
    for i in range(1, max_k + 100):
        pf_cache[i] = _prime_factors(i)
    
    def _count_primes_fast(k_list):
        primes = set()
        n = len(k_list)
        for i in range(n):
            for j in range(i + 1, n):
                s = k_list[i] + k_list[j] + 1
                if s in pf_cache:
                    primes.update(pf_cache[s])
                else:
                    primes.update(_prime_factors(s))
        return len(primes)
    
    best_score = _count_primes_fast(k)
    
    # Random restarts and local search
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize k
        if r == 0:
            curr_k = list(range(12))
        else:
            # Perturb best_k or random
            curr_k = best_k[:]
            for _ in range(3):
                idx = random.randint(0, 11)
                curr_k[idx] = random.randint(0, max_k)
                # Ensure distinct
                if len(set(curr_k)) < 12:
                    # Fix duplicates
                    used = set(curr_k)
                    for i in range(12):
                        if curr_k[i] in used and i != list(used).index(curr_k[i]):
                             # Simple fix: increment until distinct
                             while curr_k[i] in used:
                                 curr_k[i] += 1
                             used.add(curr_k[i])
        
        curr_score = _count_primes_fast(curr_k)
        
        # Local search steps
        steps = 0
        max_steps = 500
        while steps < max_steps and time.time() < deadline:
            steps += 1
            improved = False
            # Try swapping or replacing
            for i in range(12):
                if time.time() > deadline:
                    break
                old_val = curr_k[i]
                # Try new values
                # Focus on values that are "good" (small prime factors in sums)
                # Sample some candidates
                candidates = []
                # Nearby values
                for delta in range(-10, 11):
                    nv = old_val + delta
                    if 0 <= nv <= max_k:
                        candidates.append(nv)
                # Random values
                for _ in range(5):
                    candidates.append(random.randint(0, max_k))
                
                for nv in candidates:
                    if nv == old_val:
                        continue
                    # Check distinctness
                    if nv in curr_k[:i] or nv in curr_k[i+1:]:
                        continue
                    
                    curr_k[i] = nv
                    new_score = _count_primes_fast(curr_k)
                    
                    if new_score < curr_score:
                        curr_score = new_score
                        improved = True
                        if curr_score < best_score:
                            best_score = curr_score
                            best_k = curr_k[:]
                        # Continue searching from new state
                        break
                    else:
                        curr_k[i] = old_val
                
                if improved:
                    break
            
            if not improved:
                # Stuck, perturb
                idx = random.randint(0, 11)
                old_val = curr_k[idx]
                while True:
                    nv = random.randint(0, max_k)
                    if nv not in curr_k:
                        break
                curr_k[idx] = nv
                curr_score = _count_primes_fast(curr_k)

    # Construct final a_i from best_k
    # a_i = 30 * k_i + 15
    a = [30 * ki + 15 for ki in best_k]
    
    # Ensure distinct and <= 10000
    # If duplicates or > 10000, fallback to seed or fix
    if len(set(a)) < 12 or max(a) > 10000:
        # Fallback to a simple valid set
        # Try to fix by incrementing
        a_sorted = sorted(a)
        for i in range(1, 12):
            if a_sorted[i] <= a_sorted[i-1]:
                a_sorted[i] = a_sorted[i-1] + 1
        # Check limit
        if max(a_sorted) > 10000:
            # Scale down? Or just use 1..12
            return list(range(1, 13))
        return a_sorted

    return a
