"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from math import isqrt

# EVOLVE-BLOCK-START
def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    if n % 2 == 0:
        factors.add(2)
        while n % 2 == 0:
            n //= 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            factors.add(f)
            while n % f == 0:
                n //= f
        f += 2
    if n > 1:
        factors.add(n)
    return factors

def _is_smooth(n, primes):
    """Check if n is smooth over given primes."""
    for p in primes:
        while n % p == 0:
            n //= p
    return n == 1

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = 12
    max_val = 10000
    rng = random.Random(42)
    
    best_set = None
    best_primes = None
    best_count = float('inf')
    
    def evaluate(arr):
        primes = set()
        for i in range(n):
            for j in range(i + 1, n):
                primes |= _prime_factors(arr[i] + arr[j])
        return primes
    
    # Strategy 1: Try to find 12 numbers that are multiples of M where M has few prime factors
    # and the quotients produce sums with few additional primes.
    
    # Try M = 30 (primes 2,3,5). Need m_i such that m_i + m_j only has primes 2,3,5.
    # Generate 5-smooth numbers up to 10000/30 ≈ 333
    smooth_5 = []
    for a in range(0, 12):  # 2^a
        for b in range(0, 8):  # 3^b
            for c in range(0, 6):  # 5^c
                v = (2 ** a) * (3 ** b) * (5 ** c)
                if v <= 400:
                    smooth_5.append(v)
    smooth_5 = sorted(set(smooth_5))
    
    # Find 12 smooth numbers whose pairwise sums are all smooth
    # Use backtracking
    target_smooth_set = set(smooth_5)
    
    def find_smooth_set(smooth_list, k, current, idx=0):
        if len(current) == k:
            # Verify all pairwise sums are smooth
            for i in range(k):
                for j in range(i + 1, k):
                    if (current[i] + current[j]) not in target_smooth_set:
                        return None
            return list(current)
        for i in range(idx, len(smooth_list)):
            current.append(smooth_list[i])
            result = find_smooth_set(smooth_list, k, current, i + 1)
            if result is not None:
                return result
            current.pop()
        return None
    
    # This might be too slow for large k. Try smaller k first or use heuristic.
    # Actually, let's try a greedy approach with the largest smooth numbers
    # to keep sums small.
    
    # Try: use smooth numbers from a small range
    small_smooth = [x for x in smooth_5 if x <= 100]
    
    # Greedy: start with smallest, add next if all pairwise sums remain smooth
    def greedy_smooth_set(smooth_list, k):
        result = []
        for v in smooth_list:
            if len(result) >= k:
                break
            ok = True
            for existing in result:
                if (existing + v) not in target_smooth_set:
                    ok = False
                    break
            if ok:
                result.append(v)
        return result
    
    smooth_12 = greedy_smooth_set(smooth_5, 12)
    
    if smooth_12 and len(smooth_12) == 12:
        candidate = [30 * m for m in smooth_12]
        if max(candidate) <= max_val and len(set(candidate)) == n:
            primes = evaluate(candidate)
            if len(primes) < best_count:
                best_count = len(primes)
                best_primes = primes
                best_set = list(candidate)
    
    # Strategy 2: Try other multipliers
    for M in [2, 4, 6, 8, 10, 12, 14, 15, 16, 20, 24, 30, 40, 48, 60, 120]:
        if time.time() > deadline - 10:
            break
        M_primes = _prime_factors(M)
        max_m = max_val // M
        if max_m < 12:
            continue
        
        # Generate M-smooth numbers (smooth over M's prime factors) up to max_m
        M_smooth = []
        primes_list = sorted(M_primes)
        
        def gen_smooth(limit, primes, current=1, idx=0):
            if current > limit:
                return
            if idx == len(primes):
                M_smooth.append(current)
                return
            gen_smooth(limit, primes, current, idx + 1)
            p = primes[idx]
            while current * p <= limit:
                current *= p
                gen_smooth(limit, primes, current, idx + 1)
        
        gen_smooth(max_m, primes_list)
        M_smooth = sorted(set(M_smooth))
        
        if len(M_smooth) < 12:
            continue
        
        # Try to find 12 smooth numbers with smooth pairwise sums
        # Use local search
        if len(M_smooth) >= 12:
            # Start with first 12
            current_m = M_smooth[:12]
            # Verify
            valid = True
            for i in range(12):
                for j in range(i + 1, 12):
                    if not _is_smooth(current_m[i] + current_m[j], primes_list):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                candidate = [M * m for m in current_m]
                if max(candidate) <= max_val:
                    primes = evaluate(candidate)
                    if len(primes) < best_count:
                        best_count = len(primes)
                        best_primes = primes
                        best_set = list(candidate)
    
    # Strategy 3: General local search with smart initialization
    # Try several structured candidates
    candidates = []
    
    # Multiples of 30
    candidates.append([30 * (i + 1) for i in range(n)])
    # Multiples of 60
    candidates.append([60 * (i + 1) for i in range(n)])
    # Powers of 2 times small factor
    candidates.append([2 ** i for i in range(1, n + 1)])
    candidates.append([3 * (2 ** i) for i in range(1, n + 1)])
    
    # Evaluate candidates
    for cand in candidates:
        if len(set(cand)) < n or max(cand) > max_val or min(cand) < 1:
            continue
        primes = evaluate(cand)
        if len(primes) < best_count:
            best_count = len(primes)
            best_primes = primes
            best_set = list(cand)
    
    # If we haven't found something good, do local search
    if best_count > 5:  # We expect to find <= 5 primes
        # Random restarts with local search
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            
            # Generate candidate: all multiples of a random M
            M = rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120])
            max_m = max_val // M
            if max_m < n:
                continue
            m_vals = rng.sample(range(1, max_m + 1), n)
            cand = [M * m for m in m_vals]
            
            primes = evaluate(cand)
            if len(primes) < best_count:
                best_count = len(primes)
                best_primes = primes
                best_set = list(cand)
            
            # Local search: try to replace one element
            current = list(cand)
            current_primes = primes
            current_count = len(primes)
            
            for step in range(50):
                if time.time() > deadline - 5:
                    break
                improved = False
                for i in range(n):
                    if time.time() > deadline - 5:
                        break
                    old_val = current[i]
                    # Try values near old_val or multiples of same M
                    for trial in range(10):
                        if M > 1 and rng.random() < 0.5:
                            new_m = rng.randint(1, max_m)
                            new_val = M * new_m
                        else:
                            new_val = rng.randint(1, max_val)
                        
                        if new_val == old_val or new_val in current:
                            continue
                        
                        new_current = list(current)
                        new_current[i] = new_val
                        new_primes = evaluate(new_current)
                        
                        if len(new_primes) < current_count:
                            current = new_current
                            current_primes = new_primes
                            current_count = len(new_primes)
                            improved = True
                            if current_count < best_count:
                                best_count = current_count
                                best_primes = current_primes
                                best_set = list(current)
                            break
                    if improved:
                        break
                
                if not improved:
                    break
    
    # Final local search from best
    if time.time() < deadline - 15 and best_set:
        current = list(best_set)
        current_primes = evaluate(current)
        current_count = len(current_primes)
        
        for step in range(100):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                old_val = current[i]
                for trial in range(20):
                    new_val = rng.randint(1, max_val)
                    if new_val == old_val or new_val in current:
                        continue
                    
                    new_current = list(current)
                    new_current[i] = new_val
                    new_primes = evaluate(new_current)
                    
                    if len(new_primes) < current_count:
                        current = new_current
                        current_primes = new_primes
                        current_count = len(new_primes)
                        improved = True
                        if current_count < best_count:
                            best_count = current_count
                            best_primes = current_primes
                            best_set = list(current)
                        break
                if improved:
                    break
            
            if not improved:
                break
    
    return best_set
# EVOLVE-BLOCK-END
