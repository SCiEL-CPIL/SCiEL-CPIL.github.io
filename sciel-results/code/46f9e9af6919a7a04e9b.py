"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    if n <= 1:
        return factors
    if n % 2 == 0:
        factors.add(2)
        while n % 2 == 0:
            n //= 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 2
    if n > 1:
        factors.add(n)
    return factors

def _score(nums):
    """Count distinct prime factors across all pairwise sums."""
    primes = set()
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            primes |= _prime_factors(nums[i] + nums[j])
    return len(primes)

def _score_incr(nums, idx, old_val, new_val, prime_set):
    """
    Compute the change in score when replacing nums[idx] from old_val to new_val.
    Returns (new_score, new_prime_set) where new_prime_set is the updated set.
    """
    n = len(nums)
    # Remove old contributions
    old_set = set(prime_set)
    for j in range(n):
        if j == idx:
            continue
        s_old = old_val + nums[j]
        old_factors = _prime_factors(s_old)
        for p in old_factors:
            old_set.discard(p)
    
    # Add new contributions
    new_set = set(old_set)
    for j in range(n):
        if j == idx:
            continue
        s_new = new_val + nums[j]
        new_factors = _prime_factors(s_new)
        new_set |= new_factors
    
    return len(new_set), new_set

def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin before 90s kill
    
    best = list(range(1, 13))
    best_score = _score(best)
    
    # Precompute prime factor cache for small numbers
    pf_cache = {}
    def get_pf(n):
        if n not in pf_cache:
            pf_cache[n] = _prime_factors(n)
        return pf_cache[n]
    
    # Phase 1: Try structured families
    # Family: arithmetic progressions
    for a in range(1, 200):
        for d in range(1, 200):
            if time.time() > deadline:
                break
            nums = [a + i * d for i in range(12)]
            if any(x > 10000 or x <= 0 for x in nums):
                continue
            sc = _score(nums)
            if sc < best_score:
                best_score = sc
                best = nums[:]
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        return best
    
    # Family: multiples of small bases with consecutive multipliers
    for base in range(2, 100):
        if time.time() > deadline:
            break
        for start_mult in range(1, 20):
            if time.time() > deadline:
                break
            nums = [base * (start_mult + i) for i in range(12)]
            if any(x > 10000 or x <= 0 for x in nums):
                continue
            sc = _score(nums)
            if sc < best_score:
                best_score = sc
                best = nums[:]
    
    if time.time() > deadline:
        return best
    
    # Family: numbers of form 2^a * 3^b
    candidates_23 = set()
    for a in range(0, 14):
        for b in range(0, 14):
            val = (2 ** a) * (3 ** b)
            if 1 <= val <= 10000:
                candidates_23.add(val)
    candidates_23 = sorted(candidates_23)
    if len(candidates_23) >= 12:
        # Try first 12
        nums = candidates_23[:12]
        sc = _score(nums)
        if sc < best_score:
            best_score = sc
            best = nums[:]
        
        # Try last 12
        nums = candidates_23[-12:]
        sc = _score(nums)
        if sc < best_score:
            best_score = sc
            best = nums[:]
    
    if time.time() > deadline:
        return best
    
    # Family: numbers all divisible by a common factor
    for g in range(2, 50):
        if time.time() > deadline:
            break
        # Pick 12 distinct multiples
        for start in range(1, 10):
            if time.time() > deadline:
                break
            nums = [g * (start + i) for i in range(12)]
            if any(x > 10000 or x <= 0 for x in nums):
                continue
            sc = _score(nums)
            if sc < best_score:
                best_score = sc
                best = nums[:]
    
    if time.time() > deadline:
        return best
    
    # Phase 2: Local search with incremental scoring
    if time.time() < deadline - 5:
        random.seed(42)
        current = best[:]
        current_score = best_score
        current_primes = set()
        n = len(current)
        for i in range(n):
            for j in range(i + 1, n):
                current_primes |= get_pf(current[i] + current[j])
        
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline - 2 and no_improve < max_no_improve:
            i = random.randint(0, 11)
            old_val = current[i]
            
            # Try multiple replacement strategies
            improved = False
            for strategy in range(4):
                if time.time() > deadline - 2:
                    break
                
                if strategy == 0:
                    # Random value
                    new_val = random.randint(1, 10000)
                elif strategy == 1:
                    # Value near current
                    delta = random.randint(-50, 50)
                    new_val = old_val + delta
                    if new_val < 1:
                        new_val = 1
                    elif new_val > 10000:
                        new_val = 10000
                elif strategy == 2:
                    # Value that's a multiple of a small number
                    base = random.choice([2, 3, 4, 5, 6, 10, 12, 20, 30])
                    mult = random.randint(1, 10000 // base)
                    new_val = base * mult
                    if new_val > 10000:
                        new_val = 10000
                    elif new_val < 1:
                        new_val = 1
                else:
                    # Value that shares factors with existing numbers
                    existing = [v for v in current if v != old_val]
                    if existing:
                        base = random.choice(existing)
                        g = base // 2 if base % 2 == 0 else base // 3 if base % 3 == 0 else 2
                        mult = random.randint(1, 10000 // max(g, 1))
                        new_val = g * mult
                        if new_val > 10000:
                            new_val = 10000
                        elif new_val < 1:
                            new_val = 1
                    else:
                        new_val = random.randint(1, 10000)
                
                if new_val in current or new_val == old_val:
                    continue
                
                # Compute new score incrementally
                new_score, new_primes = _score_incr(current, i, old_val, new_val, current_primes)
                
                if new_score < current_score:
                    current[i] = new_val
                    current_score = new_score
                    current_primes = new_primes
                    no_improve = 0
                    improved = True
                    break
                else:
                    # Don't update current, just continue to next strategy
                    pass
            
            if not improved:
                no_improve += 1
        
        if current_score < best_score:
            best = current[:]
            best_score = current_score
    
    return best
# EVOLVE-BLOCK-END
