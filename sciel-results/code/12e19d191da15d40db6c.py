"""
Optimized search for large sets with distinct subset products.
Uses fast incremental subset product checking and high-throughput randomized greedy restarts.
"""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt
from collections import Counter

P = {'n': 40, 'maximum_size': 18}

def _is_prime(a):
    if a < 2:
        return False
    if a < 4:
        return True
    if a % 2 == 0 or a % 3 == 0:
        return False
    d = 5
    while d * d <= a:
        if a % d == 0 or a % (d + 2) == 0:
            return False
        d += 6
    return True

def _prime_factors(a):
    """Return dict of prime -> exponent for integer a."""
    factors = {}
    d = 2
    while d * d <= a:
        while a % d == 0:
            factors[d] = factors.get(d, 0) + 1
            a //= d
        d += 1 if d == 2 else 2
    if a > 1:
        factors[a] = factors.get(a, 0) + 1
    return factors

def _largest_prime_factor(n):
    lpf = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            lpf = d
            n //= d
        d += 1
    if n > 1:
        lpf = n
    return lpf

def _check_valid(S):
    """
    Validate that all subset products of S are distinct.
    """
    current = {1}
    for x in S:
        new_vals = {p * x for p in current}
        if current & new_vals:
            return False
        current |= new_vals
    return True

def _try_greedy_fast(order, candidates_set):
    """
    Greedily add elements from order to build a valid set.
    Uses incremental product set update for speed.
    """
    current_set = []
    prods = {1}
    
    for v in order:
        if v in candidates_set:
            candidates_set.remove(v) # Optimization: remove from available pool? 
            # Actually, we just need to check if v is already in current_set.
            # The order might contain duplicates or we might want to reuse the order structure.
            # But typically order is a permutation. Let's stick to simple check.
            pass
        
        # Check if v can be added
        # We iterate over current prods. If any v*p is already in prods, it's a conflict.
        # Note: We must check against the *current* prods before adding new ones.
        valid = True
        new_prods_list = []
        for p in prods:
            np = v * p
            if np in prods:
                valid = False
                break
            new_prods_list.append(np)
        
        if valid:
            current_set.append(v)
            prods.update(new_prods_list)
            
    return current_set

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    candidates = list(range(2, P['n'] + 1))
    primes = [a for a in candidates if _is_prime(a)]
    composites = [a for a in candidates if not _is_prime(a)]
    
    # Pre-compute properties
    factor_cache = {a: _prime_factors(a) for a in candidates}
    lpf_cache = {a: _largest_prime_factor(a) for a in candidates}
    
    # Count prime frequency in composites to determine "rarity"
    prime_freq = Counter()
    for c in composites:
        for p in factor_cache[c]:
            prime_freq[p] += 1
    
    def _rarity_score(a):
        """
        Higher score = rarer prime factors = better candidate for early addition.
        """
        f = factor_cache[a]
        score = 0.0
        for p, e in f.items():
            # Rarer primes (lower freq) get higher score.
            # Higher exponents are "worse" (more likely to collide), so penalize.
            score += 1.0 / (prime_freq.get(p, 1) * e)
        return score

    # Initial best: Primes are always a valid set (subset products of distinct primes are unique by FUD)
    best_set = primes[:]
    best_size = len(best_set)
    
    # Define several deterministic ordering strategies
    # 1. Primes + Composites sorted by rarity (desc)
    composites_rarity_desc = sorted(composites, key=_rarity_score, reverse=True)
    order_rarity = primes + composites_rarity_desc
    
    # 2. Primes + Composites sorted by largest prime factor (desc)
    composites_lpf_desc = sorted(composites, key=lambda x: lpf_cache[x], reverse=True)
    order_lpf = primes + composites_lpf_desc
    
    # 3. Primes + Composites sorted by value (asc)
    order_val_asc = primes + sorted(composites)
    
    # 4. Primes + Composites sorted by value (desc)
    order_val_desc = primes + sorted(composites, reverse=True)
    
    # Run deterministic strategies first
    for order in [order_rarity, order_lpf, order_val_asc, order_val_desc]:
        sol = _try_greedy_fast(order, set(candidates))
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol[:]
    
    # Randomized Restart Greedy
    # The key is to run many fast restarts.
    rng = random.Random(42)
    
    # Pre-compute a "weighted" order for shuffling?
    # We can shuffle the composites part, keeping primes fixed or slightly shuffled.
    # Strategy: 
    # 1. Keep primes at the front (they are safest).
    # 2. Shuffle composites using a biased shuffle or just uniform shuffle.
    # 3. Occasionally shuffle primes too.
    
    # To maximize throughput, we avoid complex local search.
    # We just run _try_greedy_fast thousands of times.
    
    # Optimization: 
    # If we have a set of size k, the product set has 2^k elements.
    # Checking a new element v involves checking 2^k products.
    # For k=15, that's 32768 checks. Very fast.
    # For k=18, that's 262144 checks. Still fast in Python if optimized.
    
    # Let's estimate iterations.
    # If each iteration takes ~0.5ms (optimistic for Python sets of size 2^15+), we can do ~100k in 50s.
    
    # We will mix strategies:
    # 50%: Primes fixed, Composites shuffled
    # 30%: Primes shuffled, Composites shuffled
    # 20%: All candidates shuffled
    
    while time.time() < deadline:
        r = rng.random()
        
        if r < 0.5:
            # Primes fixed, Composites shuffled
            order = primes + composites[:]
            rng.shuffle(order[len(primes):])
        elif r < 0.8:
            # Primes shuffled, Composites shuffled
            order = primes + composites[:]
            rng.shuffle(order)
        else:
            # All shuffled
            order = candidates[:]
            rng.shuffle(order)
            
        # Optional: Bias the shuffle by sorting by rarity first, then shuffling within groups?
        # Or just use the pre-sorted lists and shuffle partially?
        # Let's try a "soft" shuffle: sort by rarity, then randomly swap some elements.
        # Actually, pure random shuffle of composites is a good baseline.
        # Let's add a strategy: Sort by rarity, then reverse? No, that's deterministic.
        
        # Let's try: Sort by rarity, then apply a "temperature" shuffle?
        # Too complex. Stick to uniform shuffles of the composite part.
        
        sol = _try_greedy_fast(order, set(candidates))
        
        if len(sol) > best_size:
            best_size = len(sol)
            best_set = sol[:]
            
            # If we reach a very high size, we can try to polish?
            # But the directive says to keep the design. Greedy is the design.
            # We can stop early if we hit the theoretical max or a very high threshold?
            # Max distinct integers in 2..40 is 39. But dissociated set size is much smaller.
            # Primes up to 40: 12 primes.
            # Adding composites:
            # 4, 8, 16, 32 are powers of 2. Only one can be in a dissociated set? 
            # No, 4 and 8: products 1, 4, 8, 32. Distinct. 
            # 4, 9: 1, 4, 9, 36. Distinct.
            # 4, 9, 25: 1, 4, 9, 25, 36, 100, 225, 900. Distinct.
            # So we can have multiple squares of primes.
            
            # If best_size >= 17, we are likely near optimal. 
            # We continue to see if 18 is possible.
            
    # Final validation
    if not _check_valid(best_set):
        # Fallback to primes
        best_set = primes[:]
        
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
