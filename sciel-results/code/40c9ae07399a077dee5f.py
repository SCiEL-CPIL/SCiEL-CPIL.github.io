"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    if n % 2 == 0:
        factors.add(2)
        while n % 2 == 0:
            n //= 2
    if n % 3 == 0:
        factors.add(3)
        while n % 3 == 0:
            n //= 3
    f = 5
    while f * f <= n:
        if n % f == 0:
            factors.add(f)
            while n % f == 0:
                n //= f
        f += 2
    if n > 1:
        factors.add(n)
    return factors

def _count_distinct_primes(nums):
    """Count distinct primes dividing all pairwise sums of distinct elements."""
    n = len(nums)
    all_primes = set()
    for i in range(n):
        for j in range(i + 1, n):
            s = nums[i] + nums[j]
            all_primes |= _prime_factors(s)
    return len(all_primes)

def _count_distinct_primes_k(ks):
    """Count distinct primes dividing all pairwise sums of 6*k_i.
    Since 6*k_i + 6*k_j = 6*(k_i + k_j), primes = {2,3} union primes of (k_i+k_j).
    """
    n = len(ks)
    all_primes = {2, 3}
    for i in range(n):
        for j in range(i + 1, n):
            s = ks[i] + ks[j]
            # Factor s, add primes other than 2,3
            if s % 2 == 0:
                while s % 2 == 0:
                    s //= 2
            if s % 3 == 0:
                while s % 3 == 0:
                    s //= 3
            f = 5
            while f * f <= s:
                if s % f == 0:
                    all_primes.add(f)
                    while s % f == 0:
                        s //= f
                f += 2
            if s > 1:
                all_primes.add(s)
    return len(all_primes)

def _local_search(ks, deadline, rng, max_iters=50000):
    """Local search on k values to minimize distinct prime count."""
    n = len(ks)
    best_score = _count_distinct_primes_k(ks)
    best_ks = list(ks)
    max_k = 1666  # 10000 // 6
    
    # Precompute: for each pair (i,j), the sum k_i + k_j
    # We'll do simulated annealing / hill climbing
    
    current = list(ks)
    current_score = best_score
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick a random index to modify
        idx = rng.randint(0, n - 1)
        old_val = current[idx]
        
        # Generate a new value: random in [1, max_k] or local perturbation
        if rng.random() < 0.3:
            # Small perturbation
            delta = rng.randint(-3, 3)
            new_val = old_val + delta
        else:
            new_val = rng.randint(1, max_k)
        
        if new_val < 1 or new_val > max_k:
            continue
        
        # Check distinctness
        if new_val in current and new_val != old_val:
            continue
        
        current[idx] = new_val
        new_score = _count_distinct_primes_k(current)
        
        if new_score < best_score:
            best_score = new_score
            best_ks = list(current)
        elif new_score > current_score:
            # Revert (simple hill climbing)
            current[idx] = old_val
        else:
            # Same score, keep it (allows wandering)
            current_score = new_score
        
        current_score = new_score
    
    return best_ks, best_score

def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    rng = random.Random(42)
    
    n = 12
    max_k = 1666  # 10000 // 6
    
    best_ks = None
    best_score = float('inf')
    
    # Strategy 1: Try small k values first (small sums have fewer prime factors)
    # Initial: use the first 12 values
    initial_ks = list(range(1, 13))
    best_ks, best_score = _local_search(initial_ks, deadline, rng, max_iters=20000)
    
    # Strategy 2: Try k values that are all even (so k_i + k_j is even, 
    # meaning 2 divides the reduced sum, but we already have 2 from the 6 factor)
    # Actually, let's try k values that avoid introducing new primes.
    
    # Strategy 3: Multiple random restarts
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Random initial population biased toward small values
        ks = []
        while len(ks) < n:
            # Biased toward small values
            if rng.random() < 0.5:
                v = rng.randint(1, 20)
            else:
                v = rng.randint(1, max_k)
            if v not in ks:
                ks.append(v)
        
        cur_ks, cur_score = _local_search(ks, deadline, rng, max_iters=10000)
        
        if cur_score < best_score:
            best_score = cur_score
            best_ks = cur_ks
    
    # Strategy 4: Try structured sets - k values that are all multiples of some small number
    # If all k_i are multiples of m, then k_i + k_j is multiple of m
    # This doesn't help directly since we need to minimize primes of k_i+k_j
    
    # Strategy 5: Try k values from {1,2,3,...} with specific patterns
    # e.g., all k_i in a range where sums have limited prime factors
    
    # Try: k values that are powers of 2 or small primes
    # k = 1,2,3,4,5,6,7,8,9,10,11,12 -> sums range 3..23
    # Primes in 3..23: 2,3,5,7,11,13,17,19,23 -> 9 primes + {2,3} already counted
    # Wait, 2 and 3 are already in our base set. So additional primes from sums 3..23
    # that aren't 2 or 3: 5,7,11,13,17,19,23 -> 7 additional, total 9
    
    # Try to find k values where all pairwise sums only use primes from a small set
    # Target: can we get all sums to only have primes {2,3,5,7}?
    # That means k_i + k_j should only have prime factors 2,3,5,7
    # Numbers with only primes 2,3,5,7 are "7-smooth" numbers
    # 7-smooth numbers up to 3332 (max sum = 2*1666):
    # Let's find k values such that all pairwise sums are 7-smooth (or have limited primes)
    
    # Precompute 7-smooth numbers (primes only 2,3,5,7) up to 3332
    smooth_7 = set()
    for a in range(0, 13):  # 2^a
        for b in range(0, 8):  # 3^b
            for c in range(0, 5):  # 5^c
                for d in range(0, 4):  # 7^d
                    v = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if v <= 3332:
                        smooth_7.add(v)
    
    # Find 12 distinct k values such that all pairwise sums are in smooth_7
    # This is a constraint satisfaction problem
    # Let's try: pick k values from a set where differences work out
    
    # Alternative: use greedy - pick k values one by one, ensuring all pairwise sums with already-picked values are 7-smooth
    if time.time() < deadline:
        for trial in range(200):
            if time.time() > deadline:
                break
            # Start with a random small k
            start = rng.randint(1, 50)
            ks = [start]
            for _ in range(n - 1):
                # Try to find a new k that keeps all pairwise sums 7-smooth
                found = False
                candidates = list(range(1, 200))
                rng.shuffle(candidates)
                for cand in candidates:
                    if cand in ks:
                        continue
                    ok = True
                    for existing in ks:
                        if (cand + existing) not in smooth_7:
                            ok = False
                            break
                    if ok:
                        ks.append(cand)
                        found = True
                        break
                if not found:
                    break
            
            if len(ks) == n:
                score = _count_distinct_primes_k(ks)
                if score < best_score:
                    best_score = score
                    best_ks = list(ks)
                    # Local search to improve
                    cur_ks, cur_score = _local_search(ks, deadline, rng, max_iters=5000)
                    if cur_score < best_score:
                        best_score = cur_score
                        best_ks = cur_ks
    
    # Strategy 6: Try k values that are all the same parity and small
    # If all k_i are odd, k_i + k_j is even -> factor 2 (already have it)
    # If all k_i are even, k_i + k_j is even -> factor 2 (already have it)
    # Doesn't reduce primes beyond what we have
    
    # Final: convert best k values to actual numbers
    if best_ks is None:
        best_ks = list(range(1, 13))
    
    result = [6 * k for k in best_ks]
    
    # Verify
    assert len(result) == n
    assert len(set(result)) == n
    assert all(1 <= x <= 10000 for x in result)
    
    return result
# EVOLVE-BLOCK-END
