import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Precompute smallest prime factor sieve up to 20000
    def sieve_spf(limit):
        spf = list(range(limit + 1))
        for i in range(2, int(math.isqrt(limit)) + 1):
            if spf[i] == i:  # i is prime
                for j in range(i * i, limit + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        return spf

    spf = sieve_spf(20000)

    def get_prime_factors(n):
        """Get set of prime factors of n using SPF table."""
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    def get_primes_for_sums(nums):
        """Get union of all prime factors of pairwise sums."""
        all_primes = set()
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                s = nums[i] + nums[j]
                if s <= 20000:
                    all_primes.update(get_prime_factors(s))
        return all_primes

    def count_primes(nums):
        return len(get_primes_for_sums(nums))

    # Precompute smooth number lookup tables for various prime sets
    # A number is P-smooth if all its prime factors are in P

    def build_smooth_set(P, limit):
        """Build set of all P-smooth numbers up to limit."""
        P_set = set(P)
        smooth = set()
        # Generate all P-smooth numbers <= limit
        def gen(idx, current):
            if current > limit:
                return
            smooth.add(current)
            if idx < len(P):
                p = P[idx]
                val = current
                while val * p <= limit:
                    val *= p
                    gen(idx, val)
                gen(idx + 1, current)
        
        gen(0, 1)
        return smooth

    # Try to find good constructions
    best_set = list(range(1, 13))
    best_count = count_primes(best_set)

    # Strategy 1: Try to find 12 numbers where all pairwise sums are 2,3-smooth
    # Then scale by power of 2
    # 2,3-smooth numbers: 1,2,3,4,6,8,9,12,16,18,24,27,32,36,48,54,64,72,81,96,108,128,144,162,192,216,243,256,288,324,384,432,486,512,576,648,729,768,864,972,1024,...

    # For 2,3-smooth sums: if a_i and a_j are both 2,3-smooth, a_i + a_j may not be.
    # But if we pick a_i from a special structure...
    
    # Let's try: all a_i are powers of 2. Sums are not smooth.
    # Better: pick a_i such that a_i + a_j is always 2,3-smooth.
    
    # Try brute force for small sets with 2,3,5-smooth sums
    prime_sets_to_try = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
    ]

    for P in prime_sets_to_try:
        if time.time() > deadline - 15:
            break
        
        P_set = set(P)
        smooth_nums = build_smooth_set(P, 500)
        smooth_list = sorted(smooth_nums)
        
        # Filter to positive integers >= 1
        smooth_list = [x for x in smooth_list if x >= 1]
        
        if len(smooth_list) < 12:
            continue
        
        # Try to find 12 distinct values from smooth_list such that all pairwise sums are also P-smooth
        # This is a constraint satisfaction problem
        # Use backtracking with pruning
        
        def can_add(candidate, current, P_set, smooth_set_lookup):
            for j in current:
                s = candidate + j
                if s > 500:
                    return False
                if s not in smooth_set_lookup:
                    return False
            return True
        
        smooth_set_lookup = set(smooth_list)
        
        found = None
        def backtrack(start_idx, current):
            nonlocal found
            if found is not None:
                return
            if len(current) == 12:
                found = list(current)
                return
            if time.time() > deadline - 20:
                return
            
            # Try candidates from smooth_list starting at start_idx
            for idx in range(start_idx, min(start_idx + 50, len(smooth_list))):
                cand = smooth_list[idx]
                if can_add(cand, current, P_set, smooth_set_lookup):
                    current.append(cand)
                    backtrack(idx + 1, current)
                    current.pop()
                    if found is not None:
                        return
        
        current = []
        backtrack(0, current)
        
        if found:
            A = found
            max_a = max(A)
            # Scale by largest K such that K * max_a <= 10000
            K = 10000 // max_a
            if K < 1:
                K = 1
            X = [k * a for k, a in zip([K] * 12, A)]
            
            if len(set(X)) == 12 and all(1 <= x <= 10000 for x in X):
                c = count_primes(X)
                if c < best_count:
                    best_count = c
                    best_set = X
                if best_count <= 3:
                    break

    # Strategy 2: Local search / hill climbing on the actual numbers
    if time.time() < deadline - 30:
        # Start from best_set and try to improve
        import random
        rng = random.Random(42)
        
        def local_search(init, max_iters=2000):
            current = list(init)
            current_cost = count_primes(current)
            best_local = list(current)
            best_local_cost = current_cost
            
            for it in range(max_iters):
                if time.time() > deadline - 25:
                    break
                # Pick a random element and try to change it
                i = rng.randint(0, 11)
                old_val = current[i]
                # Try nearby values or random values
                candidates = []
                # Nearby
                for delta in range(-50, 51):
                    nv = old_val + delta
                    if 1 <= nv <= 10000 and nv != old_val:
                        candidates.append(nv)
                # Random
                for _ in range(10):
                    nv = rng.randint(1, 10000)
                    if nv != old_val:
                        candidates.append(nv)
                
                improved = False
                for nv in candidates:
                    # Check distinctness
                    if any(current[j] == nv for j in range(12) if j != i):
                        continue
                    current[i] = nv
                    new_cost = count_primes(current)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                        if new_cost < best_local_cost:
                            best_local_cost = new_cost
                            best_local = list(current)
                        break
                    else:
                        current[i] = old_val
                
                if not improved:
                    # Accept some worse moves (simulated annealing lite)
                    if rng.random() < 0.1:
                        nv = rng.choice(candidates) if candidates else old_val
                        if not any(current[j] == nv for j in range(12) if j != i):
                            current[i] = nv
                            current_cost = count_primes(current)
            
            return best_local, best_local_cost
        
        improved_set, improved_cost = local_search(best_set)
        if improved_cost < best_count:
            best_count = improved_cost
            best_set = improved_set

    # Final verification
    assert len(set(best_set)) == 12, f"Not distinct: {best_set}"
    assert all(1 <= x <= 10000 for x in best_set), f"Out of bounds: {best_set}"
    
    return best_set
