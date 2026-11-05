import itertools
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill
    
    def pairwise_coprime(vals):
        """Check if all distinct values are pairwise coprime."""
        distinct = list(set(vals))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if math.gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    # Precompute gcd table for values up to 256
    max_sum = 256
    gcd_table = [[0] * (max_sum + 1) for _ in range(max_sum + 1)]
    for i in range(1, max_sum + 1):
        for j in range(i, max_sum + 1):
            gcd_table[i][j] = math.gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]
    
    def fast_pairwise_coprime(sums_set):
        """Fast check using precomputed gcd table."""
        distinct = list(sums_set)
        n = len(distinct)
        for i in range(n):
            vi = distinct[i]
            for j in range(i + 1, n):
                vj = distinct[j]
                if gcd_table[vi][vj] != 1:
                    return False
        return True
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    # Precompute primes in [2, 256]
    primes = [p for p in range(2, 257) if is_prime(p)]
    
    # Strategy: For 9 distinct sums that are pairwise coprime,
    # the most likely candidates are primes (or 1, but sums are >= 2).
    # Key insight: if all sums are odd, then one set must be all-odd and the other all-even
    # (odd+even=odd). This halves the search space.
    
    # Phase 1: Try to find A (all odd) and B (all even) such that all 9 sums are distinct primes
    # This gives 9 distinct pairwise coprime sums.
    
    # Generate odd numbers in [1,128] and even numbers in [1,128]
    odds = [x for x in range(1, 129) if x % 2 == 1]
    evens = [x for x in range(1, 129) if x % 2 == 0]
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_count = 5
    
    # Try small odd sets first (more likely to produce small prime sums)
    # Limit to first 30 odds and first 30 evens for initial search
    odds_small = odds[:30]
    evens_small = evens[:30]
    
    A_candidates = list(itertools.combinations(odds_small, 3))
    B_candidates = list(itertools.combinations(evens_small, 3))
    
    checked = 0
    for A in A_candidates:
        if time.time() > deadline:
            break
        a1, a2, a3 = A
        for B in B_candidates:
            if time.time() > deadline:
                break
            b1, b2, b3 = B
            # Compute 9 sums
            s = (a1+b1, a1+b2, a1+b3, a2+b1, a2+b2, a2+b3, a3+b1, a3+b2, a3+b3)
            sums_set = set(s)
            count = len(sums_set)
            if count <= best_count:
                continue
            # Check pairwise coprime
            if fast_pairwise_coprime(sums_set):
                best_count = count
                best_A = list(A)
                best_B = list(B)
                if best_count == 9:
                    return [best_A, best_B]
            checked += 1
    
    # Phase 2: Broader search with all odds/evens but limited by time
    if time.time() < deadline - 10:
        # Use random sampling with the parity constraint
        import random
        random.seed(42)
        
        # Try random odd/even combinations
        for _ in range(1000000):
            if time.time() > deadline:
                break
            A = random.sample(odds, 3)
            A.sort()
            B = random.sample(evens, 3)
            B.sort()
            sums_set = set(a + b for a in A for b in B)
            count = len(sums_set)
            if count <= best_count:
                continue
            if fast_pairwise_coprime(sums_set):
                best_count = count
                best_A = A
                best_B = B
                if best_count == 9:
                    return [best_A, best_B]
    
    # Phase 3: If still not 9, try without parity constraint (mixed parity)
    if time.time() < deadline - 5:
        import random
        random.seed(123)
        all_nums = list(range(1, 129))
        for _ in range(500000):
            if time.time() > deadline:
                break
            A = random.sample(all_nums, 3)
            A.sort()
            B = random.sample(all_nums, 3)
            B.sort()
            sums_set = set(a + b for a in A for b in B)
            count = len(sums_set)
            if count <= best_count:
                continue
            if fast_pairwise_coprime(sums_set):
                best_count = count
                best_A = A
                best_B = B
                if best_count == 9:
                    return [best_A, best_B]
    
    return [best_A, best_B]
