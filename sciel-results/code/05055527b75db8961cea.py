"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import math

# EVOLVE-BLOCK-START

def _is_prime(n):
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

def _valid_sumset(A, B):
    """Check if all distinct values in A+B are pairwise coprime."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    # Check pairwise coprime
    slist = sorted(sums)
    for i in range(len(slist)):
        for j in range(i + 1, len(slist)):
            if math.gcd(slist[i], slist[j]) != 1:
                return False
    return True

def _score(A, B):
    """Return the number of distinct values in A+B (the objective)."""
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    return len(sums)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (limit is 90s)
    
    # Precompute primes in [2, 256]
    primes = [p for p in range(2, 257) if _is_prime(p)]
    prime_set = set(primes)
    
    # Strategy: try to find A, B such that all 9 sums are distinct primes.
    # This guarantees pairwise coprimality (distinct primes are pairwise coprime)
    # and maximizes cardinality at 9.
    
    # Approach 1: Systematic search with pruning
    # We need A = {a1, a2, a3}, B = {b1, b2, b3}
    # All 9 sums a_i + b_j must be distinct primes.
    
    # To make this tractable, we can fix some structure.
    # Note: if all sums are primes, they must be odd (except possibly 2).
    # Since a_i, b_j >= 1, sums >= 2.
    # If a sum is 2, then a_i=1, b_j=1.
    # For sums to be odd primes, a_i and b_j must have opposite parity (one odd, one even),
    # OR one of them is such that the sum is 2.
    
    # Let's try: make all elements of A odd and all elements of B even, or vice versa.
    # Then all sums are odd, so they can be odd primes.
    
    # Let A be odd numbers, B be even numbers.
    odd_candidates = [x for x in range(1, 129) if x % 2 == 1]
    even_candidates = [x for x in range(2, 129) if x % 2 == 0]
    
    # Also try A even, B odd
    # And mixed parity with careful selection
    
    best_score = 5  # seed gives 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # First, quick check: can we find a 9-prime sumset?
    # Try structured search: fix B, search A
    
    # Let's be smarter. The sums are a_i + b_j.
    # If we fix B = {b1, b2, b3}, then for each a, the three sums a+b1, a+b2, a+b3 must be primes.
    # And across different a's, all 9 must be distinct.
    
    # Precompute: for each a in [1,128], which b's give prime sums?
    # Actually, let's just do a bounded search.
    
    # Strategy: iterate over B triples from even numbers (or odd), then for each B,
    # find A triples such that all 9 sums are distinct primes.
    
    # To limit search: use only a subset of candidates.
    # Primes in [2,256] are ~55 primes.
    
    # Let's try: for B in combinations of even numbers, check if there exist 3 distinct odd A's
    # such that all a+b are prime and all 9 are distinct.
    
    # This is still large. Let's use a heuristic: 
    # Pick B first, then for each candidate a, check if a+b1, a+b2, a+b3 are all prime.
    # Collect such a's, then pick 3 from them that give distinct sums.
    
    # Limit B search to first N even numbers to keep it fast.
    
    # Actually, let's try a different approach: random restarts with local search.
    
    import random
    random.seed(42)
    
    # Try to find 9-prime solution with structured search
    # Fix B to be 3 even numbers, search A from odd numbers
    
    # Precompute for speed: for each even b, and odd a, is a+b prime?
    # a+b ranges from 1+2=3 to 127+128=255... wait, max is 128+128=256
    # But a is odd (max 127), b is even (max 128), so max sum = 127+128 = 255
    # Min sum = 1+2 = 3
    
    # Let's precompute a prime lookup table
    is_prime_table = [False] * 257
    for p in primes:
        is_prime_table[p] = True
    
    # Search: try B triples from a limited set of even numbers
    # To make it fast, try B where elements are small-ish
    
    # Let's try a systematic but bounded approach
    # Iterate over B triples from even numbers in [2, 60]
    # For each B, find all a in odd [1, 127] such that a+b1, a+b2, a+b3 are all prime
    # Then pick 3 such a's with all distinct sums
    
    found_9 = False
    search_B_range = list(range(2, 61, 2))  # even numbers 2 to 60
    
    for B in itertools.combinations(search_B_range, 3):
        if time.time() > deadline:
            break
        b1, b2, b3 = B
        valid_a = []
        for a in range(1, 128, 2):  # odd numbers
            s1 = a + b1
            s2 = a + b2
            s3 = a + b3
            if s1 <= 256 and s2 <= 256 and s3 <= 256:
                if is_prime_table[s1] and is_prime_table[s2] and is_prime_table[s3]:
                    valid_a.append((a, s1, s2, s3))
        
        # Now pick 3 distinct a's from valid_a such that all 9 sums are distinct
        if len(valid_a) >= 3:
            for A_combo in itertools.combinations(valid_a, 3):
                a1, s11, s12, s13 = A_combo[0]
                a2, s21, s22, s23 = A_combo[1]
                a3, s31, s32, s33 = A_combo[2]
                all_sums = [s11, s12, s13, s21, s22, s23, s31, s32, s33]
                if len(set(all_sums)) == 9:
                    # Found a 9-prime sumset!
                    A = [a1, a2, a3]
                    # Verify distinctness of A
                    if len(set(A)) == 3 and len(set(B)) == 3:
                        return A, B
                    found_9 = True
                    break
            if found_9:
                break
    
    if time.time() > deadline:
        return best_A, best_B
    
    # If no 9-prime found, try to maximize distinct sums with pairwise coprimality
    # Use random search for better solutions than the seed
    
    # Random search with local improvement
    def random_search(iterations):
        nonlocal best_score, best_A, best_B
        for _ in range(iterations):
            if time.time() > deadline:
                break
            # Random A and B
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
            if _valid_sumset(A, B):
                s = _score(A, B)
                if s > best_score:
                    best_score = s
                    best_A = A[:]
                    best_B = B[:]
                    if best_score == 9:
                        return True
        return best_score == 9
    
    # Do some random searches
    for _ in range(5000):
        if time.time() > deadline:
            break
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        if _valid_sumset(A, B):
            s = _score(A, B)
            if s > best_score:
                best_score = s
                best_A = A[:]
                best_B = B[:]
                if best_score == 9:
                    return best_A, best_B
    
    # Local search: try to improve from best_A, best_B
    if best_score < 9:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try single-element changes
            for idx in range(3):
                for new_val in range(1, 129):
                    if time.time() > deadline:
                        break
                    if new_val in best_A and best_A[idx] != new_val:
                        continue
                    A_new = best_A[:]
                    A_new[idx] = new_val
                    if len(set(A_new)) == 3:
                        if _valid_sumset(A_new, best_B):
                            s = _score(A_new, best_B)
                            if s > best_score:
                                best_score = s
                                best_A = A_new
                                improved = True
                                if best_score == 9:
                                    return best_A, best_B
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
            for idx in range(3):
                for new_val in range(1, 129):
                    if time.time() > deadline:
                        break
                    if new_val in best_B and best_B[idx] != new_val:
                        continue
                    B_new = best_B[:]
                    B_new[idx] = new_val
                    if len(set(B_new)) == 3:
                        if _valid_sumset(best_A, B_new):
                            s = _score(best_A, B_new)
                            if s > best_score:
                                best_score = s
                                best_B = B_new
                                improved = True
                                if best_score == 9:
                                    return best_A, best_B
                if time.time() > deadline:
                    break
    
    return best_A, best_B

# EVOLVE-BLOCK-END
