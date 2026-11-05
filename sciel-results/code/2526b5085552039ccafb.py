import itertools
from math import gcd
import time

def construct():
    deadline = time.time() + 85  # 5s safety margin
    
    # Precompute: for each possible sum value (2..256), store its prime factors as a frozenset
    # Actually, let's just use gcd checks since it's fast enough
    
    def pairwise_coprime(values):
        """Check if all distinct values in the list are pairwise coprime."""
        distinct = list(set(values))
        n = len(distinct)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(distinct[i], distinct[j]) != 1:
                    return False
        return True
    
    def score_pair(A, B):
        """Return (valid, count) for a pair of triples."""
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if len(distinct) != len(sums):
            # Has duplicates, but that's fine - we just need distinct to be pairwise coprime
            pass
        if not pairwise_coprime(distinct):
            return False, 0
        return True, len(distinct)
    
    best_score = 0
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    
    # Strategy: search over A first, then for each A, search over B
    # To make this tractable, use structured candidates
    
    # Generate candidate values: focus on numbers that produce coprime sums
    # Key insight: if A elements are all odd and B elements are all even (or vice versa),
    # sums have consistent parity. But we need pairwise coprime, so let's just search.
    
    # Precompute a good set of candidate values to reduce search space
    # Values that are prime or have few factors tend to work better for sums
    candidates = list(range(1, 129))
    
    # Phase 1: Try to find solutions with all 9 sums distinct and pairwise coprime
    # Iterate over A combinations (sorted for uniqueness)
    # To keep it tractable, limit the search
    
    # Use a smarter approach: fix A and find compatible B
    # For a given A = [a1, a2, a3], we need B = [b1, b2, b3] such that
    # all 9 values {a_i + b_j} are pairwise coprime
    
    # Let's try a structured search:
    # For each A, try all B combinations from a reduced candidate set
    
    # First, let's try small structured searches
    # Key observation: if we can get all 9 sums to be distinct primes or 1, that works
    
    # Try: A elements small, B elements such that sums are primes
    # Primes in range [2, 256]: let's generate them
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    primes = [p for p in range(2, 257) if is_prime(p)]
    
    # Strategy: For each A, try to find B where all sums are in a set of pairwise coprime numbers
    # Let's search over A from a reduced set first
    
    # Reduced candidate set for A and B: focus on values that give good sum properties
    # Let's try all A from combinations of a subset, and for each, all B from another subset
    
    # To keep within time, use a moderate search
    # Try A from first 60 values, B from first 60 values
    # That's C(60,3)^2 which is too large. Need smarter approach.
    
    # Better: iterate A, and for each A, greedily/search for B
    
    # Let's try: for each A (from a reasonable set), try all B from a candidate set
    # Candidate set for B: values 1-128
    
    # Even smarter: precompute for each pair (a, b) whether a+b is "compatible"
    # with existing sums
    
    # Let me try a direct but bounded search
    # Phase 1: Try A where elements are from [1..40], B from [1..128]
    # But that's still C(40,3) * C(128,3) ≈ 9880 * 341376 ≈ 3.4 billion - too large
    
    # Need much smarter approach. Let's use the pairwise coprime constraint heavily.
    
    # Key: for a fixed A, the 3 sums a1+b, a2+b, a3+b for a given b must be pairwise coprime
    # AND all 9 sums must be pairwise coprime.
    
    # Let's try: iterate over A from a small set, and for each A, iterate over b1, b2, b3
    # with early termination
    
    # Reduced A candidate set: let's try A where elements are spread out
    a_candidates = list(range(1, 51))  # 50 values
    b_candidates = list(range(1, 129))  # 128 values
    
    # For each A, we need to find B. Let's iterate over b1, b2, b3 with pruning.
    # For fixed A and b1: compute s1 = [a1+b1, a2+b1, a3+b1]
    # For b2: s2 = [a1+b2, a2+b2, a3+b2] must be pairwise coprime with each other AND with s1
    # For b3: s3 must be pairwise coprime with s1, s2, and itself
    
    # This is still large but with pruning should be manageable.
    
    # Let's limit: try A from combinations of [1..30], and for each, try B from [1..128]
    # C(30,3) = 4060 A's. For each, trying all B triples is too much.
    # Instead, for each A, try b1 from [1..128], b2 from [1..128], b3 from [1..128] with pruning.
    
    # Actually, let's be more strategic. Let's try specific structured A's first.
    
    # Try: A = [a, a+d, a+2d] (arithmetic progressions) and similar structures
    
    # Let me just do a bounded random + structured search
    
    import random
    random.seed(42)
    
    # Phase 1: Try the known solution and nearby
    # Phase 2: Structured search
    # Phase 3: Random search with deadline
    
    # First, verify the seed solution
    seed_A = [1, 7, 13]
    seed_B = [4, 10, 16]
    valid, cnt = score_pair(seed_A, seed_B)
    if valid:
        best_score = cnt
        best_A = seed_A[:]
        best_B = seed_B[:]
    
    # Now search for better
    # Try to find solutions with 9 distinct pairwise coprime sums
    
    # Structured search: try A with small spread
    for a1 in range(1, 30):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 40):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 50):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                
                # For this A, try to find B
                # Try b1 from 1 to 128, then b2, b3 with pruning
                for b1 in range(1, 129):
                    if time.time() > deadline:
                        break
                    s1 = [a1 + b1, a2 + b1, a3 + b1]
                    # Check s1 is pairwise coprime
                    if not pairwise_coprime(s1):
                        continue
                    
                    for b2 in range(1, 129):
                        if time.time() > deadline:
                            break
                        if b2 == b1:
                            continue
                        s2 = [a1 + b2, a2 + b2, a3 + b2]
                        # Check s2 is pairwise coprime
                        if not pairwise_coprime(s2):
                            continue
                        # Check all of s1 ∪ s2 is pairwise coprime
                        combined = s1 + s2
                        if not pairwise_coprime(combined):
                            continue
                        
                        for b3 in range(1, 129):
                            if time.time() > deadline:
                                break
                            if b3 == b1 or b3 == b2:
                                continue
                            s3 = [a1 + b3, a2 + b3, a3 + b3]
                            if not pairwise_coprime(s3):
                                continue
                            combined_all = s1 + s2 + s3
                            if not pairwise_coprime(combined_all):
                                continue
                            
                            B = [b1, b2, b3]
                            valid, cnt = score_pair(A, B)
                            if valid and cnt > best_score:
                                best_score = cnt
                                best_A = A[:]
                                best_B = B[:]
                                if best_score == 9:
                                    return [best_A, best_B]
    
    if time.time() > deadline:
        return [best_A, best_B]
    
    # Phase 2: Random search
    for _ in range(50000):
        if time.time() > deadline:
            break
        A = random.sample(range(1, 129), 3)
        A.sort()
        B = random.sample(range(1, 129), 3)
        B.sort()
        valid, cnt = score_pair(A, B)
        if valid and cnt > best_score:
            best_score = cnt
            best_A = A[:]
            best_B = B[:]
            if best_score == 9:
                break
    
    return [best_A, best_B]
