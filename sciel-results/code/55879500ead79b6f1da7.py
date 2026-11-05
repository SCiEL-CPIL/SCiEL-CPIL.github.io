import time
import math
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill
    
    # Precompute prime factors for numbers 2..256
    prime_factors = [set() for _ in range(257)]
    for i in range(2, 257):
        if not prime_factors[i]:  # i is prime
            for j in range(i, 257, i):
                prime_factors[j].add(i)
    
    def all_pairwise_coprime(sums_list):
        """Check if all elements in sums_list are pairwise coprime."""
        n = len(sums_list)
        for i in range(n):
            for j in range(i + 1, n):
                if prime_factors[sums_list[i]] & prime_factors[sums_list[j]]:
                    return False
        return True
    
    def score(A, B):
        """Return the number of distinct sums if ALL are pairwise coprime, else 0."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        
        if not sums:
            return 0
        
        if all_pairwise_coprime(list(sums)):
            return len(sums)
        return 0
    
    # Generate candidate numbers with few prime factors
    candidates_by_factor_count = {}
    for n in range(1, 129):
        pf = prime_factors[n]
        cnt = len(pf)
        if cnt not in candidates_by_factor_count:
            candidates_by_factor_count[cnt] = []
        candidates_by_factor_count[cnt].append(n)
    
    # Prioritize numbers with 0, 1, or 2 prime factors
    good_candidates = set()
    for cnt in [0, 1, 2]:
        if cnt in candidates_by_factor_count:
            good_candidates.update(candidates_by_factor_count[cnt])
    
    candidate_list = sorted(good_candidates)
    
    # Add some with 3 factors for diversity
    for cnt in [3]:
        if cnt in candidates_by_factor_count:
            candidate_list.extend(candidates_by_factor_count[cnt][:30])
    
    candidate_list = sorted(set(candidate_list))
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score(best_A, best_B)
    
    # Strategy: exhaustive search over small candidate set
    small_candidates = candidate_list[:40]
    
    for A in combinations(small_candidates, 3):
        if time.time() > deadline:
            break
        A = list(A)
        for B in combinations(small_candidates, 3):
            if time.time() > deadline:
                break
            B = list(B)
            s = score(A, B)
            if s > best_score:
                best_score = s
                best_A = A[:]
                best_B = B[:]
                if best_score == 9:
                    return [best_A, best_B]
    
    # If not found 9, try larger set
    if best_score < 9 and time.time() < deadline:
        for A in combinations(candidate_list, 3):
            if time.time() > deadline:
                break
            A = list(A)
            for B in combinations(candidate_list, 3):
                if time.time() > deadline:
                    break
                B = list(B)
                s = score(A, B)
                if s > best_score:
                    best_score = s
                    best_A = A[:]
                    best_B = B[:]
                    if best_score == 9:
                        return [best_A, best_B]
    
    return [best_A, best_B]

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    import time
    import math
    from itertools import combinations
    
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill
    
    # Precompute prime factors for numbers 2..256
    prime_factors = [set() for _ in range(257)]
    for i in range(2, 257):
        if not prime_factors[i]:  # i is prime
            for j in range(i, 257, i):
                prime_factors[j].add(i)
    
    def all_pairwise_coprime(sums_list):
        """Check if all elements in sums_list are pairwise coprime."""
        n = len(sums_list)
        for i in range(n):
            for j in range(i + 1, n):
                if prime_factors[sums_list[i]] & prime_factors[sums_list[j]]:
                    return False
        return True
    
    def score(A, B):
        """Return the number of distinct sums if ALL are pairwise coprime, else 0."""
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        
        if not sums:
            return 0
        
        if all_pairwise_coprime(list(sums)):
            return len(sums)
        return 0
    
    # Generate candidate numbers with few prime factors
    candidates_by_factor_count = {}
    for n in range(1, 129):
        pf = prime_factors[n]
        cnt = len(pf)
        if cnt not in candidates_by_factor_count:
            candidates_by_factor_count[cnt] = []
        candidates_by_factor_count[cnt].append(n)
    
    # Prioritize numbers with 0, 1, or 2 prime factors
    good_candidates = set()
    for cnt in [0, 1, 2]:
        if cnt in candidates_by_factor_count:
            good_candidates.update(candidates_by_factor_count[cnt])
    
    candidate_list = sorted(good_candidates)
    
    # Add some with 3 factors for diversity
    for cnt in [3]:
        if cnt in candidates_by_factor_count:
            candidate_list.extend(candidates_by_factor_count[cnt][:30])
    
    candidate_list = sorted(set(candidate_list))
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score(best_A, best_B)
    
    # Strategy: exhaustive search over small candidate set
    small_candidates = candidate_list[:40]
    
    for A in combinations(small_candidates, 3):
        if time.time() > deadline:
            break
        A = list(A)
        for B in combinations(small_candidates, 3):
            if time.time() > deadline:
                break
            B = list(B)
            s = score(A, B)
            if s > best_score:
                best_score = s
                best_A = A[:]
                best_B = B[:]
                if best_score == 9:
                    return [best_A, best_B]
    
    # If not found 9, try larger set
    if best_score < 9 and time.time() < deadline:
        for A in combinations(candidate_list, 3):
            if time.time() > deadline:
                break
            A = list(A)
            for B in combinations(candidate_list, 3):
                if time.time() > deadline:
                    break
                B = list(B)
                s = score(A, B)
                if s > best_score:
                    best_score = s
                    best_A = A[:]
                    best_B = B[:]
                    if best_score == 9:
                        return [best_A, best_B]
    
    return [best_A, best_B]

# EVOLVE-BLOCK-END
