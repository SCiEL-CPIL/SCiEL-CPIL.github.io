import math
import time
import itertools

def _is_prime_sieve(limit):
    """Sieve of Eratosthenes to find primes up to limit."""
    is_prime = [True] * (limit + 1)
    if limit >= 0:
        is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return is_prime

def _pairwise_coprime(vals):
    """Check if all pairs in vals are coprime."""
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(vals[i], vals[j]) > 1:
                return False
    return True

def construct():
    """
    Return two lists A, B, each containing exactly 3 distinct integers in [1, 128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B.
    """
    t_start = time.time()
    DEADLINE = 85  # Safety margin before 90s kill

    MAX_SUM = 256
    is_prime = _is_prime_sieve(MAX_SUM)

    # Known valid solution with 9 distinct prime sums
    # A = [1, 3, 7], B = [4, 10, 16]
    # Sums: 5, 11, 17, 7, 13, 19, 11, 17, 23 -> distinct: {5,7,11,13,17,19,23} = 7 distinct, not 9
    # Let me find one properly.
    
    # A = [1, 5, 7], B = [2, 6, 10]
    # Sums: 3,7,11, 7,11,15, 9,13,17 -> distinct: {3,7,9,11,13,15,17} = 7, and 9&3 share factor, 15&3 share factor
    
    # Let me just search quickly. The current program does this well.
    # But let me hardcode a verified solution.
    
    # Search for a small solution first
    odds = list(range(1, 129, 2))
    evens = list(range(2, 129, 2))
    
    best_A = [1, 3, 5]
    best_B = [2, 4, 6]
    best_score = 0
    
    def get_score(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        distinct = list(sums)
        if not _pairwise_coprime(distinct):
            return 0
        return len(distinct)
    
    best_score = get_score(best_A, best_B)
    
    # Quick search for 9 distinct prime sums
    # Use small ranges for speed
    odds_search = odds[:30]
    evens_search = evens[:30]
    
    for A in itertools.combinations(odds_search, 3):
        if time.time() - t_start > DEADLINE:
            break
        A = list(A)
        a_diffs = {A[1]-A[0], A[2]-A[1], A[2]-A[0]}
        
        for B in itertools.combinations(evens_search, 3):
            if time.time() - t_start > DEADLINE:
                break
            B = list(B)
            b_diffs = {B[1]-B[0], B[2]-B[1], B[2]-B[0]}
            
            if a_diffs & b_diffs:
                continue
            
            # Check all 9 sums are prime
            all_prime = True
            for a in A:
                for b in B:
                    if not is_prime[a + b]:
                        all_prime = False
                        break
                if not all_prime:
                    break
            
            if all_prime:
                return [A, B]
    
    # If not found, expand search
    if time.time() - t_start < DEADLINE - 10:
        odds_search = odds[:60]
        evens_search = evens[:60]
        
        for A in itertools.combinations(odds_search, 3):
            if time.time() - t_start > DEADLINE:
                break
            A = list(A)
            a_diffs = {A[1]-A[0], A[2]-A[1], A[2]-A[0]}
            
            for B in itertools.combinations(evens_search, 3):
                if time.time() - t_start > DEADLINE:
                    break
                B = list(B)
                b_diffs = {B[1]-B[0], B[2]-B[1], B[2]-B[0]}
                
                if a_diffs & b_diffs:
                    continue
                
                all_prime = True
                for a in A:
                    for b in B:
                        if not is_prime[a + b]:
                            all_prime = False
                            break
                    if not all_prime:
                        break
                
                if all_prime:
                    return [A, B]
    
    # Fallback: find best pairwise coprime set
    if time.time() - t_start < DEADLINE - 5:
        odds_search = odds[:25]
        evens_search = evens[:25]
        
        for A in itertools.combinations(odds_search, 3):
            if time.time() - t_start > DEADLINE:
                break
            A = list(A)
            a_diffs = {A[1]-A[0], A[2]-A[1], A[2]-A[0]}
            
            for B in itertools.combinations(evens_search, 3):
                if time.time() - t_start > DEADLINE:
                    break
                B = list(B)
                b_diffs = {B[1]-B[0], B[2]-B[1], B[2]-B[0]}
                
                if a_diffs & b_diffs:
                    continue
                
                sums = set()
                for a in A:
                    for b in B:
                        sums.add(a + b)
                
                if len(sums) == 9 and _pairwise_coprime(list(sums)):
                    return [A, B]
    
    return [best_A, best_B]

if __name__ == "__main__":
    A, B = construct()
    print(f"A: {A}")
    print(f"B: {B}")
    sums = set(a + b for a in A for b in B)
    print(f"Distinct Sums: {sorted(sums)}")
    print(f"Count: {len(sums)}")
    print(f"Pairwise Coprime: {_pairwise_coprime(list(sums))}")
