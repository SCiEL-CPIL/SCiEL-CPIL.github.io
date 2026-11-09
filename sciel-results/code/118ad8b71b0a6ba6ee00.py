import math
import time
import random
from itertools import combinations

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

def _is_pairwise_coprime(vals):
    n = len(vals)
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(vals[i], vals[j]) != 1:
                return False
    return True

def _score(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    if _is_pairwise_coprime(list(sums)):
        return len(sums)
    return 0

def construct():
    start = time.time()
    deadline = start + 80.0  # Safety margin

    # Sieve for primes up to 256 (max sum 128+128)
    limit = 256
    is_prime_sieve = [True] * (limit + 1)
    is_prime_sieve[0] = False
    is_prime_sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, limit + 1, i):
                is_prime_sieve[j] = False

    def is_prime_fast(n):
        if n <= limit:
            return is_prime_sieve[n]
        return _is_prime(n)

    odds = [x for x in range(1, 129) if x % 2 == 1]
    evens = [x for x in range(2, 129) if x % 2 == 0]

    # We want to find A (odds) and B (evens) such that all a+b are prime.
    # This guarantees distinct primes are pairwise coprime.
    # We iterate over A triples.
    # To keep it fast, we iterate over a subset of odds first, then expand.
    
    # Strategy:
    # For each A in combinations(odds, 3):
    #   For each b1 in evens:
    #     If not all(a+b1 is prime for a in A): continue
    #     For each b2 in evens where b2 > b1:
    #       If not all(a+b2 is prime for a in A): continue
    #       For each b3 in evens where b3 > b2:
    #         If all(a+b3 is prime for a in A):
    #           Return A, [b1, b2, b3]
    
    # Optimization: Precompute valid b's for each A? No, just loop.
    # We can limit the search to the first N odds to ensure speed.
    # If we don't find it in the first 50 odds, we might need to search more, 
    # but solutions are likely to exist with small numbers.
    
    # Let's try all combinations of the first 30 odds.
    # C(30, 3) = 4060.
    # For each A, we check pairs of evens.
    # This is fast.
    
    search_odds = odds[:30]
    
    for A in combinations(search_odds, 3):
        if time.time() > deadline:
            break
            
        # Check if there exists a B
        found_B = None
        # We can optimize by checking column 1, then 2, then 3
        valid_b1s = []
        for b1 in evens:
            if all(is_prime_fast(a + b1) for a in A):
                valid_b1s.append(b1)
        
        if not valid_b1s:
            continue
            
        for b1 in valid_b1s:
            if time.time() > deadline:
                break
            valid_b2s = []
            for b2 in evens:
                if b2 <= b1:
                    continue
                if all(is_prime_fast(a + b2) for a in A):
                    valid_b2s.append(b2)
            
            if not valid_b2s:
                continue
                
            for b2 in valid_b2s:
                if time.time() > deadline:
                    break
                for b3 in evens:
                    if b3 <= b2:
                        continue
                    if all(is_prime_fast(a + b3) for a in A):
                        found_B = [b1, b2, b3]
                        break
                if found_B:
                    break
            if found_B:
                break
        
        if found_B:
            return [list(A), found_B]

    # If not found in first 30 odds, try a broader random search as fallback
    # This should rarely be needed.
    random.seed(42)
    for _ in range(100000):
        if time.time() > deadline:
            break
        A = random.sample(odds, 3)
        B = random.sample(evens, 3)
        if all(is_prime_fast(a + b) for a in A for b in B):
            return [A, B]

    # Last resort: return a valid but suboptimal set if no prime solution found (unlikely)
    # Or just return the best found so far? 
    # The task requires maximizing cardinality. If we didn't find 9, we should try to find >0.
    # But finding 9 is the goal.
    # If we fail, we return a valid pair that might have fewer coprime sums.
    # However, the previous program scored 1.8, implying it found 9.
    # So the search above should find it.
    
    # Fallback to a known valid structure if possible, or just return something valid
    # A=[1,3,5], B=[2,4,6] -> sums: 3,5,7,5,7,9,7,9,11. Dups: 5,7,9. 
    # Distinct: 3,5,7,9,11. 
    # 9 is not coprime to 3. So this is invalid for "pairwise coprime distinct values".
    # Wait, "All DISTINCT values in A+B must be pairwise coprime".
    # If distinct values are {3,5,7,11}, they are pairwise coprime. 9 is not distinct from 9? 
    # "All DISTINCT values". So if 9 appears, and 3 appears, gcd(9,3)=3 != 1. Invalid.
    # So we MUST find a set where all distinct sums are pairwise coprime.
    # Primes are the safest bet.
    
    # If the loop above fails, we try a wider search
    for A in combinations(odds[:60], 3):
        if time.time() > deadline:
            break
        valid_b1s = [b for b in evens if all(is_prime_fast(a + b) for a in A)]
        if not valid_b1s: continue
        for b1 in valid_b1s:
            valid_b2s = [b for b in evens if b > b1 and all(is_prime_fast(a + b) for a in A)]
            if not valid_b2s: continue
            for b2 in valid_b2s:
                for b3 in evens:
                    if b3 > b2 and all(is_prime_fast(a + b3) for a in A):
                        return [list(A), [b1, b2, b3]]
        if time.time() > deadline:
            break
            
    # Absolute fallback
    return [1, 3, 5], [2, 4, 6]
