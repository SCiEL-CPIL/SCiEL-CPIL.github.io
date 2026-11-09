import math
import time
from itertools import combinations

def construct():
    deadline = time.time() + 80
    
    # Precompute primes up to 256
    is_prime = [True] * 257
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(256)) + 1):
        if is_prime[i]:
            for j in range(i*i, 257, i):
                is_prime[j] = False
    primes = set(i for i in range(2, 257) if is_prime[i])

    odds = [i for i in range(1, 129, 2)]
    evens = [i for i in range(2, 129, 2)]
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5 # Initial seed value
    
    # Helper to check if sums are distinct and pairwise coprime
    def check_solution(A, B):
        sums = set()
        for a in A:
            for b in B:
                s = a + b
                if 1 <= s <= 256:
                    sums.add(s)
                else:
                    return 0 # Out of bounds? Problem says A,B in [1,128], so sum in [2,256].
        
        if len(sums) < 2:
            return 0
            
        # Check pairwise coprime
        vals = list(sums)
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                if math.gcd(vals[i], vals[j]) != 1:
                    return 0
        return len(sums)

    # Strategy 1: Find A (odd), B (even) such that all 9 sums are distinct primes
    # Iterate A
    for A in combinations(odds, 3):
        if time.time() > deadline:
            break
        A = list(A)
        # Find valid B candidates (even) such that a+b is prime for all a in A
        valid_bs = []
        for b in evens:
            if time.time() > deadline:
                break
            ok = True
            for a in A:
                if not is_prime[a + b]:
                    ok = False
                    break
            if ok:
                valid_bs.append(b)
        
        if len(valid_bs) >= 3:
            # Try to find 3 distinct b's from valid_bs such that all 9 sums are distinct
            # Since sums are prime, distinctness is the only extra check (coprimality is automatic for distinct primes)
            # Wait, distinct primes are pairwise coprime.
            # So we just need distinct sums.
            found = False
            for B_cand in combinations(valid_bs, 3):
                B = list(B_cand)
                sums = set()
                distinct = True
                for a in A:
                    for b in B:
                        s = a + b
                        if s in sums:
                            distinct = False
                            break
                        sums.add(s)
                    if not distinct:
                        break
                if distinct and len(sums) == 9:
                    # Verify pairwise coprime (should be true for distinct primes)
                    # But just in case, or if we relax prime condition later
                    # For now, accept
                    best_A = A
                    best_B = B
                    best_score = 9
                    return best_A, best_B

    # If no 9-prime solution found, do a broader search for pairwise coprime sums
    # We can reuse the logic but check pairwise coprime instead of prime
    # And allow A, B to be any parity
    
    # To save time, we might just return the best found so far if we found 9 primes.
    # If not, we need to search for max cardinality pairwise coprime sums.
    
    # Given the time limit, if we didn't find 9 primes in the first loop, 
    # a full brute force for pairwise coprime might be too slow.
    # However, the first loop is O(N_odds^3 * N_evens * 3) which is manageable?
    # N_odds ~ 64. Combinations ~ 42k.
    # Inner loop: 64 evens * 3 checks = 192 ops.
    # Total ~ 42k * 192 ~ 8 million ops. This should run in < 1s in Python if optimized.
    # So we likely find a solution or exhaust the search quickly.
    
    # If we exhaust and find nothing, we fall back to a heuristic search for pairwise coprime.
    # But inspirations suggest 1.8 is reachable, implying 9 distinct pairwise coprime sums exist.
    # Since distinct primes are a subset of pairwise coprime, if a solution exists with 9 distinct primes, we find it.
    # If not, we need 9 distinct pairwise coprime non-primes.
    
    # Let's add a secondary search if the first one fails.
    # We'll search for A, B such that sums are pairwise coprime.
    # We can limit the search space to small numbers to make it feasible.
    
    if best_score < 9:
        # Search smaller range for pairwise coprime
        # A, B in [1, 50]
        small_odds = [i for i in range(1, 51, 2)]
        small_evens = [i for i in range(2, 51, 2)]
        
        # Try mixed parities too?
        # Let's just iterate all A, B in [1, 30] for speed
        range_val = 30
        nums = list(range(1, range_val + 1))
        
        for A in combinations(nums, 3):
            if time.time() > deadline:
                break
            A = list(A)
            for B in combinations(nums, 3):
                if time.time() > deadline:
                    break
                B = list(B)
                s = check_solution(A, B)
                if s > best_score:
                    best_score = s
                    best_A = A
                    best_B = B
                    if best_score == 9:
                        return best_A, best_B
                        
    return best_A, best_B
