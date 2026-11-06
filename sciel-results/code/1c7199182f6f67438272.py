"""Search for optimal pairwise-coprime sumsets using structured grid search."""
import math
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def pairwise_coprime(values):
        """Check if all distinct values in list are pairwise coprime."""
        vals = list(set(values))
        n = len(vals)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(vals[i], vals[j]) != 1:
                    return False
        return True
    
    def evaluate(A, B):
        """Return (valid, num_distinct) for the sumset."""
        sums = [a + b for a in A for b in B]
        distinct = list(set(sums))
        if pairwise_coprime(distinct):
            return True, len(distinct)
        return False, 0
    
    # Start with the known solution
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = 5  # |A+B| = 5
    
    # Precompute primes up to 256
    primes = set(p for p in range(2, 257) if is_prime(p))
    
    # Strategy: For a 3x3 grid with A=[a1,a2,a3], B=[b1,b2,b3],
    # parameterize by differences d1=a2-a1, d2=a3-a1, e1=b2-b1, e2=b3-b1
    # and base s11=a1+b1.
    # All 9 sums must be distinct and pairwise coprime.
    # Best case: all 9 are distinct primes.
    
    # Search for prime 3x3 grids
    found_9 = False
    
    # Try small differences first (more likely to find primes)
    for d1 in range(1, 30):
        for d2 in range(d1 + 1, 50):
            for e1 in range(1, 30):
                for e2 in range(e1 + 1, 50):
                    if time.time() > deadline - 5:
                        break
                    
                    # Constraints: a1+d2 <= 128, b1+e2 <= 128
                    # a1 >= 1, b1 >= 1
                    # s11 = a1+b1 >= 2
                    # a1 <= 128-d2, b1 <= 128-e2
                    max_a1 = 128 - d2
                    max_b1 = 128 - e2
                    if max_a1 < 1 or max_b1 < 1:
                        continue
                    
                    # s11 ranges from 2 to 256
                    # For each s11, check if all 9 sums are primes
                    # sums = [s11, s11+e1, s11+e2, s11+d1, s11+d1+e1,
                    #          s11+d1+e2, s11+d2, s11+d2+e1, s11+d2+e2]
                    
                    for s11 in range(2, 257):
                        # Quick check: are all 9 sums primes?
                        s_vals = [s11, s11+e1, s11+e2, s11+d1, s11+d1+e1,
                                  s11+d1+e2, s11+d2, s11+d2+e1, s11+d2+e2]
                        if not all(s in primes for s in s_vals):
                            continue
                        
                        # All 9 are distinct primes (differences are nonzero)
                        # Now find valid a1, b1
                        # a1 >= max(1, s11 - max_b1)
                        # a1 <= min(max_a1, s11 - 1)
                        # b1 = s11 - a1, must be in [1, 128]
                        a1_min = max(1, s11 - max_b1)
                        a1_max = min(max_a1, s11 - 1)
                        if a1_min > a1_max:
                            continue
                        
                        for a1 in range(a1_min, a1_max + 1):
                            b1 = s11 - a1
                            if b1 < 1 or b1 > 128:
                                continue
                            A = [a1, a1 + d1, a1 + d2]
                            B = [b1, b1 + e1, b1 + e2]
                            if all(1 <= x <= 128 for x in A + B) and \
                               len(set(A)) == 3 and len(set(B)) == 3:
                                best_A = A
                                best_B = B
                                best_score = 9
                                found_9 = True
                                break
                        if found_9:
                            break
                    if found_9:
                        break
                if found_9:
                    break
            if found_9:
                break
        if found_9:
            break
        if time.time() > deadline - 10:
            break
    
    if found_9:
        return [best_A, best_B]
    
    # If 9 not found via prime grid, try broader pairwise coprime search
    # Use random search with local improvement
    
    random.seed(42)
    
    def random_search(iterations=200000):
        nonlocal best_score, best_A, best_B
        count = 0
        while count < iterations and time.time() < deadline - 10:
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
            valid, nd = evaluate(A, B)
            if valid and nd > best_score:
                best_score = nd
                best_A = A[:]
                best_B = B[:]
                if best_score >= 8:
                    break
            count += 1
    
    random_search()
    
    # Local improvement
    def local_improve(A, B, max_iters=5000):
        nonlocal best_score, best_A, best_B
        A = A[:]
        B = B[:]
        cur_valid, cur_nd = evaluate(A, B)
        if not cur_valid:
            return
        cur_score = cur_nd
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try swapping an element in A
            for i in range(3):
                for new_val in range(1, 129):
                    if new_val in A:
                        continue
                    A_new = A[:]
                    A_new[i] = new_val
                    valid, nd = evaluate(A_new, B)
                    if valid and nd > cur_score:
                        A = A_new
                        cur_score = nd
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Try swapping an element in B
            for i in range(3):
                for new_val in range(1, 129):
                    if new_val in B:
                        continue
                    B_new = B[:]
                    B_new[i] = new_val
                    valid, nd = evaluate(A, B_new)
                    if valid and nd > cur_score:
                        B = B_new
                        cur_score = nd
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        if cur_score > best_score:
            best_score = cur_score
            best_A = A[:]
            best_B = B[:]
    
    if best_score < 9:
        local_improve(best_A, best_B)
    
    return [best_A, best_B]
