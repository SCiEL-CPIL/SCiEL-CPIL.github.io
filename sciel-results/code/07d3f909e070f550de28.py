import math
import random

def construct():
    # Sieve of Eratosthenes for primes up to 256 (max sum 128+128)
    MAX_SUM = 256
    is_prime = [True] * (MAX_SUM + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(MAX_SUM ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, MAX_SUM + 1, i):
                is_prime[j] = False

    # Helper to check if a set of numbers are all prime
    def all_prime(nums):
        for n in nums:
            if n < 0 or n > MAX_SUM or not is_prime[n]:
                return False
        return True

    # Deterministic search for 9 distinct prime sums
    # A = {min_A, min_A+d1, min_A+d2}, B = {min_B, min_B+e1, min_B+e2}
    # Sums = min_A+min_B + R, where R = {a+b | a in {0,d1,d2}, b in {0,e1,e2}}
    # We need R to have 9 distinct elements.
    # And min_A+min_B+r to be prime for all r in R.
    
    found = False
    # Try small differences first
    for bound in [5, 10, 20, 50, 127]:
        diffs_A = []
        for d1 in range(1, bound + 1):
            for d2 in range(d1 + 1, bound + 1):
                diffs_A.append((d1, d2))
        
        for d1, d2 in diffs_A:
            if found: break
            A_diffs = (0, d1, d2)
            for e1, e2 in diffs_A:
                if found: break
                B_diffs = (0, e1, e2)
                
                R = set()
                for a in A_diffs:
                    for b in B_diffs:
                        R.add(a + b)
                
                if len(R) != 9:
                    continue
                
                max_r = max(R)
                # S = min_A + min_B
                # 1 <= min_A <= 128-d2
                # 1 <= min_B <= 128-e2
                # S >= 2
                # S + max_r <= 256
                min_S = 2
                max_S = 256 - max_r
                
                for S in range(min_S, max_S + 1):
                    if all_prime([S + r for r in R]):
                        # Check if we can realize S with valid min_A, min_B
                        # min_A + min_B = S
                        # min_A <= 128 - d2
                        # min_B <= 128 - e2  => S - min_A <= 128 - e2 => min_A >= S - (128 - e2)
                        low_A = max(1, S - (128 - e2))
                        high_A = min(128 - d2, S - 1)
                        
                        if low_A <= high_A:
                            min_A = low_A
                            min_B = S - min_A
                            A = [min_A, min_A + d1, min_A + d2]
                            B = [min_B, min_B + e1, min_B + e2]
                            
                            # Final verification
                            sums = set()
                            valid = True
                            for a in A:
                                for b in B:
                                    s = a + b
                                    sums.add(s)
                                    if not is_prime[s]:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid and len(sums) == 9:
                                return sorted(A), sorted(B)
                        found = True
                        break
                if found:
                    break
        if found:
            break

    # Fallback: Random search for max coprime sums
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def is_pairwise_coprime(sums):
        n = len(sums)
        for i in range(n):
            for j in range(i+1, n):
                if gcd(sums[i], sums[j]) > 1:
                    return False
        return True

    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if not sums:
            return 0
        if not is_pairwise_coprime(list(sums)):
            return 0
        return len(sums)

    best_score = 0
    best_A = [1, 2, 3]
    best_B = [1, 2, 3]
    
    # Initial seed
    s = evaluate([1, 7, 13], [4, 10, 16])
    if s > best_score:
        best_score = s
        best_A = [1, 7, 13]
        best_B = [4, 10, 16]

    for _ in range(5000):
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        score = evaluate(A, B)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = B
            
    return sorted(best_A), sorted(best_B)

# EVOLVE-BLOCK-START
# The code above is the logic. I will wrap it properly.
# EVOLVE-BLOCK-END
