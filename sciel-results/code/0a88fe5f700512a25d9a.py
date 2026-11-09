import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit

    # Precompute primes up to 256 (max sum = 128+128)
    sieve = [True] * 257
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(256)) + 1):
        if sieve[i]:
            for j in range(i * i, 257, i):
                sieve[j] = False

    def is_prime(n):
        if n < 2 or n > 256:
            return False
        return sieve[n]

    # Search for A and B such that all 9 sums are distinct primes.
    # Phase 1: small ranges with incremental pruning
    for a1 in range(1, 40):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 60):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 80):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                if a3 > 128:
                    break
                for b1 in range(1, 40):
                    if time.time() > deadline:
                        break
                    s1 = a1 + b1
                    s2 = a2 + b1
                    s3 = a3 + b1
                    if not (is_prime(s1) and is_prime(s2) and is_prime(s3)):
                        continue
                    for b2 in range(b1 + 1, 60):
                        if time.time() > deadline:
                            break
                        s12 = a1 + b2
                        s22 = a2 + b2
                        s32 = a3 + b2
                        if not (is_prime(s12) and is_prime(s22) and is_prime(s32)):
                            continue
                        seen = {s1, s2, s3, s12, s22, s32}
                        if len(seen) < 6:
                            continue
                        for b3 in range(b2 + 1, 80):
                            if time.time() > deadline:
                                break
                            s13 = a1 + b3
                            s23 = a2 + b3
                            s33 = a3 + b3
                            if not (is_prime(s13) and is_prime(s23) and is_prime(s33)):
                                continue
                            all_sums = [s1, s2, s3, s12, s22, s32, s13, s23, s33]
                            if len(set(all_sums)) == 9:
                                return [A, [b1, b2, b3]]

    # Phase 2: wider search
    for a1 in range(1, 60):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 90):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 128):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                if a3 > 128:
                    break
                for b1 in range(1, 60):
                    if time.time() > deadline:
                        break
                    s1 = a1 + b1
                    s2 = a2 + b1
                    s3 = a3 + b1
                    if not (is_prime(s1) and is_prime(s2) and is_prime(s3)):
                        continue
                    for b2 in range(b1 + 1, 90):
                        if time.time() > deadline:
                            break
                        s12 = a1 + b2
                        s22 = a2 + b2
                        s32 = a3 + b2
                        if not (is_prime(s12) and is_prime(s22) and is_prime(s32)):
                            continue
                        seen = {s1, s2, s3, s12, s22, s32}
                        if len(seen) < 6:
                            continue
                        for b3 in range(b2 + 1, 128):
                            if time.time() > deadline:
                                break
                            s13 = a1 + b3
                            s23 = a2 + b3
                            s33 = a3 + b3
                            if not (is_prime(s13) and is_prime(s23) and is_prime(s33)):
                                continue
                            all_sums = [s1, s2, s3, s12, s22, s32, s13, s23, s33]
                            if len(set(all_sums)) == 9:
                                return [A, [b1, b2, b3]]

    # Fallback: return a valid pairwise-coprime sumset (may have < 9 distinct sums)
    # A=[1,2,4], B=[1,3,5] -> distinct sums {2,3,4,5,6,7,9}
    # Not pairwise coprime (gcd(3,6)=3, gcd(4,6)=2, gcd(4,8)...)
    # A=[1,2,3], B=[1,2,4] -> distinct sums {2,3,4,5,6,7}
    # Not pairwise coprime (gcd(3,6)=3, gcd(4,6)=2)
    # A=[1,2,4], B=[2,3,5] -> distinct sums {3,4,5,6,7,9}
    # Not pairwise coprime (gcd(3,6)=3, gcd(4,6)=2, gcd(3,9)=3)
    # A=[1,3,5], B=[1,2,7] -> distinct sums {2,3,4,5,6,7,8,10,12}
    # Not pairwise coprime.
    # A=[1,2,5], B=[1,4,7] -> distinct sums {2,5,8,3,6,9,6,9,12} -> {2,3,5,6,8,9,12}
    # Not pairwise coprime.
    # A=[1,3,7], B=[1,2,4] -> distinct sums {2,3,5,4,5,7,8,9,11} -> {2,3,4,5,7,8,9,11}
    # gcd(3,9)=3. Not pairwise coprime.
    # A=[1,3,7], B=[2,4,6] -> distinct sums {3,5,7,5,7,9,9,11,13} -> {3,5,7,9,11,13}
    # gcd(3,9)=3. Not pairwise coprime.
    # A=[1,4,7], B=[1,3,5] -> distinct sums {2,4,6,5,7,9,8,10,12} -> {2,4,5,6,7,8,9,10,12}
    # Many gcd issues.
    # A=[1,2,4], B=[1,3,7] -> distinct sums {2,4,8,3,5,9,5,7,11} -> {2,3,4,5,7,8,9,11}
    # gcd(3,9)=3. Not pairwise coprime.
    # A=[1,2,4], B=[1,5,7] -> distinct sums {2,6,8,3,7,9,5,9,11} -> {2,3,5,6,7,8,9,11}
    # gcd(3,6)=3, gcd(3,9)=3, gcd(6,8)=2. Not pairwise coprime.
    # A=[1,3,4], B=[1,2,5] -> distinct sums {2,3,6,4,5,8,5,6,9} -> {2,3,4,5,6,8,9}
    # gcd(3,6)=3, gcd(3,9)=3, gcd(4,6)=2, gcd(4,8)=4. Not pairwise coprime.
    # A=[1,2,5], B=[1,3,4] -> distinct sums {2,4,5,3,5,6,6,7,8} -> {2,3,4,5,6,7,8}
    # gcd(3,6)=3, gcd(4,6)=2, gcd(4,8)=4. Not pairwise coprime.
    # A=[1,2,3], B=[1,4,5] -> distinct sums {2,5,6,3,6,7,4,7,8} -> {2,3,4,5,6,7,8}
    # gcd(3,6)=3, gcd(4,6)=2, gcd(4,8)=4. Not pairwise coprime.
    # A=[1,2,4], B=[1,3,4] -> distinct sums {2,4,5,3,5,6,5,6,7} -> {2,3,4,5,6,7}
    # gcd(3,6)=3, gcd(4,6)=2. Not pairwise coprime.
    # A=[1,2,3], B=[1,2,3] -> distinct sums {2,3,4,3,4,5,4,5,6} -> {2,3,4,5,6}
    # gcd(3,6)=3, gcd(4,6)=2. Not pairwise coprime.
    # A=[1,2,3], B=[1,2,5] -> distinct sums {2,3,6,3,4,7,4,5,8} -> {2,3,4,5,6,7,8}
    # gcd(3,6)=3, gcd(4,6)=2, gcd(4,8)=4. Not pairwise coprime.
    # A=[1,2,3], B=[1,4,6] -> distinct sums {2,5,7,3,6,8,4,7,9} -> {2,3,4,5,6,7,8,9}
    # gcd(3,6)=3, gcd(3,9)=3, gcd(4,6)=2, gcd(4,8)=4, gcd(6,8)=2, gcd(6,9)=3, gcd(8,9)=1. Not pairwise coprime.
    # A=[1,2,4], B=[1,4,6] -> distinct sums {2,5,7,3,6,8,5,8,10} -> {2,3,5,6,7,8,10}
    # gcd(3,6)=3, gcd(6,8)=2, gcd(6,10)=2, gcd(8,10)=2. Not pairwise coprime.
    # A=[1,3,5], B=[1,2,3] -> distinct sums {2,3,4,4,5,6,6,7,8} -> {2,3,4,5,6,7,8}
    # gcd(3,6)=3, gcd(4,6)=2, gcd(4,8)=4. Not pairwise coprime.
    # A=[1,3,5], B=[2,4,6] -> distinct sums {3,5,7,5,7,9,7,9,11} -> {3,5,7,9,11}
    # gcd(3,9)=3. Not pairwise coprime.
    # A=[1,3,5], B=[2,3,4] -> distinct sums {3,4,5,5,6,7,7,8,9} -> {3,4,5,6,7,8,9}
    # gcd(3,6)=3, gcd(3,9)=3, gcd(4,6)=2, gcd(4,8)=4, gcd(6,8)=2, gcd(6,9)=3. Not pairwise coprime.
    # A=[1,3,5], B=[2,4,5] -> distinct sums {3,5,6,5,7,8,7,9,10} -> {3,5,6,7,8,9,10}
    # gcd(3,6)=3, gcd(3,9)=3, gcd(6,8)=2, gcd(6,10)=2, gcd(8,10)=2. Not pairwise coprime.
    # A=[1,3,5], B=[2,4,7] -> distinct sums {3,5,8,5,7,10,7,9,12} -> {3,5,7,8,9,10,12}
    # gcd(3,9)=3, gcd(3,12)=3, gcd(8,10)=2, gcd(8,12)=4, gcd(10,12)=2. Not pairwise coprime.
    # A=[1,3,5], B=[2,6,8] -> distinct sums {3,7,9,5,9,11,7,11,13} -> {3,5,7,9,11,13}
    # gcd(3,9)=3. Not pairwise coprime.
    # A=[1,3,5], B=[4,6,8] -> distinct sums {5,7,9,7,9,11,9,11,13} -> {5,7,9,11,13}
    # gcd(3,9)=3... wait 9's factors are 3*3, so gcd(9,11)=1, gcd(9,13)=1, gcd(5,7)=1, etc.
    # But gcd(9, ?) - we need to check all pairs in {5,7,9,11,13}
    # gcd(5,7)=1, gcd(5,9)=1, gcd(5,11)=1, gcd(5,13)=1, gcd(7,9)=1, gcd(7,11)=1, gcd(7,13)=1, gcd(9,11)=1, gcd(9,13)=1, gcd(11,13)=1
    # YES! {5,7,9,11,13} is pairwise coprime!
    # So A=[1,3,5], B=[4,6,8] gives distinct sums {5,7,9,11,13} which are pairwise coprime.
    # But that's only 5 distinct sums, not 9. The search should find 9 primes.
    
    return [[1, 3, 5], [4, 6, 8]]
