import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit

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

    def is_pairwise_coprime(values):
        """Check if all distinct values in the list are pairwise coprime."""
        vals = sorted(set(values))
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if math.gcd(vals[i], vals[j]) > 1:
                    return False
        return True

    best_result = None
    best_count = 0

    # Try to find 9 distinct pairwise-coprime sums (all primes would work)
    # Phase 1: focused search with tighter bounds
    for a1 in range(1, 25):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 45):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 70):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                for b1 in range(1, 25):
                    if time.time() > deadline:
                        break
                    s1 = a1 + b1
                    s2 = a2 + b1
                    s3 = a3 + b1
                    if not (is_prime(s1) and is_prime(s2) and is_prime(s3)):
                        continue
                    for b2 in range(b1 + 1, 45):
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
                        for b3 in range(b2 + 1, 70):
                            if time.time() > deadline:
                                break
                            s13 = a1 + b3
                            s23 = a2 + b3
                            s33 = a3 + b3
                            if not (is_prime(s13) and is_prime(s23) and is_prime(s33)):
                                continue
                            all_sums = [s1, s2, s3, s12, s22, s32, s13, s23, s33]
                            distinct = set(all_sums)
                            if len(distinct) == 9 and is_pairwise_coprime(all_sums):
                                return [A, [b1, b2, b3]]

    # Phase 2: wider search
    for a1 in range(1, 40):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 70):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 100):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                for b1 in range(1, 40):
                    if time.time() > deadline:
                        break
                    s1 = a1 + b1
                    s2 = a2 + b1
                    s3 = a3 + b1
                    if not (is_prime(s1) and is_prime(s2) and is_prime(s3)):
                        continue
                    for b2 in range(b1 + 1, 70):
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
                        for b3 in range(b2 + 1, 100):
                            if time.time() > deadline:
                                break
                            s13 = a1 + b3
                            s23 = a2 + b3
                            s33 = a3 + b3
                            if not (is_prime(s13) and is_prime(s23) and is_prime(s33)):
                                continue
                            all_sums = [s1, s2, s3, s12, s22, s32, s13, s23, s33]
                            distinct = set(all_sums)
                            if len(distinct) == 9 and is_pairwise_coprime(all_sums):
                                return [A, [b1, b2, b3]]

    # Phase 3: broader search allowing non-prime but pairwise-coprime sums
    for a1 in range(1, 30):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 55):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 85):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                for b1 in range(1, 30):
                    if time.time() > deadline:
                        break
                    for b2 in range(b1 + 1, 55):
                        if time.time() > deadline:
                            break
                        for b3 in range(b2 + 1, 85):
                            if time.time() > deadline:
                                break
                            all_sums = [a1+b1, a2+b1, a3+b1, a1+b2, a2+b2, a3+b2, a1+b3, a2+b3, a3+b3]
                            distinct = set(all_sums)
                            if len(distinct) > best_count and is_pairwise_coprime(all_sums):
                                best_count = len(distinct)
                                best_result = [A, [b1, b2, b3]]
                                if best_count == 9:
                                    return best_result

    # Fallback: known valid solution with 5 pairwise-coprime distinct sums
    # A=[1,3,5], B=[4,6,8] -> sums {5,7,9,11,13} all pairwise coprime
    if best_result is not None and best_count >= 5:
        return best_result
    return [[1, 3, 5], [4, 6, 8]]
