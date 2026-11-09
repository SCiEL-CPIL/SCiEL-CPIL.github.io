import math
import time

def construct():
    deadline = time.time() + 80

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

    def is_pairwise_coprime(values):
        vals = list(set(values))
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if math.gcd(vals[i], vals[j]) != 1:
                    return False
        return True

    def score(A, B):
        sums = set()
        for a in A:
            for b in B:
                s = a + b
                if 1 <= s <= 256:
                    sums.add(s)
        if len(sums) < 2:
            return 0
        if not is_pairwise_coprime(sums):
            return 0
        return len(sums)

    best = 5
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]

    # Try to find sets where all 9 sums are primes (hence pairwise coprime)
    # Use a smarter search: iterate over possible A, then find B that makes all sums prime
    
    # Precompute primes up to 256
    primes = set()
    for n in range(2, 257):
        if is_prime(n):
            primes.add(n)

    # Try structured approach: use small A values and search B
    for a1 in range(1, 30):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 30):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 30):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                # For each candidate B triple, check if all 9 sums are prime
                for b1 in range(1, 30):
                    if time.time() > deadline:
                        break
                    for b2 in range(b1 + 1, 30):
                        if time.time() > deadline:
                            break
                        for b3 in range(b2 + 1, 30):
                            if time.time() > deadline:
                                break
                            B = [b1, b2, b3]
                            sums = set()
                            valid = True
                            for a in A:
                                for b in B:
                                    s = a + b
                                    if not is_prime(s):
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                best = 9
                                best_A = A
                                best_B = B
                                return best_A, best_B
    
    # If no 9-prime solution found, try to maximize with pairwise coprime (not necessarily prime)
    # Search more broadly
    for a1 in range(1, 50):
        if time.time() > deadline:
            break
        for a2 in range(a1 + 1, 50):
            if time.time() > deadline:
                break
            for a3 in range(a2 + 1, 50):
                if time.time() > deadline:
                    break
                A = [a1, a2, a3]
                for b1 in range(1, 50):
                    if time.time() > deadline:
                        break
                    for b2 in range(b1 + 1, 50):
                        if time.time() > deadline:
                            break
                        for b3 in range(b2 + 1, 50):
                            if time.time() > deadline:
                                break
                            B = [b1, b2, b3]
                            s = score(A, B)
                            if s > best:
                                best = s
                                best_A = A
                                best_B = B
                                if best == 9:
                                    return best_A, best_B
    
    return best_A, best_B
