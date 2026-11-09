import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    limit = 10**18

    # First 30 primes for swap candidates
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]

    # Starting point: product of first 15 primes (squarefree)
    base_primes = primes[:15]
    n = 1
    for p in base_primes:
        n *= p

    def compute_tau_perp(num):
        """Count adjacent coprime pairs in sorted divisor list of num."""
        # Factorize
        factors = []
        temp = num
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.append(p)
                temp //= p
        if temp > 1:
            factors.append(temp)

        # Generate all divisors
        divisors = [1]
        for p in factors:
            new_divs = [d * p for d in divisors]
            divisors.extend(new_divs)

        divisors.sort()

        # Count adjacent coprime pairs
        count = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i + 1]) == 1:
                count += 1
        return count

    best_n = n
    best_score = compute_tau_perp(n)

    # Greedy single-swap hill climbing
    current_primes = list(base_primes)
    current_product = best_n

    improved = True
    while improved:
        improved = False
        for i in range(len(current_primes)):
            if time.time() > deadline - 10:
                return best_n
            for j in range(len(current_primes), len(primes)):
                if time.time() > deadline - 10:
                    return best_n
                candidate = current_product // current_primes[i] * primes[j]
                if candidate > limit:
                    break
                if candidate == best_n:
                    continue
                score = compute_tau_perp(candidate)
                if score > best_score:
                    best_score = score
                    best_n = candidate
                    current_product = candidate
                    current_primes[i] = primes[j]
                    current_primes.sort()
                    improved = True
                    break  # restart i-loop after improvement

    # Double-swap refinement if time remains
    if time.time() < deadline - 30:
        # Re-extract primes from best_n
        temp = best_n
        current_primes = []
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                current_primes.append(p)
                temp //= p
        if temp > 1:
            current_primes.append(temp)
        current_product = best_n

        for i in range(len(current_primes)):
            if time.time() > deadline - 10:
                return best_n
            for j in range(i + 1, len(current_primes)):
                if time.time() > deadline - 10:
                    return best_n
                for a in range(len(current_primes), len(primes)):
                    if time.time() > deadline - 10:
                        return best_n
                    for b in range(a + 1, len(primes)):
                        if time.time() > deadline - 10:
                            return best_n
                        candidate = current_product // current_primes[i] // current_primes[j] * primes[a] * primes[b]
                        if candidate > limit:
                            break
                        if candidate == best_n:
                            continue
                        score = compute_tau_perp(candidate)
                        if score > best_score:
                            best_score = score
                            best_n = candidate
                            current_product = candidate
                            current_primes = list(current_primes)
                            current_primes[i] = primes[a]
                            current_primes[j] = primes[b]
                            current_primes.sort()

    return best_n
