"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    n = 256
    deadline = time.time() + 50  # safety margin before 90s kill
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def is_coprime_to_all(a, selected):
        for b in selected:
            if gcd(a, b) != 1:
                return False
        return True
    
    # Strategy 1: Greedy descending
    selected1 = [1]
    for a in range(n, 0, -1):
        if time.time() > deadline:
            break
        if a == 1:
            continue
        if is_coprime_to_all(a, selected1):
            selected1.append(a)
    sum1 = sum(selected1)
    
    # Strategy 2: Largest prime power for each prime
    def is_prime(num):
        if num < 2:
            return False
        if num < 4:
            return True
        if num % 2 == 0 or num % 3 == 0:
            return False
        i = 5
        while i * i <= num:
            if num % i == 0 or num % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = [p for p in range(2, n + 1) if is_prime(p)]
    selected2 = [1]
    for p in primes:
        pp = 1
        while pp * p <= n:
            pp *= p
        selected2.append(pp)
    # Verify
    valid2 = True
    for i in range(len(selected2)):
        for j in range(i + 1, len(selected2)):
            if gcd(selected2[i], selected2[j]) != 1:
                valid2 = False
                break
        if not valid2:
            break
    sum2 = sum(selected2) if valid2 else 0
    
    # Strategy 3: For each prime, try largest multiple that is coprime to others
    # Use a smarter greedy: sort primes by the largest power descending, assign greedily
    prime_info = []
    for p in primes:
        pp = 1
        while pp * p <= n:
            pp *= p
        prime_info.append((pp, p))
    prime_info.sort(key=lambda x: -x[0])
    
    selected3 = [1]
    used_primes = set()
    for pp, p in prime_info:
        if p in used_primes:
            continue
        # Check if pp is coprime to all in selected3
        if is_coprime_to_all(pp, selected3):
            selected3.append(pp)
            # Mark all prime factors of pp as used
            temp = pp
            for q in primes:
                while temp % q == 0:
                    used_primes.add(q)
                    temp //= q
    sum3 = sum(selected3)
    
    # Pick the best strategy
    best_sum = max(sum1, sum2, sum3)
    if best_sum == sum1:
        best = selected1
    elif best_sum == sum2:
        best = selected2
    else:
        best = selected3
    
    # Local improvement: try to swap elements to increase sum
    best = list(best)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best)):
            for a in range(n, 1, -1):
                if a in best:
                    continue
                # Try replacing best[i] with a
                new_best = best[:i] + [a] + best[i+1:]
                valid = True
                for j in range(len(new_best)):
                    for k in range(j + 1, len(new_best)):
                        if gcd(new_best[j], new_best[k]) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid and sum(new_best) > sum(best):
                    best = new_best
                    improved = True
                    break
            if improved:
                break
    
    # Final validation
    for i in range(len(best)):
        for j in range(i + 1, len(best)):
            if gcd(best[i], best[j]) != 1:
                # Fallback to safest option
                best = [1] + [p for p in primes]
                break
        else:
            continue
        break
    
    return [int(x) for x in best]

# EVOLVE-BLOCK-END
