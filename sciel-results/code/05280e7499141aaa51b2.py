"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 70  # 70s limit, leaving margin before 90s kill
    
    MAX_N = 256
    
    # Generate primes up to MAX_N
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(n) + 1):
            if sieve[i]:
                for j in range(i * i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(MAX_N)
    prime_set = set(primes)
    
    # Factorize all numbers from 2 to 256
    def factorize(n):
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors
    
    # For each number, get its distinct prime factors
    num_factors = {}
    for n in range(2, MAX_N + 1):
        num_factors[n] = factorize(n)
    
    # All candidate numbers: 1 and 2..256
    # 1 has no prime factors
    candidates = [1] + list(range(2, MAX_N + 1))
    
    # Initial solution: pick all primes (each prime is coprime to others) plus 1
    # Sum of primes up to 256
    initial = [1] + primes
    initial_sum = sum(initial)
    
    # Better initial: for each prime p, consider the largest multiple of p <= 256
    # that only uses primes <= p (to avoid conflicts)
    # Actually, let's do a greedy: sort numbers by value descending, pick if coprime to all selected
    
    def is_coprime_to_all(n, selected):
        for s in selected:
            if s == 1:
                continue
            if gcd(n, s) != 1:
                return False
        return True
    
    # Greedy: sort by value descending
    greedy_selected = [1]
    for n in sorted(range(MAX_N, 1, -1)):
        if is_coprime_to_all(n, greedy_selected):
            greedy_selected.append(n)
    
    greedy_sum = sum(greedy_selected)
    
    # Local search: try to swap or add
    best_selected = greedy_selected if greedy_sum > initial_sum else initial
    best_sum = max(greedy_sum, initial_sum)
    
    # Try local improvements
    # For each selected number, try replacing it with a larger number that's coprime to all others
    selected_set = set(best_selected)
    selected_list = list(best_selected)
    
    # Build list of all numbers not in selected
    unselected = [n for n in range(2, MAX_N + 1) if n not in selected_set]
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i, s in enumerate(selected_list):
            if time.time() > deadline:
                break
            # Try to replace s with a larger number
            others = [x for j, x in enumerate(selected_list) if j != i and x != 1]
            for n in sorted(unselected, reverse=True):
                if n <= s:
                    break
                # Check if n is coprime to all others
                ok = True
                for o in others:
                    if gcd(n, o) != 1:
                        ok = False
                        break
                if ok:
                    # Replace s with n
                    selected_list[i] = n
                    selected_set.remove(s)
                    selected_set.add(n)
                    unselected.remove(n)
                    unselected.append(s)
                    best_sum += (n - s)
                    improved = True
                    break
            if time.time() > deadline:
                break
    
    # Try adding numbers (if any slot is free - but we can't add if all primes are used)
    # Actually we can only add a number if it's coprime to ALL selected
    for n in sorted(unselected, reverse=True):
        if time.time() > deadline:
            break
        if is_coprime_to_all(n, selected_list):
            selected_list.append(n)
            selected_set.add(n)
            unselected.remove(n)
            best_sum += n
            improved = True
    
    # Final: ensure all are Python ints
    result = [int(x) for x in selected_list]
    # Ensure distinct
    result = list(set(result))
    # Ensure all in 1..256
    result = [x for x in result if 1 <= x <= 256]
    # Ensure nonempty
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
