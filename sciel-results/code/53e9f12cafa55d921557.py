import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # Primes up to 2000 is plenty for n <= 10^18
    primes = sieve(2000)
    
    def get_divisors(factors):
        # factors: list of (p, e)
        divs = [1]
        for p, e in factors:
            new_divs = []
            powers = [1]
            for _ in range(e):
                powers.append(powers[-1] * p)
            for d in divs:
                for pow_val in powers:
                    new_divs.append(d * pow_val)
            divs = new_divs
        return divs

    def tau_perp(n, factors):
        # factors: list of (p, e)
        # n is the product
        if len(factors) == 0:
            return 0
        divs = get_divisors(factors)
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Initialize with a strong candidate
    # Start with 2^a * 3^b * ...
    # Heuristic: exponents decreasing
    best_n = 1
    best_score = 0
    best_factors = []
    
    # Try to build a good initial set
    # Let's try 2^4 * 3^2 * 5 * 7 * 11 * ...
    init_primes = primes[:12]
    init_exps = [4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    init_factors = list(zip(init_primes, init_exps))
    
    # Check product
    prod = 1
    for p, e in init_factors:
        prod *= p ** e
    if prod > 10**18:
        # Scale down
        while prod > 10**18 and init_exps[0] > 0:
            init_exps[0] -= 1
            prod //= init_primes[0]
        # If still too big, remove primes
        # Simple approach: just use fewer primes or lower exps
        # For now, let's try a simpler init
        init_factors = []
        prod = 1
        for i, p in enumerate(primes):
            e = max(1, 5 - i) # Decreasing exponents
            if prod * (p ** e) <= 10**18:
                init_factors.append((p, e))
                prod *= p ** e
            else:
                break
        
    # Evaluate init
    if init_factors:
        s = tau_perp(prod, init_factors)
        if s > best_score:
            best_score = s
            best_n = prod
            best_factors = list(init_factors)
    else:
        # Fallback to smallest primes
        for p in primes:
            if best_n * p > 10**18: break
            best_n *= p
            best_factors.append((p, 1))
        best_score = tau_perp(best_n, best_factors)

    current_factors = list(best_factors)
    current_prod = best_n
    current_score = best_score

    # Local Search
    # We will perform a best-improvement hill climbing
    # Moves:
    # 1. Increase/Decrease exponent of existing prime
    # 2. Swap prime p with prime q (same exponent)
    # 3. Add new prime q (exponent 1)
    # 4. Remove prime p (if exponent > 1, reduce exp; if exp==1, remove)
    
    improved = True
    no_improve_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        best_move_score = current_score
        best_new_factors = None
        best_new_prod = current_prod
        
        # 1. Exponent changes
        for i, (p, e) in enumerate(current_factors):
            # Decrease
            if e > 1:
                new_e = e - 1
                new_prod = current_prod // p
                if new_prod >= 2:
                    new_factors = list(current_factors)
                    new_factors[i] = (p, new_e)
                    s = tau_perp(new_prod, new_factors)
                    if s > best_move_score:
                        best_move_score = s
                        best_new_factors = new_factors
                        best_new_prod = new_prod
                        
            # Increase
            if current_prod * p <= 10**18:
                new_e = e + 1
                new_prod = current_prod * p
                new_factors = list(current_factors)
                new_factors[i] = (p, new_e)
                s = tau_perp(new_prod, new_factors)
                if s > best_move_score:
                    best_move_score = s
                    best_new_factors = new_factors
                    best_new_prod = new_prod

        # 2. Swap primes
        current_primes_set = {p for p, e in current_factors}
        for i, (p, e) in enumerate(current_factors):
            # Try swapping p with other primes
            # Limit search to primes close to p or smaller
            for q in primes:
                if q in current_primes_set:
                    continue
                # Heuristic: q should not be too large
                # new_prod = current_prod * (q / p)
                # Check if integer division works (it will if we track carefully)
                # Actually, current_prod is divisible by p**e.
                base = current_prod // (p ** e)
                new_prod = base * (q ** e)
                if new_prod > 10**18:
                    continue
                if new_prod < 2:
                    continue
                new_factors = list(current_factors)
                new_factors[i] = (q, e)
                s = tau_perp(new_prod, new_factors)
                if s > best_move_score:
                    best_move_score = s
                    best_new_factors = new_factors
                    best_new_prod = new_prod

        # 3. Add new prime
        # Try adding smallest primes not in set
        for q in primes:
            if q in current_primes_set:
                continue
            new_prod = current_prod * q
            if new_prod > 10**18:
                continue
            new_factors = list(current_factors)
            new_factors.append((q, 1))
            s = tau_perp(new_prod, new_factors)
            if s > best_move_score:
                best_move_score = s
                best_new_factors = new_factors
                best_new_prod = new_prod

        if best_new_factors is not None:
            current_factors = best_new_factors
            current_prod = best_new_prod
            current_score = best_move_score
            improved = True
            no_improve_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_n = current_prod
                best_factors = list(current_factors)
        else:
            no_improve_count += 1
            # If stuck, try a random restart or perturbation?
            # For now, just break if we've tried enough
            if no_improve_count > 5:
                break

    return best_n
