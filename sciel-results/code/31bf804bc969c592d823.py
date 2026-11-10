import time
import math
import random

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

    # Primes up to 5000 to allow more search space
    primes = sieve(5000)
    
    def get_divisors(factors):
        """Generate divisors from factorization using iterative multiplication."""
        divs = [1]
        for p, e in factors:
            new_divs = []
            # Precompute powers of p
            powers = [1]
            for _ in range(e):
                powers.append(powers[-1] * p)
            for d in divs:
                for pow_val in powers:
                    new_divs.append(d * pow_val)
            divs = new_divs
        return divs

    def tau_perp_fast(factors):
        """
        Fast tau_perp computation.
        Key insight: Two divisors d1, d2 are coprime iff they share no prime factors.
        We can represent each divisor by its prime support (bitmask).
        Adjacent divisors in sorted order that have disjoint supports contribute to score.
        
        For efficiency, we generate divisors with their prime support masks.
        """
        if not factors:
            return 0
        
        num_primes = len(factors)
        if num_primes > 20:
            # Too many primes for bitmask approach, fall back to gcd
            divs = get_divisors(factors)
            divs.sort()
            score = 0
            for i in range(len(divs) - 1):
                if math.gcd(divs[i], divs[i+1]) == 1:
                    score += 1
            return score
        
        # Generate divisors with prime support masks
        # Each divisor is represented as (value, mask) where mask is bitmask of primes used
        divs = [(1, 0)]
        for idx, (p, e) in enumerate(factors):
            bit = 1 << idx
            new_divs = []
            powers = [(1, 0)]
            for _ in range(e):
                val, mask = powers[-1]
                powers.append((val * p, mask | bit))
            for d_val, d_mask in divs:
                for pow_val, pow_mask in powers:
                    new_divs.append((d_val * pow_val, d_mask | pow_mask))
            divs = new_divs
        
        # Sort by value
        divs.sort(key=lambda x: x[0])
        
        # Count adjacent pairs with disjoint supports
        score = 0
        for i in range(len(divs) - 1):
            if (divs[i][1] & divs[i+1][1]) == 0:
                score += 1
        return score

    # Initialize with a strong candidate
    # Heuristic: Use small primes with decreasing exponents
    # Product of first k primes: 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 ≈ 6.1e17
    # So we can use up to ~15 primes with exponent 1, or fewer with higher exponents
    
    best_n = 1
    best_score = 0
    best_factors = []
    
    # Try multiple initial configurations
    init_configs = [
        # (primes_used, exponents)
        (primes[:15], [1]*15),
        (primes[:14], [2,1,1,1,1,1,1,1,1,1,1,1,1,1]),
        (primes[:13], [3,2,1,1,1,1,1,1,1,1,1,1,1]),
        (primes[:12], [4,3,2,1,1,1,1,1,1,1,1,1]),
        (primes[:11], [5,4,3,2,1,1,1,1,1,1,1]),
        (primes[:10], [6,5,4,3,2,1,1,1,1,1]),
    ]
    
    for primes_used, exps in init_configs:
        if len(primes_used) != len(exps):
            continue
        factors = list(zip(primes_used, exps))
        prod = 1
        for p, e in factors:
            prod *= p ** e
        if 2 <= prod <= 10**18:
            s = tau_perp_fast(factors)
            if s > best_score:
                best_score = s
                best_n = prod
                best_factors = list(factors)
    
    # Fallback if no valid init
    if not best_factors:
        for p in primes:
            if best_n * p > 10**18:
                break
            best_n *= p
            best_factors.append((p, 1))
        best_score = tau_perp_fast(best_factors)
    
    current_factors = list(best_factors)
    current_prod = best_n
    current_score = best_score

    # Local Search with multiple move types
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
                    s = tau_perp_fast(new_factors)
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
                s = tau_perp_fast(new_factors)
                if s > best_move_score:
                    best_move_score = s
                    best_new_factors = new_factors
                    best_new_prod = new_prod

        # 2. Swap primes (replace prime p with prime q at same exponent)
        current_primes_set = {p for p, e in current_factors}
        for i, (p, e) in enumerate(current_factors):
            base = current_prod // (p ** e)
            # Try primes not in current set
            for q in primes:
                if q in current_primes_set:
                    continue
                new_prod = base * (q ** e)
                if new_prod > 10**18 or new_prod < 2:
                    continue
                new_factors = list(current_factors)
                new_factors[i] = (q, e)
                s = tau_perp_fast(new_factors)
                if s > best_move_score:
                    best_move_score = s
                    best_new_factors = new_factors
                    best_new_prod = new_prod

        # 3. Add new prime
        for q in primes:
            if q in current_primes_set:
                continue
            new_prod = current_prod * q
            if new_prod > 10**18:
                continue
            new_factors = list(current_factors)
            new_factors.append((q, 1))
            s = tau_perp_fast(new_factors)
            if s > best_move_score:
                best_move_score = s
                best_new_factors = new_factors
                best_new_prod = new_prod

        # 4. Remove prime (if exponent > 1, reduce; if exp==1, remove)
        for i, (p, e) in enumerate(current_factors):
            if e == 1:
                # Remove entirely
                if len(current_factors) > 1:  # Keep at least one prime
                    new_prod = current_prod // p
                    new_factors = list(current_factors)
                    new_factors.pop(i)
                    s = tau_perp_fast(new_factors)
                    if s > best_move_score:
                        best_move_score = s
                        best_new_factors = new_factors
                        best_new_prod = new_prod
            else:
                # Reduce exponent by 1
                new_e = e - 1
                new_prod = current_prod // p
                if new_prod >= 2:
                    new_factors = list(current_factors)
                    new_factors[i] = (p, new_e)
                    s = tau_perp_fast(new_factors)
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
            if no_improve_count > 10:
                break

    return best_n
