"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 148.0  # Safety margin before 150s limit
    
    LIMIT = 10**18
    
    # First 20 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    def get_divisors_from_factors(factors):
        """Generate all divisors from list of (prime, exponent) pairs."""
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        return sorted(divs)
    
    def tau_perp_from_factors(factors):
        """Count adjacent coprime pairs in sorted divisors."""
        divs = get_divisors_from_factors(factors)
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count
    
    best_n = 0
    best_score = 0
    
    def try_candidate(factors):
        nonlocal best_n, best_score
        n = 1
        for p, e in factors:
            n *= p ** e
        if n > LIMIT or n < 2:
            return
        score = tau_perp_from_factors(factors)
        if score > best_score:
            best_score = score
            best_n = n
    
    # Try squarefree with k primes for k = 12 to 16
    for k in range(12, 17):
        if k <= len(primes):
            try_candidate([(p, 1) for p in primes[:k]])
    
    # Systematic search: for each number of "free" primes (with exponent 1),
    # try increasing exponents of the smallest primes
    # Base: use first m primes, with first r having variable exponents, rest exponent 1
    
    # 16 primes total: 2^a * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 * 53
    base_16 = 1
    for p in primes[1:16]:
        base_16 *= p
    for a in range(1, 10):
        n = (2**a) * base_16
        if n <= LIMIT:
            try_candidate([(2, a)] + [(p, 1) for p in primes[1:16]])
        else:
            break
    
    # 16 primes: 2^a * 3^b * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 * 53
    base_16_b = 1
    for p in primes[2:16]:
        base_16_b *= p
    for a in range(1, 8):
        for b in range(1, 5):
            n = (2**a) * (3**b) * base_16_b
            if n <= LIMIT:
                try_candidate([(2, a), (3, b)] + [(p, 1) for p in primes[2:16]])
    
    # 15 primes total: 2^a * 3^b * 5^c * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    base_15_c = 1
    for p in primes[3:15]:
        base_15_c *= p
    for a in range(1, 10):
        for b in range(1, 6):
            for c in range(1, 4):
                n = (2**a) * (3**b) * (5**c) * base_15_c
                if n <= LIMIT:
                    try_candidate([(2, a), (3, b), (5, c)] + [(p, 1) for p in primes[3:15]])
    
    # 14 primes total: 2^a * 3^b * 5^c * 7^d * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43
    base_14_d = 1
    for p in primes[4:14]:
        base_14_d *= p
    for a in range(1, 12):
        for b in range(1, 7):
            for c in range(1, 5):
                for d in range(1, 4):
                    n = (2**a) * (3**b) * (5**c) * (7**d) * base_14_d
                    if n <= LIMIT:
                        try_candidate([(2, a), (3, b), (5, c), (7, d)] + [(p, 1) for p in primes[4:14]])
    
    # 13 primes total: 2^a * 3^b * 5^c * 7^d * 11^e * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41
    base_13_e = 1
    for p in primes[5:13]:
        base_13_e *= p
    for a in range(1, 14):
        for b in range(1, 8):
            for c in range(1, 6):
                for d in range(1, 5):
                    for e in range(1, 4):
                        n = (2**a) * (3**b) * (5**c) * (7**d) * (11**e) * base_13_e
                        if n <= LIMIT:
                            try_candidate([(2, a), (3, b), (5, c), (7, d), (11, e)] + [(p, 1) for p in primes[5:13]])
    
    if time.time() - start_time > DEADLINE:
        return best_n
    
    # Local search: try to improve by adjusting exponents or swapping primes
    best_factors = None
    if best_n > 0:
        # Reconstruct factors from best_n
        temp = best_n
        factors = []
        for p in primes:
            if temp % p == 0:
                e = 0
                while temp % p == 0:
                    temp //= p
                    e += 1
                factors.append((p, e))
        best_factors = factors
    
    if best_factors is not None:
        # Try local improvements
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            current_factors = list(best_factors)
            
            # Try increasing each exponent by 1
            for i in range(len(current_factors)):
                p, e = current_factors[i]
                if e < 12:
                    new_factors = list(current_factors)
                    new_factors[i] = (p, e + 1)
                    n = 1
                    for pp, ee in new_factors:
                        n *= pp ** ee
                    if n <= LIMIT:
                        score = tau_perp_from_factors(new_factors)
                        if score > best_score:
                            best_score = score
                            best_n = n
                            best_factors = new_factors
                            improved = True
                            break
            
            # Try decreasing each exponent by 1 (if > 1)
            for i in range(len(current_factors)):
                p, e = current_factors[i]
                if e > 1:
                    new_factors = list(current_factors)
                    new_factors[i] = (p, e - 1)
                    n = 1
                    for pp, ee in new_factors:
                        n *= pp ** ee
                    if 2 <= n <= LIMIT:
                        score = tau_perp_from_factors(new_factors)
                        if score > best_score:
                            best_score = score
                            best_n = n
                            best_factors = new_factors
                            improved = True
                            break
            
            # Try adding a new prime
            if len(current_factors) < 17:
                current_primes = set(p for p, e in current_factors)
                for p in primes:
                    if p not in current_primes:
                        new_factors = list(current_factors) + [(p, 1)]
                        n = 1
                        for pp, ee in new_factors:
                            n *= pp ** ee
                        if n <= LIMIT:
                            score = tau_perp_from_factors(new_factors)
                            if score > best_score:
                                best_score = score
                                best_n = n
                                best_factors = new_factors
                                improved = True
                                break
            
            # Try removing a prime with exponent 1 (if we have many primes)
            if len(current_factors) > 10:
                for i in range(len(current_factors)):
                    p, e = current_factors[i]
                    if e == 1:
                        new_factors = [f for j, f in enumerate(current_factors) if j != i]
                        if new_factors:
                            n = 1
                            for pp, ee in new_factors:
                                n *= pp ** ee
                            if 2 <= n <= LIMIT:
                                score = tau_perp_from_factors(new_factors)
                                if score > best_score:
                                    best_score = score
                                    best_n = n
                                    best_factors = new_factors
                                    improved = True
                                    break
    
    return best_n
# EVOLVE-BLOCK-END
