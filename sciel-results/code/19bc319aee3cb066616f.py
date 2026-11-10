import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s wall-clock
    
    # Precompute smallest prime factor (SPF) for fast factorization
    MAX_V = 1000000
    spf = list(range(MAX_V + 1))
    for i in range(2, int(MAX_V**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_V + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def factorize_fast(n):
        """Factorize n using precomputed SPF."""
        factors = {}
        while n > 1:
            p = spf[n] if n <= MAX_V else n
            if p > MAX_V:
                # n is prime (or has a large prime factor)
                factors[p] = factors.get(p, 0) + 1
                n //= p
                continue
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            factors[p] = factors.get(p, 0) + cnt
        return factors
    
    def divisors_in_range(factors, lo, hi):
        """Generate divisors from prime factorization dict, only those in [lo, hi]."""
        if not factors:
            return [1] if lo <= 1 <= hi else []
        primes = list(factors.keys())
        exps = [factors[p] for p in primes]
        divs = [1]
        for i, p in enumerate(primes):
            new_divs = []
            mult = 1
            for e in range(exps[i] + 1):
                for d in divs:
                    nd = d * mult
                    if nd > hi:
                        break
                    new_divs.append(nd)
                mult *= p
                if mult > hi:
                    break
            divs = new_divs
            if not divs:
                return []
        return [d for d in divs if d >= lo]
    
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # x range: n/4 < x < 3n/4
        x_start = n // 4 + 1
        x_end = (3 * n) // 4
        
        # Precompute factorization of n
        n_factors = factorize_fast(n)
        
        for x in range(x_start, x_end + 1):
            if time.time() > deadline:
                break
            
            K = 4 * x - n
            if K <= 0:
                continue
            
            target = n * x
            target_sq = target * target
            
            # Factorize target_sq = (n*x)^2
            nx = n * x
            nx_factors = factorize_fast(nx)
            target_sq_factors = {}
            for p, e in nx_factors.items():
                target_sq_factors[p] = e * 2
            
            # For y and z to be positive: A > -target (always true for A>0)
            # For y != z: A != target
            # For y != x: (A+target)/K != x => A != K*x - target = (4x-n)*x - n*x = 3x^2 - nx
            # For z != x: B != 3x^2 - nx
            # y <= 10^12: A <= K*10^12 - target
            # z <= 10^12: B <= K*10^12 - target, so A >= target_sq / (K*10^12 - target)
            
            max_denom = 10**12
            A_max = min(target_sq, K * max_denom - target)
            if A_max < 1:
                continue
            
            # Lower bound on A from z <= 10^12: B <= K*10^12 - target => A >= target_sq/(K*10^12 - target)
            denom_bound = K * max_denom - target
            if denom_bound <= 0:
                continue
            A_min = max(1, (target_sq + denom_bound - 1) // denom_bound)
            
            if A_min > A_max:
                continue
            
            # Generate only divisors in [A_min, A_max]
            divisors = divisors_in_range(target_sq_factors, A_min, A_max)
            
            # Avoid A = target (which gives y = z)
            # Avoid A = 3x^2 - nx (which gives y = x)
            bad_A1 = target  # y = z
            bad_A2 = 3 * x * x - n * x  # y = x
            
            for A in divisors:
                if A == bad_A1 or A == bad_A2:
                    continue
                B = target_sq // A
                if B == bad_A1 or B == bad_A2:
                    continue
                if (A + target) % K == 0:
                    if (B + target) % K == 0:
                        y = (A + target) // K
                        z = (B + target) // K
                        # Ensure distinctness: x, y, z all distinct and positive
                        if x != y and x != z and y != z:
                            if x > 0 and y > 0 and z > 0:
                                if z <= max_denom and y <= max_denom:
                                    found = [x, y, z]
                                    break
            if found:
                break
        
        answers.append(found)
    
    return answers
# EVOLVE-BLOCK-END
