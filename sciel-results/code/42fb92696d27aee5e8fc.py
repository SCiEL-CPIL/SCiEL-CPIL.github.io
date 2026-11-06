import math
import time

def construct():
    start_time = time.time()
    LIMIT = 10**18
    DEADLINE = 110.0  # Safety margin
    
    # Primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    def get_divisors(n, prime_factors):
        """
        n: the number
        prime_factors: list of (prime, exponent)
        Returns sorted list of divisors.
        """
        divisors = [1]
        for p, e in prime_factors:
            new_divs = []
            for d in divisors:
                val = d
                for _ in range(e):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        divisors.sort()
        return divisors

    def calc_score(n):
        if n < 2:
            return 0, n
        # Factorize n using known primes or simple trial division
        # Since we generate n from known primes, we can pass factors.
        # But for generality/safety in this block, let's just factorize.
        # However, n is up to 1e18. Trial division up to sqrt(n) is too slow if n is prime.
        # But our candidates are constructed from small primes.
        # We will pass the factorization explicitly to avoid re-factoring.
        pass

    def score_from_factors(factors):
        """
        factors: list of (p, e)
        Returns (score, n)
        """
        n = 1
        for p, e in factors:
            n *= (p ** e)
        if n > LIMIT or n < 2:
            return -1, n
            
        divs = get_divisors(n, factors)
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score, n

    best_score = 0
    best_n = 2
    
    # Helper to update best
    def update_best(score, n):
        nonlocal best_score, best_n
        if score > best_score:
            best_score = score
            best_n = n

    # Candidate 1: Product of first 15 primes
    # 2..47
    factors_15 = [(p, 1) for p in primes[:15]]
    s, n = score_from_factors(factors_15)
    if s > best_score:
        best_score = s
        best_n = n
        
    # Candidate 2: P_14 * m
    # P_14 = 2..43
    base_14 = 1
    for p in primes[:14]:
        base_14 *= p
        
    # Multipliers up to LIMIT // base_14
    max_m = LIMIT // base_14
    # We test multipliers composed of small primes to keep the structure "dense"
    # or just test all integers? 77 is small.
    for m in range(1, max_m + 1):
        if time.time() > DEADLINE:
            break
        # Construct factors for base_14 * m
        # We need to merge the factors of m into the base factors
        m_factors = {}
        temp_m = m
        # Factorize m (small)
        for p in primes:
            if p * p > temp_m:
                break
            while temp_m % p == 0:
                m_factors[p] = m_factors.get(p, 0) + 1
                temp_m //= p
        if temp_m > 1:
            m_factors[temp_m] = m_factors.get(temp_m, 0) + 1
            
        # Merge with base factors (all exponent 1 for first 14 primes)
        full_factors = {}
        for p in primes[:14]:
            full_factors[p] = 1
        for p, e in m_factors.items():
            if p in full_factors:
                full_factors[p] += e
            else:
                full_factors[p] = e
                
        factor_list = [(p, e) for p, e in full_factors.items()]
        s, n = score_from_factors(factor_list)
        if s > best_score:
            best_score = s
            best_n = n

    # Candidate 3: P_13 * m
    base_13 = base_14 // 43
    max_m = LIMIT // base_13
    # To avoid too many iterations, we only test m composed of 2, 3, 5, 7
    # This covers many "smooth" numbers which are likely optimal
    def gen_smooth(max_val, primes_list):
        # Generate all numbers <= max_val composed of primes_list
        # Using a priority queue or simple recursion
        # Since max_val is ~3333, simple recursion is fine
        results = [1]
        for p in primes_list:
            new_results = []
            for r in results:
                val = r
                while val <= max_val:
                    new_results.append(val)
                    val *= p
            results = new_results
        return sorted(list(set(results)))
    
    smooth_mults = gen_smooth(max_m, [2, 3, 5, 7])
    for m in smooth_mults:
        if time.time() > DEADLINE:
            break
        m_factors = {}
        temp_m = m
        for p in [2, 3, 5, 7]:
            while temp_m % p == 0:
                m_factors[p] = m_factors.get(p, 0) + 1
                temp_m //= p
        if temp_m > 1:
            m_factors[temp_m] = m_factors.get(temp_m, 0) + 1
            
        full_factors = {}
        for p in primes[:13]:
            full_factors[p] = 1
        for p, e in m_factors.items():
            if p in full_factors:
                full_factors[p] += e
            else:
                full_factors[p] = e
                
        factor_list = [(p, e) for p, e in full_factors.items()]
        s, n = score_from_factors(factor_list)
        if s > best_score:
            best_score = s
            best_n = n

    # Candidate 4: P_12 * m (smooth with 2,3,5,7,11)
    base_12 = base_13 // 41
    max_m = LIMIT // base_12
    smooth_mults = gen_smooth(max_m, [2, 3, 5, 7, 11])
    for m in smooth_mults:
        if time.time() > DEADLINE:
            break
        m_factors = {}
        temp_m = m
        for p in [2, 3, 5, 7, 11]:
            while temp_m % p == 0:
                m_factors[p] = m_factors.get(p, 0) + 1
                temp_m //= p
        if temp_m > 1:
            m_factors[temp_m] = m_factors.get(temp_m, 0) + 1
            
        full_factors = {}
        for p in primes[:12]:
            full_factors[p] = 1
        for p, e in m_factors.items():
            if p in full_factors:
                full_factors[p] += e
            else:
                full_factors[p] = e
                
        factor_list = [(p, e) for p, e in full_factors.items()]
        s, n = score_from_factors(factor_list)
        if s > best_score:
            best_score = s
            best_n = n

    return best_n

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# We just need to expose it.
# EVOLVE-BLOCK-END
