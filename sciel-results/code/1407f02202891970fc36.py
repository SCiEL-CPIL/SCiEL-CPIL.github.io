import math
import time
from collections import defaultdict

def _sigma_from_factors(factors):
    """Calculate sigma from prime factorization {p: e}"""
    result = 1
    for p, e in factors.items():
        if e == 0:
            continue
        result *= (p**(e+1) - 1) // (p - 1)
    return result

def _factorize(n, primes_list):
    """Factor n using given prime list. Returns dict {p: e} or None if not fully factored."""
    factors = {}
    for p in primes_list:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            factors[p] = e
    if n > 1:
        # n is prime (or has a prime factor not in our list)
        if n in primes_list:
            factors[n] = factors.get(n, 0) + 1
        else:
            return None
    return factors

def _sigma(n, primes_list):
    """Compute sigma(n) using trial division with primes_list. Returns None if n has prime factor not in list."""
    if n <= 0:
        return None
    factors = _factorize(n, primes_list)
    if factors is None:
        return None
    return _sigma_from_factors(factors)

def _count_solutions(M, primes_list, max_a=30):
    """Count and collect solutions k with k*sigma(k)=M, k<=sqrt(M)."""
    solutions = set()
    A = 0
    M_temp = M
    while M_temp % 2 == 0:
        M_temp //= 2
        A += 1
    
    # M = 2^A * N, N odd
    N = M_temp
    sqrt_M = math.isqrt(M)
    
    for a in range(1, min(A + 1, max_a) + 1):
        Q_a = (1 << (a + 1)) - 1
        # Need: 2^a * d * Q_a * sigma(d) = M = 2^A * N
        # So d * Q_a * sigma(d) = 2^(A-a) * N
        # Since d, Q_a, sigma(d) are all odd, LHS is odd.
        # RHS = 2^(A-a) * N. For this to be odd, need A-a=0, i.e., a=A.
        # Wait, that's wrong again. Let me just check directly.
        
        # k = 2^a * d, d odd
        # k*sigma(k) = 2^a * d * (2^(a+1)-1) * sigma(d)
        # This must equal M.
        # So d * (2^(a+1)-1) * sigma(d) = M / (2^a)
        # M / (2^a) must be integer, so a <= A.
        # And d * (2^(a+1)-1) * sigma(d) = M / (2^a)
        
        if M % (1 << a) != 0:
            continue
        target = M // (1 << a)
        # Need d * Q_a * sigma(d) = target, d odd
        # d must be a divisor of target
        # Also d must be odd
        
        # Iterate over possible d values
        # d is odd, d | target
        # d * Q_a * sigma(d) = target
        # So Q_a must divide target
        if target % Q_a != 0:
            continue
        T = target // Q_a
        # Now d * sigma(d) = T
        # d must be odd and d | T (since sigma(d) >= d+1 for d>1, and sigma(1)=1)
        # Actually d * sigma(d) = T, and sigma(d) >= d+1 for d>1, sigma(1)=1
        # So d^2 + d <= T for d > 1, and d=1 works if T=1
        
        # Find all d such that d*sigma(d) = T
        # d must be odd
        # d <= sqrt(T) roughly, since sigma(d) >= d+1 for d>1
        # Actually d * sigma(d) = T, sigma(d) >= 1, so d <= T
        # But sigma(d) >= d+1 for d > 1, so d*(d+1) <= T, d <= sqrt(T)
        # For d=1: 1*1=1, so T must be 1
        
        found_d = []
        if T == 1:
            found_d.append(1)
        
        # Try d from 1 to isqrt(T), odd only
        # But T can be large. Let's be smarter.
        # d * sigma(d) = T
        # sigma(d) = T/d
        # So we need T/d to be integer and sigma(d) = T/d
        # d must divide T
        
        # Generate odd divisors of T
        # But T can be up to 10^18, too many divisors potentially
        
        # Alternative: iterate d and check if d*sigma(d) == T
        # d must be odd, d <= T
        # But sigma(d) grows roughly like d, so d*sigma(d) ~ d^2
        # So d <= sqrt(T)
        
        max_d = math.isqrt(T)
        # Check d=1 separately
        # For d >= 3 (odd), check up to max_d
        # This could be slow if T is large
        
        # Let's just check a few small d values and d=T (if T is odd)
        # Actually, let's iterate odd d up to min(max_d, 100000) for speed
        # And also check d = T (if T is odd and T*sigma(T)==T, i.e., sigma(T)==1, only T=1)
        
        limit = min(max_d, 200000)
        for d in range(1, limit + 1, 2):  # odd d only
            if T % d != 0:
                continue
            s_val = T // d
            # Check if sigma(d) == s_val
            sig_d = _sigma(d, primes_list)
            if sig_d is not None and sig_d == s_val:
                k = (1 << a) * d
                if k <= sqrt_M and k * k <= M:
                    solutions.add(k)
        
        # Also check if T itself is a solution (d=T, sigma(T)=1, only if T=1)
        # Already handled
        
    return solutions

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # safety margin
    
    # Primes for factorization
    primes_list = list(range(3, 2000, 2))  # odd primes up to 2000
    # Add some specific primes needed
    extra_primes = [2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999]
    primes_list = primes_list + extra_primes
    primes_list = sorted(set(primes_list))
    
    # Try the seed solution first
    best_M = 333312
    best_solutions = {336, 372, 434}
    
    # Try to find better M values
    # Strategy: construct M = 2^A * N where N is odd
    # For each a in 1..A, we want d*sigma(d) = M/(2^a * Q_a) to have a solution
    
    # Let's try various A and N
    # Key insight: for a=A, we need d*sigma(d) = N/Q_A
    # For a=A-1, we need d*sigma(d) = 2*N/Q_{A-1} (but d is odd, so this requires 2*N/Q_{A-1} to be odd, which means Q_{A-1} must have a factor of 2... but Q_a is always odd. So 2*N/Q_{A-1} is even, but d*sigma(d) for odd d: d is odd, sigma(d) can be even or odd.
    
    # Actually, let me just try a computational search over small M values that are products of small primes
    
    # Try M = 2^A * prod of small odd primes
    # Let's try A from 1 to 20, and N as product of subsets of small primes
    
    small_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Precompute Q_a for a=1..30
    Q_vals = {}
    for a in range(1, 31):
        Q_vals[a] = (1 << (a+1)) - 1
    
    # Try different constructions
    # Construction 1: M = 2^A * N where N is product of primes that divide various Q_a
    
    # For efficiency, let's try a few specific M values
    
    candidate_Ms = []
    
    # The seed
    candidate_Ms.append(333312)
    
    # Try M with more prime factors
    # M = 2^A * 3^a3 * 5^a5 * 7^a7 * ...
    
    # Let's try to maximize the number of a values for which a solution exists
    # For a given M = 2^A * N, a solution exists for a if:
    # 1. a <= A
    # 2. Q_a divides M/2^a = 2^(A-a) * N
    # 3. There exists odd d with d*sigma(d) = M/(2^a * Q_a)
    
    # Since Q_a is odd, condition 2 means Q_a | N (if A>=a)
    
    # Let's try N as product of primes dividing Q_a for a=1..K
    # Primes in Q_a for a=1..15: 3,5,7,11,13,17,23,31,43,73,89,127,151,191,331
    
    # Try M = 2^A * 3^3 * 5^2 * 7^2 * 11 * 13 * 17 * 23 * 31 * 43 * 73 * 89 * 127 * 151 * 191
    # This might be too large
    
    # Let's just try a few and see
    
    for A in [6, 8, 9, 10, 12, 14, 16]:
        for n_primes in [3, 5, 7]:
            N = 1
            for p in small_primes[:n_primes]:
                N *= p
            M = (1 << A) * N
            if M <= 10**18:
                candidate_Ms.append(M)
    
    # Also try the original construction
    N_orig = 3**3 * 5**2 * 7**2 * 11 * 13 * 17 * 23 * 31 * 73 * 89 * 127
    for A in [8, 9, 10, 11, 12]:
        M = (1 << A) * N_orig
        if M <= 10**18:
            candidate_Ms.append(M)
    
    # Deduplicate
    candidate_Ms = list(set(candidate_Ms))
    
    # For each candidate, count solutions
    for M in candidate_Ms:
        if time.time() > deadline:
            break
        sols = _count_solutions(M, primes_list)
        if len(sols) > len(best_solutions):
            best_M = M
            best_solutions = sols
    
    # Also try a targeted search: for small k values, compute k*sigma(k) and find M with many hits
    # k from 1 to 10000
    if time.time() < deadline - 30:
        from collections import Counter
        M_counts = Counter()
        M_solutions = defaultdict(set)
        
        for k in range(1, 50000):
            if time.time() > deadline - 30:
                break
            sig = _sigma(k, primes_list)
            if sig is None:
                continue
            M_val = k * sig
            if M_val <= 10**18 and M_val >= 1:
                M_counts[M_val] += 1
                if M_val not in M_solutions or len(M_solutions[M_val]) < 1000:
                    M_solutions[M_val].add(k)
        
        # Find M with most solutions
        for M_val, count in M_counts.most_common(50):
            if time.time() > deadline:
                break
            sols = M_solutions[M_val]
            # Filter: k <= sqrt(M_val)
            sqrt_M = math.isqrt(M_val)
            valid_sols = {k for k in sols if k <= sqrt_M}
            if len(valid_sols) > len(best_solutions):
                best_M = M_val
                best_solutions = valid_sols
    
    if not best_solutions:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Convert to list of Python ints
    k_list = sorted([int(k) for k in best_solutions])
    
    return {'M': int(best_M), 'k': k_list}
