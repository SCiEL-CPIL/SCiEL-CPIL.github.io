import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # The problem asks to maximize the number of adjacent coprime divisors.
    # Let n = prod p_i^{e_i}. The divisors are sorted.
    # The number of divisors is tau(n) = prod (e_i + 1).
    # The number of adjacent pairs is tau(n) - 1.
    # We want to maximize the count of pairs (d_i, d_{i+1}) with gcd(d_i, d_{i+1}) = 1.
    
    # Heuristic Insight:
    # 1. Maximize tau(n) first. The "density" of coprime pairs in the sorted divisor list
    #    is roughly constant for numbers with many small prime factors.
    #    Specifically, for a square-free number with k primes, the fraction of coprime
    #    adjacent pairs is significantly higher than for numbers with large exponents,
    #    because large exponents create "blocks" of divisors differing only by that prime,
    #    which are never coprime to each other.
    # 2. Therefore, the optimal strategy is likely to use the smallest primes with exponent 1,
    #    and potentially add exponents to the smallest primes if there is "slack" in the 10^18 bound.
    
    # Strategy:
    # 1. Identify the set of primes that can be included with exponent 1 (square-free core).
    # 2. Use DP or greedy to decide if we should swap a larger prime for an extra power of a smaller prime.
    # 3. Since the number of distinct primes is small (max ~15), we can enumerate combinations of exponents
    #    for the first few primes and check the rest.
    
    # Primes up to ~50 are sufficient.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    # First, find the maximum number of distinct primes (square-free case)
    # Product of first k primes <= 10^18
    prod = 1
    max_k = 0
    for i, p in enumerate(primes):
        if prod * p <= 10**18:
            prod *= p
            max_k = i + 1
        else:
            break
            
    # The best candidate is likely the product of the first max_k primes.
    # Let's call this n_base.
    # n_base = product of primes[0:max_k]
    
    # Is it better to have max_k primes with all exponents 1, 
    # or max_k - 1 primes with one exponent increased?
    # Example: 2^15 vs 2^14 * 3? No, 2^15 is not square free.
    # Compare:
    # Option A: k primes, all e=1. tau = 2^k.
    # Option B: k-1 primes, one e=2 (on smallest prime 2). 
    #   Primes: 2^2, 3, 5, ..., p_{k-1}.
    #   tau = 3 * 2^{k-2}.
    #   Ratio tau_B / tau_A = (3 * 2^{k-2}) / 2^k = 3/4 = 0.75.
    #   So Option A has 33% more divisors.
    #   Does Option B have a higher *fraction* of coprime pairs?
    #   In Option A (square-free), adjacent divisors in sorted order often differ by a factor of a small prime.
    #   If d and d*p are adjacent, gcd is d (if d>1) or p (if d=1). 
    #   Wait, if d=1, next is 2, gcd(1,2)=1. 
    #   If d=2, next is 3? gcd(2,3)=1.
    #   If d=3, next is 5? gcd(3,5)=1.
    #   Generally, in the sorted list of square-free divisors, adjacent elements d_i, d_{i+1} 
    #   satisfy d_{i+1} = d_i * p / q? No, they are just the next integer in the divisor set.
    #   
    #   Let's empirically verify the "density" argument by calculating the score for the top few candidates.
    #   Given the time limit, we can afford to calculate the score for ~10-20 candidates.
    
    candidates = []
    
    # Candidate 1: Square-free primorial of first max_k primes
    n1 = 1
    exps1 = [1] * max_k
    for i in range(max_k):
        n1 *= primes[i]
    candidates.append((n1, exps1))
    
    # Candidate 2: If we have room, maybe we can add a 16th prime? 
    # max_k is already the max for square-free.
    
    # Candidate 3: Replace the largest prime with a square of the smallest?
    # E.g., if max_k=15, n1 = 2*3*...*47.
    # Remove 47, multiply by 2 (so 2^2).
    # n2 = (n1 / 47) * 2
    if max_k >= 2:
        n2 = (n1 // primes[max_k-1]) * 2
        if n2 <= 10**18 and n2 != n1:
            exps2 = [1] * max_k
            exps2[0] = 2 # 2^2
            exps2[max_k-1] = 0 # 47^0
            candidates.append((n2, exps2))
            
    # Candidate 4: Replace two largest primes with squares of two smallest?
    # Remove 47, 43. Multiply by 2 (2^2) and 3 (3^2).
    if max_k >= 4:
        n3 = (n1 // primes[max_k-1]) // primes[max_k-2] * 2 * 3
        if n3 <= 10**18 and n3 != n1:
            exps3 = [1] * max_k
            exps3[0] = 2
            exps3[1] = 2
            exps3[max_k-1] = 0
            exps3[max_k-2] = 0
            candidates.append((n3, exps3))
            
    # Candidate 5: What if we use fewer primes but higher powers?
    # E.g. 2^a * 3^b * ...
    # The number of divisors grows exponentially with the number of distinct primes.
    # 2^10 * 3^6 * 5^4 ... is a common highly composite structure.
    # Let's try a "balanced" highly composite number candidate.
    # We want to maximize tau(n).
    # Standard approach: exponents should be non-increasing.
    
    # Let's generate a few highly composite candidates using a greedy exponent assignment.
    # Start with 1, multiply by smallest prime that keeps n <= 10^18,
    # but prioritize keeping exponents low to maintain square-free-like properties?
    # Actually, the current program's insight was that square-free is best.
    # Let's stick to the hypothesis that square-free primorial is optimal, 
    # but verify it against a few "near-miss" structures.
    
    # Let's implement a fast score calculator.
    def calc_score(n, exps, prime_list):
        # Generate divisors
        divs = [1]
        for i, p in enumerate(prime_list):
            e = exps[i]
            if e == 0:
                continue
            powers = [1]
            for _ in range(e):
                powers.append(powers[-1] * p)
            new_divs = []
            for d in divs:
                for pow_p in powers:
                    new_divs.append(d * pow_p)
            divs = new_divs
            
        divs.sort()
        
        score = 0
        # Optimization: We only need to check gcd.
        # For square-free numbers, divisors are bitmasks of primes.
        # gcd(d1, d2) = 1 iff they share no prime factors.
        # We can represent each divisor as a bitmask of the primes it contains.
        # This is much faster than math.gcd for large numbers if k is small.
        
        # Re-do with bitmask for speed if max_k <= 20 (which it is, max 15-16)
        # Map prime index to bit position
        # divisor_mask[i] = mask of primes dividing divs[i]
        
        # Wait, generating divisors as integers is fine for 32k divisors.
        # math.gcd is fast in C.
        # Let's just use math.gcd.
        
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    best_n = candidates[0][0]
    best_score = -1
    
    for n_c, exps_c in candidates:
        if time.time() > deadline:
            break
        s = calc_score(n_c, exps_c, primes[:max_k])
        if s > best_score:
            best_score = s
            best_n = n_c
            
    # If we have time, do a small local search around the best candidate.
    # Try swapping the largest prime with a power of a smaller prime.
    # E.g., if best is 2*3*...*47, try 2^2*3*...*41 (remove 43, 47? No, remove 47, add 2).
    # This is already in candidates.
    
    # Let's try one more structural variant:
    # What if we drop the largest prime and use the slack to increase the exponent of 2 to 3?
    # n4 = (n1 // primes[max_k-1]) * 4  (2^3 instead of 2^1, so multiply by 2)
    # Wait, 2^1 * 2^2 = 2^3. So multiply by 4.
    if max_k >= 2:
        n4 = (n1 // primes[max_k-1]) * 4
        if n4 <= 10**18 and n4 not in [c[0] for c in candidates]:
            exps4 = [1] * max_k
            exps4[0] = 3
            exps4[max_k-1] = 0
            candidates.append((n4, exps4))
            s = calc_score(n4, exps4, primes[:max_k])
            if s > best_score:
                best_score = s
                best_n = n4
                
    return best_n

# Run the function to ensure it works and returns an integer
# construct()
