import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    LIMIT = 10**18

    # Precompute primes up to a reasonable bound
    def sieve(n):
        is_prime = [True] * (n + 1)
        if n >= 0: is_prime[0] = False
        if n >= 1: is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]

    all_primes = sieve(1000)

    def calc_tau_perp(n):
        """Compute tau_perp(n) by generating all divisors, sorting, counting adjacent coprime pairs."""
        if n < 2:
            return 0
        # Factorize n
        temp = n
        factors = {}
        for p in all_primes:
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                factors[p] = cnt
        if temp > 1:
            factors[temp] = 1
        
        # Generate all divisors
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divs = new_divs
        
        divs.sort()
        score = 0
        prev = divs[0]
        for i in range(1, len(divs)):
            curr = divs[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
        return score

    # The core insight: For squarefree n, the divisors are products of subsets of primes.
    # Two divisors are coprime if and only if their subset masks are disjoint.
    # The sorted order of divisors is determined by the actual integer values.
    # If we choose primes that are "well-separated" in magnitude, the sorting order
    # tends to follow the lexicographic order of the subset masks (or a similar structured order).
    # 
    # Specifically, if p_i grows very fast (e.g., p_{i+1} > p_1 * ... * p_i), 
    # then any product containing p_{i+1} is larger than any product not containing it.
    # This creates a recursive structure:
    # Divisors = [D without p_k] + [p_k * D without p_k]
    # The boundary between the two halves is the only place where a coprime pair 
    # can exist across the halves, plus the internal scores of the halves.
    #
    # Let S(primes) be the score for the set of primes.
    # If we pick primes such that p_k > product(p_1...p_{k-1}), the structure is a clean binary tree.
    # In this "Huffman-like" or "super-increasing" sequence, the score can be computed recursively.
    # 
    # However, we are constrained by n <= 10^18. We want to maximize the number of coprime adjacencies.
    # The current best is ~21.92, likely from the primorial of the first 15 primes (2*3*...*47).
    # Let's verify the structure of the primorial.
    # For the primorial, the primes are small and close. The "super-increasing" property fails 
    # (e.g., 5 > 2*3=6 is false). So the sorting is mixed.
    #
    # Strategy: 
    # 1. Identify that the maximum number of divisors for n <= 10^18 is achieved by 
    #    using the smallest primes with smallest exponents.
    #    The number of divisors is 2^k for squarefree.
    #    Max k for squarefree: 2*3*5*...*47 (15 primes) = 614889782588491410 < 10^18.
    #    16th prime is 53. 614889782588491410 * 53 > 10^18.
    #    So max squarefree divisors = 2^15 = 32768.
    #    Max possible coprime pairs is 32767.
    #    
    # 2. The score 21.92 is very low compared to 32767. Why?
    #    Because adjacent divisors in the sorted list of a primorial rarely share NO common factors.
    #    Example: 2, 3, 5, 6, 7, 10, 14, 15...
    #    (2,3) coprime? Yes.
    #    (3,5) coprime? Yes.
    #    (5,6) coprime? Yes.
    #    (6,7) coprime? Yes.
    #    (7,10) coprime? Yes.
    #    (10,14) gcd=2 No.
    #    (14,15) coprime? Yes.
    #    (15,21) gcd=3 No.
    #    
    #    It seems the score is actually much higher than 21.92 for the primorial?
    #    Let's re-read the task. "score tau_perp(n) is the number of adjacent pairs... with gcd 1".
    #    Wait, the current score is 21.9184. This is a FLOAT.
    #    "The evaluator measures the raw value (higher is better). The search score is value/seed".
    #    If the raw value is an integer, the score should be integer.
    #    Unless... the "value" is not the integer count?
    #    "score tau_perp(n) is the number of adjacent pairs d_i < d_{i+1} with gcd 1."
    #    This is an integer.
    #    Why is the score 21.9184?
    #    Maybe the "seed" is not 1.0? "1.0 equals the seed program".
    #    If the seed program returns a specific n, and its tau_perp is, say, 21, and the seed value is 21, then 21/21 = 1.0.
    #    But the current program scores 21.9184. This implies the raw value is 21.9184 * seed.
    #    Or perhaps the "value" is not just the count?
    #    Let's check the contract: "The evaluator measures the raw value".
    #    If the raw value is the count of coprime pairs, it must be an integer.
    #    Is it possible the evaluator computes something else?
    #    "Maximize. Large values below 10^18 are finite lower-bound constructions for the maximal order of tau_perp."
    #    
    #    Let's assume the raw value IS the integer count.
    #    If the current program scores 21.9184, and the seed is 1.0, then the raw value is 21.9184? Impossible for a count.
    #    UNLESS the "seed" program returns a value such that value/seed = 21.9184.
    #    Maybe the seed program returns n=2? tau_perp(2) = 1 (pair (1,2) is coprime).
    #    If seed value is 1, then 1/1 = 1.0.
    #    If current program returns n with tau_perp = 21.9184... impossible.
    #    
    #    Let's look at the inspirations. "score 21.9184".
    #    Perhaps the "value" is NOT the count, but something else?
    #    No, "score tau_perp(n) is the number of adjacent pairs".
    #    
    #    Hypothesis: The "score" reported in the prompt (21.9184) is the RATIO value/seed.
    #    If the seed program has a raw value of, say, 1, and the current program has a raw value of 21.9184... no, integer.
    #    Maybe the seed program's raw value is not 1.
    #    Let's assume the current program found an n with tau_perp(n) = 21 or 22.
    #    And the seed program found an n with tau_perp(n) = 1.
    #    Then 21/1 = 21. 22/1 = 22.
    #    21.9184 is very specific.
    #    
    #    Wait, could the "value" be the number of divisors? No, "score tau_perp".
    #    
    #    Let's look at the "Inspiration 1" text: "score 21.9184".
    #    It's possible the evaluator uses a different metric or there's a typo in my understanding of "integer".
    #    But "number of adjacent pairs" is definitely an integer.
    #    
    #    Let's ignore the float precision for a moment and focus on MAXIMIZING the integer count.
    #    If the current best is ~22, that is VERY low for 15 primes.
    #    Let's manually estimate for n = 30 (2*3*5).
    #    Divisors: 1, 2, 3, 5, 6, 10, 15, 30.
    #    Pairs:
    #    (1,2) gcd 1 -> Yes
    #    (2,3) gcd 1 -> Yes
    #    (3,5) gcd 1 -> Yes
    #    (5,6) gcd 1 -> Yes
    #    (6,10) gcd 2 -> No
    #    (10,15) gcd 5 -> No
    #    (15,30) gcd 15 -> No
    #    Total: 4.
    #    
    #    For n = 210 (2*3*5*7).
    #    Divisors: 1, 2, 3, 5, 6, 7, 10, 14, 15, 21, 30, 35, 42, 70, 105, 210.
    #    (1,2) Y
    #    (2,3) Y
    #    (3,5) Y
    #    (5,6) Y
    #    (6,7) Y
    #    (7,10) Y
    #    (10,14) N
    #    (14,15) Y
    #    (15,21) N
    #    (21,30) N
    #    (30,35) N
    #    (35,42) N
    #    (42,70) N
    #    (70,105) N
    #    (105,210) N
    #    Total: 7.
    #    
    #    For n = 2310 (2*3*5*7*11).
    #    Likely around 10-12.
    #    
    #    So 22 for 15 primes is plausible if the density of coprime pairs is low.
    #    
    #    How to increase it?
    #    We want adjacent divisors to be coprime.
    #    This happens if the "next" divisor in sorted order shares no factors with the "current" one.
    #    
    #    Consider the structure of the divisor lattice.
    #    If we can arrange the primes such that the sorted order alternates between "small support" and "large support" disjointly?
    #    
    #    Actually, the key is the "gaps".
    #    If p_i are chosen such that p_i > p_{i-1}^2, the divisors are very spread out.
    #    But we are limited by 10^18.
    #    
    #    Let's try a different approach: 
    #    Instead of just the primorial, what if we use FEWER primes but with EXONENTS?
    #    Exponents create more divisors, but do they help coprime pairs?
    #    n = 2^a. Divisors: 1, 2, 4, ..., 2^a.
    #    Pairs: (1,2) Y, (2,4) N, (4,8) N...
    #    Only 1 coprime pair. Bad.
    #    
    #    n = 2^a * 3^b.
    #    Divisors are mixed.
    #    
    #    It seems squarefree is best for coprime pairs because any shared factor is a full prime.
    #    
    #    Let's look at the "greedy interval-merging" idea from the directive.
    #    Model the sorted divisors as intervals.
    #    
    #    Alternative Strategy: 
    #    Use a genetic algorithm to optimize the SET of primes.
    #    Since we established max 15 primes, we can search over subsets of a larger pool of primes.
    #    The pool should be primes that allow for "better" adjacency.
    #    Which primes give better adjacency?
    #    Primes that are "far apart" in the multiplicative sense?
    #    
    #    Let's try to compute the score for the first 15 primes (primorial) to confirm the baseline.
    #    Then, try swapping some large primes with larger primes to see if the score changes.
    #    
    #    Wait, if I swap a prime p with a larger prime q, the divisors change.
    #    Larger q pushes divisors containing q further right.
    #    This might create larger gaps, but does it increase coprime pairs?
    #    
    #    Let's implement a fast scorer and a search.
    
    def fast_score(primes_tuple):
        """Compute score for a tuple of distinct primes (squarefree n)."""
        # Generate divisors
        # Use bitmask representation for speed if k is small, but k=15 is 32768 divisors.
        # Generating and sorting 32768 integers is fast in Python (< 0.1s).
        # The bottleneck is the GCD check. 32768 GCDs is also fast.
        # So we can afford ~1000 evaluations per second.
        
        divs = [1]
        for p in primes_tuple:
            new_divs = []
            for d in divs:
                new_divs.append(d * p)
            divs.extend(new_divs)
        
        divs.sort()
        score = 0
        prev = divs[0]
        for i in range(1, len(divs)):
            curr = divs[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
        return score

    # Baseline: First 15 primes
    baseline_primes = tuple(all_primes[:15])
    baseline_n = 1
    for p in baseline_primes:
        baseline_n *= p
    baseline_score = fast_score(baseline_primes)
    
    best_score = baseline_score
    best_n = baseline_n
    best_primes = baseline_primes
    
    # Search Strategy:
    # 1. Start with first 15 primes.
    # 2. Try swapping one prime with a larger prime from the pool (primes up to 1000).
    # 3. If score improves, accept.
    # 4. Repeat.
    # 5. Also try removing one prime and adding two smaller/larger ones? 
    #    No, keeping k=15 is likely optimal for divisor count.
    #    But maybe k=14 with better primes is better?
    #    Let's stick to k=15 first.
    
    pool = all_primes[15:100] # Primes from index 15 onwards
    
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        # Try swapping each prime in best_primes with a candidate from pool
        for i in range(len(best_primes)):
            if time.time() > DEADLINE: break
            p_old = best_primes[i]
            # Try candidates larger than p_old to maintain some order? 
            # Or any candidate not in best_primes.
            # To avoid re-evaluating same sets, we can just iterate.
            for p_new in pool:
                if p_new in best_primes:
                    continue
                # Check if n stays <= LIMIT
                # n_new = n_old / p_old * p_new
                # We can check log or just multiply.
                # Since we are swapping, n_new = best_n // p_old * p_new
                if best_n // p_old * p_new > LIMIT:
                    continue
                
                new_primes_list = list(best_primes)
                new_primes_list[i] = p_new
                # Sort to keep canonical form? Not strictly needed for score, but good for consistency.
                # However, the order in the tuple doesn't matter for the set of divisors.
                # So we can just compute score.
                
                score = fast_score(tuple(new_primes_list))
                
                if score > best_score:
                    best_score = score
                    best_n = best_n // p_old * p_new
                    best_primes = tuple(new_primes_list)
                    improved = True
                    # Update pool? No, pool is fixed.
                    # We can break early if we want to be greedy, but hill climbing is better.
                    # Let's continue to see if another swap is even better in this iteration.
                    # Actually, if we change best_primes, the loop indices might refer to old primes.
                    # It's safer to restart the inner loop or just mark improved and continue to next i.
                    # Given time constraints, simple hill climbing:
                    # If we find an improvement, update state and restart the outer loop?
                    # Or just continue scanning.
                    pass

    # Also try k=14, 13... 
    # Sometimes fewer primes allow for much larger gaps, which might help?
    # But 2^14 = 16384 divisors vs 32768.
    # Max possible score for k=14 is 16383.
    # If density is similar, score is half.
    # So k=15 is almost certainly better.
    
    # Final check: ensure best_n is valid
    if best_n < 2 or best_n > LIMIT:
        # Fallback to 2
        best_n = 2
    
    return best_n

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
