import math
import time
import sys

# We need primes up to a reasonable limit.
# Product of first k primes <= 10^18.
# 2*3*5*...*47 (15 primes) is approx 6e17.
# 2*3*...*53 (16 primes) is approx 3e19 > 10^18.
# So max 15 distinct primes for square-free part.
# We will consider the first 15 primes as candidates for the "core" of n.
# Larger primes are unlikely to be useful because they increase n quickly
# without adding much to the divisor count or coprime structure relative to cost.

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

PRIMES = _sieve(100)
CANDIDATE_PRIMES = PRIMES[:15]
MAX_N = 10**18

def _calculate_score(exps, primes):
    """
    Calculate the exact score tau_perp(n) for a given exponent vector.
    Uses a set-based approach to count adjacent coprime pairs in the sorted divisor list.
    To be fast, we generate divisors, sort them, and check gcd.
    For n <= 10^18, the number of divisors is at most ~100k (for highly composite numbers),
    so this is fast enough for BnB nodes if we prune well.
    """
    n = 1
    for i, p in enumerate(primes):
        e = exps[i]
        if e > 0:
            n *= p ** e
    
    if n < 2 or n > MAX_N:
        return 0

    # Generate divisors
    divs = [1]
    for i, p in enumerate(primes):
        e = exps[i]
        if e == 0:
            continue
        powers = [1]
        for _ in range(e):
            powers.append(powers[-1] * p)
        new_divs = []
        for d in divs:
            for pw in powers:
                new_divs.append(d * pw)
        divs = new_divs
        
    divs.sort()
    
    if len(divs) < 2:
        return 0
        
    score = 0
    prev = divs[0]
    for i in range(1, len(divs)):
        curr = divs[i]
        if math.gcd(prev, curr) == 1:
            score += 1
        prev = curr
        
    return score

def _upper_bound(exps, primes, remaining_primes_idx, current_n, current_div_count, current_coprime_boundary):
    """
    Estimate the maximum possible score for the subtree rooted at this state.
    
    The score is bounded by the number of divisors that are "coprime to the largest prime factor used".
    Actually, a tighter bound:
    Let P be the set of primes used.
    The number of adjacent coprime pairs is at most the number of divisors d such that
    there exists a neighbor d' with gcd(d, d')=1.
    
    A known upper bound for tau_perp(n) is related to the number of divisors coprime to the
    largest prime factor p_max. Let n = p_max^a * m where gcd(m, p_max)=1.
    The divisors of n are {d * p_max^k}.
    Two adjacent divisors in the sorted list can be coprime only if they don't share any prime factor.
    
    A simpler, safe upper bound for pruning:
    The score cannot exceed the total number of divisors tau(n).
    Also, score <= number of divisors of m (the part coprime to p_max) * something?
    
    Let's use a heuristic bound:
    Max score <= tau(n)
    And specifically, if we add more primes, tau increases.
    
    A better bound:
    Consider the "boundary" between divisors divisible by p_max and those not.
    Actually, let's just use tau(n) as a loose bound and rely on the fact that we are
    maximizing. If current_best > tau(upper_bound_n), prune.
    
    To make it tighter:
    The number of coprime adjacent pairs is at most the number of divisors of n that are
    coprime to at least one "large" prime?
    
    Let's stick to a simple but effective bound:
    Upper bound on score <= number of divisors of the "coprime part" relative to the largest prime.
    If n = p^a * m, gcd(p,m)=1.
    The divisors are d_i * p^j.
    A pair (d_i * p^j, d_{i+1} * p^{j'}) is coprime only if j=0 and j'=0 is not possible for adjacent?
    Actually, if j > 0, the divisor is divisible by p. If j' > 0, it's divisible by p.
    So any pair where both are divisible by p has gcd >= p > 1.
    Thus, coprime pairs can ONLY occur between:
    1. Two divisors not divisible by p (i.e., from m).
    2. One divisible by p and one not? No, if one is divisible by p and the other isn't,
       their gcd could still be 1 if they share no other primes.
       
    However, a strong upper bound is:
    Score <= tau(m) + (number of transitions between p-divisible and non-p-divisible blocks)
    This is getting complex.
    
    Let's use a different structural insight:
    For a fixed set of primes, the score is maximized when exponents are small?
    Usually, exponents > 1 on small primes (2, 3) create "clumps" of divisors that share those primes,
    reducing the chance of coprime adjacency.
    So, a good heuristic bound:
    Assume all exponents are 1 (square-free) for the upper bound calculation of the "potential"
    of the prime set, but allow the current exponents to be evaluated exactly.
    
    Actually, the most effective BnB here is to search over the **set of primes** and **exponents**.
    Since the number of primes is small (<=15), we can iterate over subsets? No, 2^15 = 32768, which is small.
    But exponents vary.
    
    Let's structure the search:
    1. Iterate over possible "largest prime" p_k.
    2. For each p_k, determine the max exponent a for p_k such that p_k^a <= MAX_N.
    3. For the remaining budget MAX_N / p_k^a, find the best "m" from smaller primes.
    
    This recursive decomposition is efficient.
    
    Refined Strategy:
    Recursive function solve(k, limit, current_n, current_exps, best_score_holder):
    k: index of next prime to consider (from large to small? or small to large?)
    Let's go from large to small to fix the "boundary" prime early.
    Actually, going small to large is standard for factorization.
    
    Let's try a different angle:
    The score is roughly proportional to the number of "independent" prime factors.
    The best candidates are likely square-free or near-square-free.
    
    Let's implement a Branch and Bound that:
    1. Starts with no primes.
    2. Adds primes one by one (smallest to largest).
    3. For each prime, tries exponents 1, 2, ... up to max allowed.
    4. Prunes if:
       - n > MAX_N
       - Upper Bound < Current Best
    
    Upper Bound Calculation:
    Let current state have primes P = {p_1, ..., p_k} with exponents e_1, ..., e_k.
    Let remaining primes be R.
    The maximum additional score we can get is bounded by the number of divisors of the
    product of remaining primes (assuming exponent 1 for all, which maximizes diversity).
    Actually, a simpler bound:
    Total Score <= Number of Divisors of (current_n * product_of_all_remaining_primes).
    If this upper bound is <= current_best, prune.
    
    This bound is loose but safe. To tighten, we can note that adding a prime p
    multiplies the divisor count by (e+1).
    
    Let's refine the bound:
    The score cannot exceed the number of divisors of n.
    tau(n) = prod(e_i + 1).
    If we are at step k, and we have fixed e_1..e_k, and we assume we will add all remaining
    primes with exponent 1 (which is usually optimal for maximizing coprime pairs),
    then tau_max = (prod_{i=1..k} (e_i+1)) * (2^(num_remaining_primes)).
    
    If tau_max <= current_best, we can prune.
    """
    pass # Implemented in the main loop below

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin for 150s limit
    
    # State
    best_n = 2
    best_score = 1 # gcd(1,2)=1
    
    # Precompute powers of primes for quick access
    # max_exps[p] = max exponent such that p^exp <= MAX_N
    max_exps = {}
    for p in CANDIDATE_PRIMES:
        max_e = 0
        prod = 1
        while prod * p <= MAX_N:
            prod *= p
            max_e += 1
        max_exps[p] = max_e
        
    # We will perform a Depth First Search (DFS) with pruning.
    # Order: Smallest primes first?
    # If we go smallest first, we fix the "core" quickly.
    # If we go largest first, we fix the "boundary" quickly.
    # Let's try Smallest First, as it constrains the total product quickly.
    
    current_exps = [0] * len(CANDIDATE_PRIMES)
    current_n = 1
    
    # Helper to calculate tau quickly
    def calc_tau(exps):
        t = 1
        for e in exps:
            t *= (e + 1)
        return t
    
    # Helper to calculate upper bound on score
    # Score <= tau(n)
    # Also, Score <= number of divisors of n that are coprime to the largest prime factor?
    # Let's use tau(n) as the primary bound for now, it's easy.
    
    def dfs(idx, current_n, current_exps, current_tau):
        nonlocal best_n, best_score
        
        if time.time() > deadline:
            return
            
        # Pruning:
        # If current_tau (which is an upper bound for the final tau if we stop here,
        # but we can still add primes) is already <= best_score, we might still improve
        # if we add primes? No, adding primes increases tau.
        # So we need the UPPER bound of tau for the COMPLETED number.
        
        # Calculate max possible tau if we add all remaining primes with exponent 1.
        # This is a valid upper bound for the number of divisors.
        # And Score <= tau(n).
        
        remaining_primes_count = len(CANDIDATE_PRIMES) - idx
        # Max additional factor to tau is 2^remaining (if we add all with exp 1)
        # But we are limited by MAX_N.
        
        # A tighter bound:
        # The product of remaining primes (smallest ones) will be added.
        # Let's just use:
        # Upper Bound Score <= current_tau * (2 ** remaining_primes_count)
        # This is an overestimate because we might not be able to add all primes.
        # But it's a safe upper bound.
        
        ub_tau = current_tau * (2 ** remaining_primes_count)
        
        if ub_tau <= best_score:
            return
            
        # If we've considered all primes
        if idx == len(CANDIDATE_PRIMES):
            # Evaluate this candidate
            if current_n > 1: # n must be >= 2
                score = _calculate_score(current_exps, CANDIDATE_PRIMES)
                if score > best_score:
                    best_score = score
                    best_n = current_n
            return
        
        p = CANDIDATE_PRIMES[idx]
        
        # Option 1: Skip this prime (exponent 0)
        dfs(idx + 1, current_n, current_exps, current_tau)
        
        # Option 2: Include this prime with exponent e = 1, 2, ...
        # Max exponent allowed by MAX_N
        max_e_allowed = 0
        temp_n = current_n
        while temp_n * p <= MAX_N and max_e_allowed < max_exps[p]:
            max_e_allowed += 1
            temp_n *= p
            
        # Try exponents from 1 to max_e_allowed
        # Heuristic: Try smaller exponents first?
        # Usually exponent 1 is best for coprime structure.
        for e in range(1, max_e_allowed + 1):
            new_n = current_n * (p ** e)
            if new_n > MAX_N:
                break
                
            new_tau = current_tau * (e + 1)
            
            # Check if even the max possible tau from here can beat best
            # (Already checked in dfs entry, but let's be safe)
            
            current_exps[idx] = e
            dfs(idx + 1, new_n, current_exps, new_tau)
            current_exps[idx] = 0 # Reset
            
        # Note: The order of trying exponents matters for finding good solutions early.
        # Let's try e=1 first, then e=2, etc.

    # Initialize
    dfs(0, 1, current_exps, 1)
    
    return best_n

# EVOLVE-BLOCK-END
