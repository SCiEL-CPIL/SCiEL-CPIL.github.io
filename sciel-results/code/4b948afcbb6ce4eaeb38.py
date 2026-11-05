import math
import time
from functools import lru_cache

def _sieve(n):
    """Return a list of primes up to n."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

# We need primes up to a certain limit. 
# The product of the first 15 primes is ~6.1e17. 
# The product of the first 16 primes is ~3.2e19 > 10^18.
# So we need at most 15 primes.
# Let's generate primes up to 100 to be safe.
PRIMES = _sieve(100)

def generate_candidates(limit=10**18, max_primes=15):
    """
    Generate all exponent vectors (a1, ..., ak) such that 
    p1^a1 * ... * pk^ak <= limit and a1 >= a2 >= ... >= ak >= 1.
    Returns a list of tuples: (n, tau, exponents)
    """
    candidates = []
    
    def backtrack(idx, max_exp, current_n, current_tau, exponents):
        if current_n > limit:
            return
        if idx >= max_primes:
            # We can stop if we've used all allowed primes, 
            # but we could also use fewer primes with higher exponents.
            # The recursion handles this by allowing idx to go up to max_primes.
            # Actually, we should record the candidate if exponents is not empty.
            if exponents:
                candidates.append((current_n, current_tau, tuple(exponents)))
            return
        
        p = PRIMES[idx]
        
        # Try exponents from max_exp down to 1
        # We can also stop trying exponents if current_n * p > limit
        # But since we are multiplying, we just check inside the loop.
        
        # Optimization: If current_n * p > limit, we can't use this prime at all.
        # But we might still use later primes? No, primes are increasing.
        # If p is too big, all subsequent primes are bigger. So we can stop.
        # However, we might have skipped this prime and used a later one? No, the sequence must be contiguous from p1=2.
        # So if p^1 * current_n > limit, we cannot use p or any prime > p.
        
        # The condition a_i >= a_{i+1} means max_exp for next is the current chosen exponent.
        # The condition a_i >= 1.
        
        # We can also choose to stop adding primes here.
        # This is handled by the base case or by returning early.
        # Actually, the base case `if idx >= max_primes` is not the only exit.
        # We should record the candidate at every step? No, because we can always add more primes.
        # But adding more primes increases tau. So the "best" candidates will likely use as many primes as possible or high exponents.
        # To be safe, we should record the candidate at every valid step in the recursion, 
        # but since we are going deeper (adding more primes), the deeper nodes have more factors.
        # Wait, if I stop at idx=3, I have a candidate with 3 primes.
        # If I continue to idx=4, I have a candidate with 4 primes.
        # Both are valid.
        # So I should record the candidate BEFORE recursing? Or at the leaf?
        # If I record at the leaf, I miss the candidates that stop earlier.
        # So I should record the candidate at every valid state (idx, max_exp, current_n, current_tau, exponents).
        
        if exponents:
             candidates.append((current_n, current_tau, tuple(exponents)))

        # Try adding the current prime with exponent e
        for e in range(max_exp, 0, -1):
            new_n = current_n * (p ** e)
            if new_n > limit:
                continue
            new_tau = current_tau * (e + 1)
            exponents.append(e)
            backtrack(idx + 1, e, new_n, new_tau, exponents)
            exponents.pop()
            
    backtrack(0, 10**9, 1, 1, []) # Initial max_exp can be large, bounded by log2(limit)
    
    return candidates

def compute_score(n):
    """Compute tau_perp for n."""
    if n < 2:
        return 0
    # Factorize n
    temp = n
    factors = []
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            cnt = 0
            while temp % d == 0:
                temp //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1 if d == 2 else 2
    if temp > 1:
        factors.append((temp, 1))
    
    # Generate divisors
    divs = [1]
    for p, e in factors:
        current = list(divs)
        mult = 1
        for _ in range(e):
            mult *= p
            divs.extend([d * mult for d in current])
    
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i + 1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    # Generate all valid exponent candidates
    # This step is fast.
    candidates = generate_candidates(10**18, 15)
    
    # Sort by number of divisors (tau) descending
    # Candidates with higher tau are more likely to have higher tau_perp
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
    
    # Evaluate top candidates
    # We don't know how many to evaluate, so we loop until time runs out.
    # To avoid re-evaluating the same n, we can use a set, but n is unique per exponent vector.
    
    # Optimization: If the time per evaluation is ~5ms, we can do ~2000 in 10s.
    # Let's just iterate.
    
    for n, tau, exponents in candidates:
        if time.time() > deadline:
            break
        
        score = compute_score(n)
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n
