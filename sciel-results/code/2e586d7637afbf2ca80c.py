import math
import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 148.0
    LIMIT = 10**18
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes_pool = sieve(1000)
    
    # We want to choose exactly 15 distinct primes.
    # Let P be the set of chosen primes.
    # The divisors of n = prod p_i^{e_i} are of the form prod p_i^{a_i} where 0 <= a_i <= e_i.
    # We want to maximize the number of adjacent pairs in the sorted list of divisors that are coprime.
    
    # Key Insight:
    # Consider the divisors sorted by value.
    # Two divisors d, d' are adjacent.
    # If we look at the structure of divisors, they form a grid in log-space or similar.
    # Actually, there is a known result for this specific problem "max coprime adjacent divisors".
    # Let's analyze small cases.
    # n = p^e. Divisors: 1, p, p^2, ..., p^e.
    # Pairs: (1, p) gcd 1. (p, p^2) gcd p. ...
    # Score = 1.
    
    # n = p q. Divisors: 1, p, q, pq (assuming p<q).
    # Pairs: (1, p) gcd 1. (p, q) gcd 1 (if p!=q). (q, pq) gcd q.
    # Score = 2.
    
    # n = p q r. Divisors: 1, p, q, r, pq, pr, qr, pqr (assuming p<q<r).
    # Sorted: 1, p, q, r, pq, pr, qr, pqr.
    # Pairs:
    # (1, p): 1
    # (p, q): 1
    # (q, r): 1
    # (r, pq): gcd(r, pq)=1 (since p,q,r distinct primes). 1
    # (pq, pr): gcd(pq, pr)=p. 0
    # (pr, qr): gcd(pr, qr)=r. 0
    # (qr, pqr): gcd(qr, pqr)=qr. 0
    # Score = 4.
    
    # Let's check n = p^2 q. Divisors: 1, p, p^2, q, pq, p^2 q.
    # Sorted: 1, p, p^2, q, pq, p^2 q.
    # (1, p): 1
    # (p, p^2): 0
    # (p^2, q): 1
    # (q, pq): 0
    # (pq, p^2 q): 0
    # Score = 2.
    # So exponents > 1 generally reduce score unless they allow "skipping" over non-coprime neighbors?
    # Actually, higher exponents add more divisors that are multiples of p, creating clusters of non-coprime divisors.
    # It is widely suspected that the optimal n is square-free.
    # If n is square-free with k primes, the divisors correspond to subsets of {1..k}.
    # The sorted order of divisors depends on the values of the primes.
    
    # Let's verify if square-free is always optimal.
    # If we increase exponent of p from 1 to 2, we add divisors d*p where d|n/p.
    # These new divisors are multiples of p.
    # In the sorted list, they will be interleaved.
    # Usually, this creates more adjacent pairs with gcd > 1.
    # Given the time constraint and the score 21.9, which is close to the number of primes (15) * something?
    # Wait, the current score is 21.9184.
    # If n is product of 15 primes, score is roughly 2*k - 2?
    # For k=3, score 4. 2*3-2 = 4.
    # For k=2, score 2. 2*2-2 = 2.
    # For k=1, score 1. 2*1-1 = 1? No, 2*1-2=0.
    # Let's check k=4. p<q<r<s.
    # Divisors: 1, p, q, r, s, pq, pr, ps, qr, qs, rs, pqr, pqs, prs, qrs, pqrs.
    # Order depends on primes.
    # If primes are close, order is roughly by sum of exponents (weight).
    # But values matter.
    
    # Hypothesis: The optimal n is the product of the first 15 primes (primorial).
    # Let's calculate the score for the primorial of 15 primes.
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47.
    # Product is ~ 6.1e17.
    
    # Can we do better with a different set of primes?
    # If we make primes closer together, the divisors are more "interleaved" in a way that might increase coprime pairs?
    # Or if we make them further apart?
    
    # Let's try to compute the score for a square-free number with k primes efficiently.
    # We can't enumerate all divisors if k=15 (2^15 = 32768, which is fine).
    # 32768 divisors. Sorting takes 32768 log 32768 ~ 500k ops.
    # We can afford to evaluate ~1000 candidates in 150s.
    
    # Strategy:
    # 1. Assume square-free.
    # 2. Search for the best set of 15 primes.
    # 3. The "natural" choice is the first 15 primes.
    # 4. Try swapping out large primes for smaller ones? No, we need 15 distinct primes.
    # 5. Try swapping in primes that are "better spaced"?
    
    # Let's write a fast evaluator for square-free n.
    
    def eval_square_free(primes):
        """
        Calculate tau_perp for n = prod(p for p in primes).
        primes is a list of 15 distinct primes.
        """
        k = len(primes)
        # Generate all divisors
        divs = [1]
        for p in primes:
            # For each existing divisor d, add d*p
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

    # Initial best: first 15 primes
    initial_primes = primes_pool[:15]
    best_score = eval_square_free(initial_primes)
    best_primes = initial_primes[:]
    
    # Hill Climbing on the set of primes
    # We represent the state as a tuple of 15 primes.
    # Moves:
    # 1. Swap one prime with another prime from the pool (not in current set).
    #    Constraint: Product must be <= LIMIT.
    #    Since we start with the smallest 15, we have room to swap in larger primes?
    #    Product of first 15 is 6.1e17. Limit 1e18.
    #    We can swap a small prime p_i for a larger prime p_j if p_j < p_i * (1e18 / product).
    #    Actually, just check product < LIMIT.
    
    current_primes = best_primes[:]
    current_score = best_score
    
    T = 10.0
    cooling = 0.99
    
    # Precompute products? No, just calculate.
    
    # To speed up, we can cache scores?
    # The state space is huge.
    
    # Let's try a few random restarts or just one long annealing run.
    
    while time.time() < DEADLINE:
        # Generate a neighbor
        idx = random.randint(0, 14)
        p_old = current_primes[idx]
        
        # Find candidate primes to swap in
        # We want p_new not in current_primes
        # And product(new_primes) <= LIMIT
        
        # Current product
        prod = 1
        for p in current_primes:
            prod *= p
        
        # Max p_new
        max_p_new = int((LIMIT * p_old) / prod)
        
        # Find primes in pool not in current_primes and <= max_p_new
        # To make it faster, we can just pick a random prime from the pool
        # and check.
        
        found = False
        for _ in range(20): # Try 20 random candidates
            p_new = random.choice(primes_pool)
            if p_new in current_primes:
                continue
            if p_new > max_p_new:
                continue
            
            new_primes = current_primes[:]
            new_primes[idx] = p_new
            
            # Check product again to be safe (integer math)
            new_prod = prod // p_old * p_new
            if new_prod > LIMIT:
                continue
            
            score_new = eval_square_free(new_primes)
            
            delta = score_new - current_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_primes = new_primes
                current_score = score_new
                found = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_primes = current_primes[:]
            break # Only try one move per iteration for simplicity? Or loop?
            # If we break, we do one move per loop iteration.
            # If we don't break, we might do many moves.
            # Let's do one move per iteration to control time.
            
        # Cooling
        T *= cooling
        if T < 0.1:
            T = 0.1

    # Construct n from best_primes
    n = 1
    for p in best_primes:
        n *= p
        
    return n

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
