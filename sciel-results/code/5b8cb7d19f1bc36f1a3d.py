import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit
    LIMIT = 10**18
    
    # Generate a pool of primes. 
    # We know we can have at most 15 distinct prime factors.
    # The product of the first 15 is ~6e17.
    # We can swap in larger primes as long as product <= 10^18.
    # Let's generate primes up to 300 to have a good pool.
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes_pool = sieve(300)
    
    def calc_score(n, prime_factors):
        """
        Calculate tau_perp(n) for n with given prime_factors (list of (p, e)).
        """
        divisors = [1]
        for p, e in prime_factors:
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divisors:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divisors = new_divs
            
        divisors.sort()
        
        score = 0
        prev = divisors[0]
        for i in range(1, len(divisors)):
            curr = divisors[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
            
        return score

    def get_prime_factors(exps, primes_used):
        pf = []
        for i, p in enumerate(primes_used):
            if exps[i] > 0:
                pf.append((p, exps[i]))
        return pf

    def calc_n(exps, primes_used):
        n = 1
        for i, p in enumerate(primes_used):
            if exps[i] > 0:
                n *= (p ** exps[i])
        return n

    best_score = -1
    best_n = 1
    best_state = None
    
    # Initial state: smallest 15 primes, exponents 1
    initial_primes = primes_pool[:15]
    initial_exps = [1] * 15
    n_init = calc_n(initial_exps, initial_primes)
    if n_init <= LIMIT:
        score_init = calc_score(n_init, get_prime_factors(initial_exps, initial_primes))
        best_score = score_init
        best_n = n_init
        best_state = (initial_primes[:], initial_exps[:])
    
    # Hill Climbing
    # State: (primes_used, exps)
    # primes_used is a list of 15 primes (indices into primes_pool or just the values)
    # Let's represent state as (tuple of primes, tuple of exps)
    
    def neighbor(state):
        primes_used, exps = state
        n_current = calc_n(exps, primes_used)
        
        # Move 1: Swap a prime
        idx = random.randint(0, 14)
        p_old = primes_used[idx]
        # Find a replacement prime not in primes_used
        # To be efficient, pick a random index from pool
        # But we must ensure it's not already in the set
        # And that the new product fits
        # p_new = p_old * (LIMIT / n_current) is max ratio?
        # n_new = n_current / p_old * p_new <= LIMIT
        # p_new <= LIMIT * p_old / n_current
        
        max_p_new = int((LIMIT * p_old) / n_current)
        
        # Pick a random prime from pool that is not in primes_used and <= max_p_new
        # For efficiency, we can just try a few random swaps
        candidates = []
        for p in primes_pool:
            if p in primes_used:
                continue
            if p <= max_p_new:
                candidates.append(p)
        
        if candidates:
            p_new = random.choice(candidates)
            new_primes = primes_used[:]
            new_primes[idx] = p_new
            new_exps = exps[:]
            # Exponent of the new prime at this position is kept as is? 
            # Or reset to 1?
            # Swapping primes changes the value. 
            # If we keep the exponent, n changes by factor p_new/p_old.
            # If p_new > p_old, n increases. We checked limit.
            return (new_primes, new_exps)
            
        # Move 2: Change exponent
        idx = random.randint(0, 14)
        p = primes_used[idx]
        e_old = exps[idx]
        
        # Increase
        if e_old < 20: # Arbitrary cap
            if n_current * p <= LIMIT:
                new_exps = exps[:]
                new_exps[idx] = e_old + 1
                return (primes_used, new_exps)
        
        # Decrease
        if e_old > 1:
            new_exps = exps[:]
            new_exps[idx] = e_old - 1
            return (primes_used, new_exps)
            
        # If no move possible, return None
        return None

    current_state = best_state
    current_score = best_score
    
    T = 10.0
    cooling = 0.99
    
    while time.time() < DEADLINE:
        next_state = neighbor(current_state)
        if next_state is None:
            # Restart if stuck
            current_state = best_state
            current_score = best_score
            continue
            
        primes_next, exps_next = next_state
        n_next = calc_n(exps_next, primes_next)
        
        if n_next > LIMIT or n_next < 2:
            continue
            
        score_next = calc_score(n_next, get_prime_factors(exps_next, primes_next))
        
        delta = score_next - current_score
        
        if delta >= 0 or random.random() < math.exp(delta / T):
            current_state = next_state
            current_score = score_next
            n_current = n_next
            
            if score_next > best_score:
                best_score = score_next
                best_n = n_next
                best_state = next_state
                
        T *= cooling
        if T < 0.1:
            T = 0.1

    return best_n

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
