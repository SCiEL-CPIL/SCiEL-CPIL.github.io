import math
import time
import random

MAX_N = 10**18
DEADLINE = 140.0 # Safety margin for 150s limit

# Generate primes up to 200
def generate_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = generate_primes(200)

def get_divisors(primes_list):
    """
    primes_list: list of primes (exponents are 1)
    Returns sorted list of divisors
    """
    divs = [1]
    for p in primes_list:
        new_divs = []
        for d in divs:
            new_divs.append(d)
            new_divs.append(d * p)
        divs = new_divs
    divs.sort()
    return divs

def score_from_divs(divs):
    c = 0
    for i in range(1, len(divs)):
        if math.gcd(divs[i-1], divs[i]) == 1:
            c += 1
    return c

def evaluate_set(primes_set):
    """
    primes_set: set of primes
    Returns (score, n)
    """
    n = 1
    for p in primes_set:
        n *= p
        if n > MAX_N:
            return -1, n
    # Convert to list for get_divisors
    p_list = list(primes_set)
    divs = get_divisors(p_list)
    s = score_from_divs(divs)
    return s, n

def construct():
    start_time = time.time()
    
    # Initial best: First 15 primes
    init_primes = PRIMES[:15]
    init_set = set(init_primes)
    best_score, best_n = evaluate_set(init_set)
    best_set = init_set
    
    # Local search
    # We try to swap a prime in the set with a prime not in the set
    # We maintain a list of primes to try.
    # To be efficient, we only consider primes that could fit.
    
    # Precompute prefix products to quickly check validity?
    # Or just check n * p_new // p_old <= MAX_N
    
    # We'll do multiple passes of local search
    improved = True
    while improved and time.time() - start_time < DEADLINE:
        improved = False
        current_set = set(best_set)
        current_score = best_score
        
        # Get current product
        current_n = 1
        for p in current_set:
            current_n *= p
            
        # Try all swaps
        # For each p_in in current_set
        # For each p_out in PRIMES (not in current_set)
        # If current_n * p_out // p_in <= MAX_N:
        #    Try swap
        
        # Optimization: Sort primes in current_set
        sorted_in = sorted(current_set)
        
        # We can limit p_out to be > p_in to avoid symmetric swaps?
        # No, we might want to swap a large prime for a small one?
        # But we start with small primes. Swapping small for large increases product.
        # Swapping large for small decreases product, allowing other swaps later?
        # But we want to maximize score.
        # Let's allow any swap.
        
        for p_in in sorted_in:
            # Prune: if p_in is very small, p_out can be large.
            # if p_in is very large, p_out must be small.
            max_p_out = current_n * 100 // p_in # Heuristic upper bound
            # Actually, just check the condition.
            
            for p_out in PRIMES:
                if p_out in current_set:
                    continue
                if p_out == p_in:
                    continue
                
                # Check product
                # current_n * p_out // p_in <= MAX_N
                # To avoid overflow or floating point issues, use integer math
                # current_n // p_in * p_out <= MAX_N
                # This is exact since p_in divides current_n
                
                new_n = (current_n // p_in) * p_out
                if new_n > MAX_N:
                    continue
                if new_n < 2:
                    continue
                
                # Evaluate
                new_set = current_set.copy()
                new_set.remove(p_in)
                new_set.add(p_out)
                
                s, n = evaluate_set(new_set)
                
                if s > best_score:
                    best_score = s
                    best_n = n
                    best_set = new_set
                    current_set = new_set
                    current_n = n
                    current_score = s
                    improved = True
                    # Restart search from new best
                    # Break out of loops to restart while loop
                    break
            
            if improved:
                break
            
    return best_n

construct()
