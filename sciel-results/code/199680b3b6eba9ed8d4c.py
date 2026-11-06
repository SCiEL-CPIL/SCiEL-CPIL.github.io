import math
import time
import random
import numpy as np

def _get_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

_PRIMES = _get_primes(1000000)
_LIMIT = 10**18

def _factorize(n):
    factors = []
    for p in _PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            factors.append((p, exp))
    if n > 1:
        factors.append((n, 1))
    return factors

def _get_divisors(factors):
    divs = [1]
    for p, e in factors:
        new_divs = []
        for d in divs:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    return divs

def _score(n, factors=None):
    if factors is None:
        factors = _factorize(n)
    divs = _get_divisors(factors)
    divs.sort()
    if len(divs) < 2:
        return 0
    d_arr = np.array(divs, dtype=np.int64)
    # Check for overflow in multiplication? n <= 10^18, fits in int64.
    # gcd of adjacent
    g = np.gcd(d_arr[:-1], d_arr[1:])
    return int(np.sum(g == 1))

def _search():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial candidates
    # 1. Primorial 15
    n1 = 1
    for p in _PRIMES[:15]:
        n1 *= p
    f1 = _factorize(n1)
    s1 = _score(n1, f1)
    
    # 2. Highly composite-like
    # Greedy construction for max divisors
    n2 = 1
    f2 = []
    # Try to maximize divisors
    # We can just pick a known good one or construct one.
    # Let's construct: multiply by smallest prime that fits.
    # But to get high divisor count, we need exponents.
    # Simple heuristic:
    # Start with 2^a 3^b ...
    # We'll just use a random candidate with many small primes as a backup.
    n2 = 1
    for p in _PRIMES[:10]:
        # Try to increase exponent
        while n2 * p <= _LIMIT:
             n2 *= p
             if n2 > _LIMIT: break
    # This creates a number like 2^k 3^j ...
    f2 = _factorize(n2)
    s2 = _score(n2, f2)
    
    best_n = n1 if s1 >= s2 else n2
    best_s = max(s1, s2)
    best_f = f1 if s1 >= s2 else f2
    
    # Local Search
    # We will try to improve best_n
    # Moves:
    # 1. Change exponent of a prime
    # 2. Swap a prime with another
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try all single-step moves
        current_f = best_f[:]
        current_n = best_n
        
        # Generate list of moves to try
        moves = []
        
        # Exponent changes
        for i, (p, e) in enumerate(current_f):
            # Increase exponent
            if current_n * p <= _LIMIT:
                new_f = current_f[:]
                new_f[i] = (p, e + 1)
                new_n = current_n * p
                moves.append((new_n, new_f))
            # Decrease exponent
            if e > 1:
                new_f = current_f[:]
                new_f[i] = (p, e - 1)
                new_n = current_n // p
                moves.append((new_n, new_f))
        
        # Prime swaps
        # For each prime p_i, try swapping with q
        # q must be such that new_n <= LIMIT
        # To limit moves, we only try a few random q's or nearby primes
        for i, (p, e) in enumerate(current_f):
            # Max q
            max_q = _LIMIT * p // current_n
            if max_q < 2:
                continue
            
            # Find candidate q's
            # We want q not in current_f
            used_primes = {p for p, _ in current_f}
            
            # Try a few random primes in range
            # We need primes <= max_q
            # And not in used_primes
            
            # Simple strategy: try next few primes and some random ones
            # But we need to be careful with the search space.
            # Let's just try swapping with the smallest available prime not in used_primes?
            # Or try to swap with a prime that is "far" to change structure?
            
            # Let's try 5 random swaps per prime factor
            for _ in range(5):
                # Pick random prime q
                # We need q <= max_q
                # If max_q is small, pick from small primes.
                # If max_q is large, pick random.
                
                # Approximate index in _PRIMES
                # Using binary search would be better, but for simplicity:
                # We'll just pick a random prime from _PRIMES and check if it fits.
                # This is inefficient if max_q is small.
                
                # Better:
                # Find index of p in _PRIMES
                # Try primes in a window around p or smaller/larger
                
                # Let's just try swapping with a random prime from the first 100 primes
                # and see if it fits and is not used.
                q = random.choice(_PRIMES[:100])
                if q in used_primes:
                    continue
                if q > max_q:
                    continue
                
                new_f = current_f[:]
                new_f[i] = (q, e)
                new_n = current_n // p * q
                if new_n > _LIMIT:
                    continue
                moves.append((new_n, new_f))
                
        # Evaluate moves
        best_move_n = best_n
        best_move_s = best_s
        best_move_f = best_f
        
        for n, f in moves:
            if time.time() > deadline - 5:
                break
            s = _score(n, f)
            if s > best_move_s:
                best_move_s = s
                best_move_n = n
                best_move_f = f
                improved = True
        
        if improved:
            best_n = best_move_n
            best_s = best_move_s
            best_f = best_move_f
        else:
            # Stuck. Try a random restart or perturbation
            # Perturb: swap a random prime with a random other prime
            if len(best_f) > 1:
                i = random.randint(0, len(best_f)-1)
                p, e = best_f[i]
                max_q = _LIMIT * p // best_n
                if max_q > 2:
                    # Find a valid q
                    for _ in range(20):
                        q = random.choice(_PRIMES[:100])
                        if q not in {p for p, _ in best_f} and q <= max_q:
                            new_f = best_f[:]
                            new_f[i] = (q, e)
                            new_n = best_n // p * q
                            if new_n <= _LIMIT:
                                s = _score(new_n, new_f)
                                # Accept even if worse? Simulated annealing?
                                # For now, just accept if it's not too bad, or just break out of local search
                                # Let's just break and try a different initial candidate if we have time
                                break
            # If we are stuck, we can try adding a prime if possible
            # But we are likely near the limit.
            # Let's just break the local search loop and try a different seed if time permits.
            # For simplicity, we break.
            break
            
    return best_n

def construct():
    return _search()
