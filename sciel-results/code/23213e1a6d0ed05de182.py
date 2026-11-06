"""Optimized construct for maximizing tau_perp(n) over n <= 10^18."""
import math
import heapq
import time

def _tau_perp_from_exponents(primes, exponents):
    """Compute tau_perp(n) given sorted primes and their exponents."""
    n = 1
    for i in range(len(primes)):
        n *= primes[i] ** exponents[i]
    if n < 2 or n > 10**18:
        return 0
    
    # Generate all divisors
    divisors = [1]
    for i in range(len(primes)):
        p = primes[i]
        e = exponents[i]
        new_divs = []
        for d in divisors:
            val = d
            for k in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    
    divisors.sort()
    score = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            score += 1
    return score

def _n_from_exponents(primes, exponents):
    n = 1
    for i in range(len(primes)):
        if exponents[i] > 0:
            n *= primes[i] ** exponents[i]
    return n

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # First 20 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    nprimes = len(primes)
    log_limit = math.log(10**18)
    
    # Precompute log primes
    log_primes = [math.log(p) for p in primes]
    
    # Initial: all exponents = 1 for as many primes as fit
    # 2*3*5*7*11*13*17*19*23*29*31*37 = 7.42e12, *41 = 3.04e14, *43 = 1.3e16, *47 = 6.2e17, *53 = 3.3e19 > 1e18
    # So first 14 primes with exponent 1 fits: 2*3*5*7*11*13*17*19*23*29*31*37*41*43 = 1.3e16
    # Actually let me compute: product of first k primes
    # primorial: 2=2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870, 6469693230, 200862915270, 7420738134810, 304250263527210, 13082761331670030
    # 13th prime 41: 13082761331670030 * 41 = 5.36e17
    # 14th prime 43: 5.36e17 * 43 = 2.3e19 > 1e18
    # So first 13 primes (up to 41) with exponent 1: 5.36e17
    # Or first 12 primes (up to 37): 7.42e12, lots of room for exponents
    
    # Start with first 13 primes, exponent 1 each
    best_exponents = [1] * 13 + [0] * 7
    best_score = _tau_perp_from_exponents(primes, best_exponents)
    best_n = _n_from_exponents(primes, best_exponents)
    
    # Priority queue: (-score, log_n, exponents_tuple)
    # State: exponents tuple
    initial_log = sum(log_primes[i] * best_exponents[i] for i in range(nprimes))
    heap = [(-best_score, initial_log, tuple(best_exponents))]
    visited = set()
    visited.add(tuple(best_exponents))
    
    # Also try other starting points
    # First 12 primes with some exponents > 1
    start_configs = []
    # Config: first 12 primes, exponents = [2,1,1,...] etc.
    for i in range(12):
        cfg = [1] * 12 + [0] * 8
        cfg[i] = 2
        n_val = _n_from_exponents(primes[:12], cfg[:12])
        if n_val <= 10**18:
            start_configs.append(tuple(cfg))
    
    # Also: fewer primes with higher exponents
    # First 10 primes, exponents [2,2,1,1,...]
    for i in range(10):
        for j in range(i+1, 10):
            cfg = [1] * 10 + [0] * 10
            cfg[i] = 2
            cfg[j] = 2
            n_val = _n_from_exponents(primes[:10], cfg[:10])
            if n_val <= 10**18:
                start_configs.append(tuple(cfg))
    
    for cfg in start_configs:
        if cfg not in visited:
            s = _tau_perp_from_exponents(primes, list(cfg))
            ln = sum(log_primes[i] * cfg[i] for i in range(nprimes))
            if s > best_score:
                best_score = s
                best_exponents = list(cfg)
                best_n = _n_from_exponents(primes, list(cfg))
            heapq.heappush(heap, (-s, ln, cfg))
            visited.add(cfg)
    
    iterations = 0
    while heap and time.time() < deadline:
        iterations += 1
        neg_score, log_n, exps = heapq.heappop(heap)
        score = -neg_score
        
        # Generate neighbors: increment or decrement one exponent, or swap
        # Budget: log_n must stay <= log_limit
        for i in range(nprimes):
            # Try incrementing exponent i
            new_exps = list(exps)
            new_exps[i] += 1
            new_log = log_n + log_primes[i]
            if new_log <= log_limit:
                key = tuple(new_exps)
                if key not in visited:
                    visited.add(key)
                    s = _tau_perp_from_exponents(primes, new_exps)
                    if s > best_score:
                        best_score = s
                        best_exponents = new_exps[:]
                        best_n = _n_from_exponents(primes, new_exps)
                    heapq.heappush(heap, (-s, new_log, key))
            
            # Try decrementing exponent i (if > 0)
            if exps[i] > 0:
                new_exps = list(exps)
                new_exps[i] -= 1
                new_log = log_n - log_primes[i]
                key = tuple(new_exps)
                if key not in visited:
                    visited.add(key)
                    s = _tau_perp_from_exponents(primes, new_exps)
                    if s > best_score:
                        best_score = s
                        best_exponents = new_exps[:]
                        best_n = _n_from_exponents(primes, new_exps)
                    heapq.heappush(heap, (-s, new_log, key))
        
        # Swap moves: decrement i, increment j (i != j)
        for i in range(nprimes):
            if exps[i] > 0:
                for j in range(nprimes):
                    if j == i:
                        continue
                    new_exps = list(exps)
                    new_exps[i] -= 1
                    new_exps[j] += 1
                    new_log = log_n - log_primes[i] + log_primes[j]
                    if new_log <= log_limit:
                        key = tuple(new_exps)
                        if key not in visited:
                            visited.add(key)
                            s = _tau_perp_from_exponents(primes, new_exps)
                            if s > best_score:
                                best_score = s
                                best_exponents = new_exps[:]
                                best_n = _n_from_exponents(primes, new_exps)
                            heapq.heappush(heap, (-s, new_log, key))
        
        if len(visited) > 500000:
            break
    
    return best_n

# Quick test
if __name__ == "__main__":
    n = construct()
    print(f"n = {n}")
    print(f"tau_perp = {_tau_perp_from_exponents([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71], [0]*20)}")
