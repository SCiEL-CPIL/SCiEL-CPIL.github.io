import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill
    
    # Primes list
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def tau_perp(n):
        """Compute the number of adjacent coprime pairs among divisors of n."""
        # Generate all divisors
        # Factor n
        temp = n
        factors = {}
        for p in primes:
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
        
        # Generate divisors
        divisors = [1]
        for p, e in factors.items():
            new_divs = []
            pe = 1
            for i in range(e + 1):
                for d in divisors:
                    new_divs.append(d * pe)
                pe *= p
            divisors = new_divs
        
        divisors.sort()
        count = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i + 1]) == 1:
                count += 1
        return count
    
    # We know squarefree 15 primes gives ~21.92
    # 2^4 * prod(14 primes) gives ~22.63
    # Let's search over: n = 2^a * 3^b * 5^c * ... with remaining primes squarefree
    
    # Strategy: fix the set of primes used, vary exponents of small primes
    # The product must be <= 10^18
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47  # fallback
    
    # First, let's try: 2^a * (product of next k-1 primes)
    # where the k-1 primes are 3,5,7,...,p_k
    
    # For a given set of primes with exponents, compute n and check <= 10^18
    # Then compute tau_perp
    
    # Let's be systematic. Try different numbers of "extra" exponents on small primes.
    
    # Case 1: Only 2 gets extra exponent
    # n = 2^a * 3 * 5 * 7 * ... * p_k
    # Product of first k primes with 2^a: 2^(a-1) * prod(primes[1:k])
    # Must be <= 10^18
    
    # Precompute product of primes[1:] (i.e., 3*5*7*...)
    # primes[1:] = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, ...]
    
    # Let's try various configurations
    
    configurations = []
    
    # Configuration type 1: 2^a * prod of next k-1 smallest primes
    # a >= 1, and the primes are 3, 5, 7, ..., up to some limit
    for a in range(1, 20):
        # Start with just 2^a, add primes one by one
        n_val = 2 ** a
        k = 1  # number of primes used (including 2)
        for i in range(1, len(primes)):  # primes[1:] = 3,5,7,...
            p = primes[i]
            if n_val * p > 10**18:
                break
            n_val *= p
            k += 1
            if time.time() > deadline - 10:
                break
            # Evaluate this configuration
            score = tau_perp(n_val)
            if score > best_score:
                best_score = score
                best_n = n_val
        if time.time() > deadline - 10:
            break
    
    # Configuration type 2: 2^a * 3^b * prod of next primes
    for a in range(1, 15):
        for b in range(1, 10):
            n_val = (2 ** a) * (3 ** b)
            if n_val > 10**18:
                break
            # Add more primes
            for i in range(2, len(primes)):
                p = primes[i]
                if n_val * p > 10**18:
                    break
                n_val *= p
                if time.time() > deadline - 10:
                    break
                score = tau_perp(n_val)
                if score > best_score:
                    best_score = score
                    best_n = n_val
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    # Configuration type 3: 2^a * 3^b * 5^c * prod of next primes
    for a in range(1, 12):
        for b in range(1, 8):
            for c in range(1, 6):
                n_val = (2 ** a) * (3 ** b) * (5 ** c)
                if n_val > 10**18:
                    break
                for i in range(3, len(primes)):
                    p = primes[i]
                    if n_val * p > 10**18:
                        break
                    n_val *= p
                    if time.time() > deadline - 10:
                        break
                    score = tau_perp(n_val)
                    if score > best_score:
                        best_score = score
                        best_n = n_val
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    # Also try: pure squarefree with different prime sets (maybe skipping a small prime for a larger one?)
    # Actually, for squarefree, using the smallest primes is optimal for the lattice structure
    # But let's verify: try 15 primes but not the first 15
    # This is unlikely to help, skip for now
    
    # Try: 2^a * 3^b * 5^c * 7^d * prod of next primes
    for a in range(1, 10):
        for b in range(1, 7):
            for c in range(1, 5):
                for d in range(1, 4):
                    n_val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                    if n_val > 10**18:
                        break
                    for i in range(4, len(primes)):
                        p = primes[i]
                        if n_val * p > 10**18:
                            break
                        n_val *= p
                        if time.time() > deadline - 10:
                            break
                        score = tau_perp(n_val)
                        if score > best_score:
                            best_score = score
                            best_n = n_val
                    if time.time() > deadline - 10:
                        break
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    # Final safety: ensure we return a valid n
    if best_n < 2 or best_n > 10**18:
        best_n = 614889782588491410  # known valid
    
    return best_n
