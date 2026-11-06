import math
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 120s limit
    
    max_val = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    best_score = -1
    best_n = 2
    
    def compute_tau_perp(n):
        """Compute tau_perp(n) exactly."""
        # Factorize n
        factors = []
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                cnt = 0
                while temp % p == 0:
                    temp //= p
                    cnt += 1
                factors.append((p, cnt))
        if temp > 1:
            factors.append((temp, 1))
        
        # Generate all divisors
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        
        divs.sort()
        
        # Count adjacent coprime pairs
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i + 1]) == 1:
                score += 1
        return score
    
    def try_candidate(n):
        nonlocal best_score, best_n
        if n < 2 or n > max_val:
            return
        if time.time() > deadline:
            return
        try:
            s = compute_tau_perp(n)
            if s > best_score:
                best_score = s
                best_n = n
        except:
            pass
    
    # Strategy 1: Primorials and near-primorials
    # P_k * m where m uses small primes
    primorials = [1]
    for p in primes:
        primorials.append(primorials[-1] * p)
    
    # Try primorials directly
    for pk in primorials:
        if pk > max_val:
            break
        try_candidate(pk)
    
    # Try P_k * 2^a for small a
    for k in range(1, len(primorials)):
        pk = primorials[k]
        if pk > max_val // 2:
            break
        for a in range(1, 6):
            n = pk * (2 ** a)
            if n <= max_val:
                try_candidate(n)
            else:
                break
    
    # Try P_k * 3^b
    for k in range(1, len(primorials)):
        pk = primorials[k]
        if pk > max_val // 3:
            break
        for b in range(1, 5):
            n = pk * (3 ** b)
            if n <= max_val:
                try_candidate(n)
            else:
                break
    
    # Try P_k * 2^a * 3^b
    for k in range(1, len(primorials)):
        pk = primorials[k]
        if pk > max_val // 4:
            break
        for a in range(1, 5):
            if pk * (2 ** a) > max_val // 3:
                break
            for b in range(1, 4):
                n = pk * (2 ** a) * (3 ** b)
                if n <= max_val:
                    try_candidate(n)
                else:
                    break
    
    # Strategy 2: DFS over exponent vectors
    # n = 2^a * 3^b * 5^c * 7^d * ... with non-increasing exponents
    # Prioritize having many distinct primes
    
    def dfs_exponents(idx, max_exp, current_n, current_tau):
        nonlocal best_score, best_n
        if time.time() > deadline:
            return
        
        if current_n > 1:
            try_candidate(current_n)
        
        if idx >= len(primes):
            return
        
        p = primes[idx]
        limit_n = max_val // current_n
        
        e_max = 0
        temp = 1
        while temp * p <= limit_n and e_max < max_exp:
            temp *= p
            e_max += 1
        
        # Try exponents from max down to 0 (prioritize more distinct primes first? 
        # Actually, for tau_perp, having many distinct primes with exponent 1 is key)
        # Let's try all exponents but stop if time is up
        for e in range(e_max, -1, -1):
            if e == 0 and idx > 0:
                # Skip: if we skip a prime, we should stop (non-increasing)
                # Actually in canonical form, once we skip, we stop
                break
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            dfs_exponents(idx + 1, e, next_n, next_tau)
            if time.time() > deadline:
                return
    
    # Start DFS - this explores numbers with non-increasing exponents
    # The key candidates have many primes with exponent 1, then maybe some with exponent 2
    dfs_exponents(0, 60, 1, 1)
    
    # Strategy 3: Targeted search around known good structures
    # Numbers like 2^a * 3^b * 5 * 7 * 11 * 13 * ... 
    # where the tail is a primorial
    
    # For each subset of "large" primes (exponent 1), try small prime powers
    # Tail primorials: product of primes from index i to j
    # Head: 2^a * 3^b * 5^c
    
    for i in range(3, min(16, len(primes))):
        # Tail: primes[i:] all with exponent 1
        tail = 1
        for j in range(i, len(primes)):
            tail *= primes[j]
            if tail > max_val:
                tail //= primes[j]
                break
        
        if tail < 2:
            continue
        
        # Head: 2^a * 3^b * 5^c with small exponents
        for a in range(0, 8):
            h2 = 2 ** a
            if h2 * tail > max_val:
                break
            for b in range(0, 6):
                h3 = 3 ** b
                if h2 * h3 * tail > max_val:
                    break
                for c in range(0, 5):
                    h5 = 5 ** c
                    n = h2 * h3 * h5 * tail
                    if n <= max_val:
                        try_candidate(n)
    
    # Strategy 4: Random sampling in promising regions
    import random
    random.seed(42)
    
    for _ in range(500):
        if time.time() > deadline:
            break
        # Random number with many small prime factors
        n = 1
        for p in primes:
            if n > max_val // p:
                break
            # Random exponent: mostly 1, sometimes 2
            if random.random() < 0.7:
                e = 1
            elif random.random() < 0.5:
                e = 2
            else:
                e = random.randint(1, 3)
            if n * (p ** e) > max_val:
                break
            n *= (p ** e)
        if n >= 2:
            try_candidate(n)
    
    return best_n
