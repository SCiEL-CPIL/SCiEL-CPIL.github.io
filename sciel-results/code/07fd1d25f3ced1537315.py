import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Generate small primes for the CRT sieve and factorization
    # We need primes up to a certain limit to build the covering set Q.
    # For p up to 10^9, p^2 is up to 10^18.
    # Small primes up to 1000 are sufficient for high-density coverage.
    limit_small = 1000
    sieve = [True] * (limit_small + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(math.isqrt(limit_small)) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    # Precompute quadratic residues for small primes
    # q_res[q] = set of residues r such that x^2 = r mod q
    q_res = {}
    for q in small_primes:
        if q == 2:
            q_res[q] = {0, 1}
        else:
            res = set()
            for x in range(q):
                res.add((x * x) % q)
            q_res[q] = res

    # Helper: Deterministic Miller-Rabin for n < 2^64
    def is_prime_mr(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Helper: Pollard-Brent factorization
    def pollard_brent(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
        y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), random.randint(1, n - 1)
        g, r, q = 1, 1, 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        return g

    def factor(n):
        factors = []
        stack = [n]
        while stack:
            x = stack.pop()
            if x == 1: continue
            if is_prime_mr(x):
                factors.append(x)
            else:
                d = pollard_brent(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    def is_p_smooth(n, p):
        if n == 1: return True
        temp = n
        for q in small_primes:
            if q > p: break
            if q * q > temp: break
            while temp % q == 0:
                temp //= q
        if temp == 1: return True
        if is_prime_mr(temp): return temp <= p
        factors = factor(temp)
        return all(f <= p for f in factors)

    best_k = 0
    best_p = 3

    # Phase 1: Quick check for small primes to establish a baseline
    # We can do this very fast using the q_res table to skip obvious failures.
    for p in small_primes:
        if time.time() > deadline:
            break
        base = p * p
        k = 0
        for offset in range(1, 201):
            if time.time() > deadline:
                break
            if is_p_smooth(base + offset, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: CRT-based Search for Larger Primes
    # Strategy:
    # We want to find p such that p^2 + 1, ..., p^2 + K are "easy" to factor or smooth.
    # A sufficient condition for p^2 + i to have a small factor is that p^2 = -i (mod q) for some small q.
    # Let's try to construct p such that for a target length K, each i in 1..K is "covered" by some q in a set Q.
    # We don't need a perfect cover for ALL i, just enough to make the numbers "loose" or to guide the search.
    # Actually, if p^2 + i is divisible by q, it reduces the size of the remaining factor.
    
    # Let's iterate over potential target lengths K from best_k+1 up to 200.
    # For each K, we try to find a set of residues for p modulo a product of small primes M.
    # This is equivalent to finding a p that satisfies a system of quadratic congruences.
    
    # Instead of solving for a specific K directly which is a hard CSP, 
    # we can sample random "patterns" of divisibility.
    
    # Alternative Structural Change:
    # Use the "Sieve of Eratosthenes" logic on the space of p.
    # For a fixed small prime q, the set of p such that q | p^2 + i is a union of arithmetic progressions.
    # We want to find p that are NOT in the "bad" sets for any i in 1..K?
    # No, we WANT p to be in the "good" sets (where p^2+i has small factors).
    
    # Let's try a different angle: 
    # Randomly select a set of small primes Q.
    # For each q in Q, randomly select a residue r_q such that r_q^2 = -i_q mod q for some i_q.
    # This is getting complicated.
    
    # Let's stick to the "Pre-filtering" inspiration but make it structural.
    # The current program fails because it checks p sequentially.
    # Let's generate candidates p using a "Smooth Number Generator" approach.
    
    # We know that if p^2 + i is p-smooth, it is composed of primes <= p.
    # For large p, the "small" primes are just the first few primes.
    # Let's generate numbers N = p^2 + i that are "very smooth" (only small prime factors) 
    # and check if they form a block.
    
    # Generate "Hamming Numbers" (5-smooth, 7-smooth, etc.) in the range [10^8, 10^12].
    # If we find a block of consecutive 5-smooth numbers, say N, N+1, ..., N+K,
    # then p^2 = N - 1.
    # We need N-1 to be a perfect square of a prime p.
    # This is extremely unlikely.
    
    # Let's go back to the CRT idea but simplified.
    # We will search for p in a specific arithmetic progression defined by CRT.
    # Let M be the product of first few primes, e.g., 2*3*5*7*11*13 = 30030.
    # There are phi(M) possible residues for p modulo M.
    # For each residue r, we can check the "score" of r: how many i in 1..200 
    # have p^2 + i divisible by some small prime q | M?
    # If a residue r has a high score, then any p = r + k*M will have many p^2+i 
    # divisible by small primes.
    # We can then scan p = r + k*M for k=1,2,... and check the full smoothness.
    
    # Step 2a: Precompute "Good Residues"
    # M = 30030 (product of primes up to 13)
    # Primes involved: 2, 3, 5, 7, 11, 13
    M_primes = [2, 3, 5, 7, 11, 13]
    M = 1
    for q in M_primes:
        M *= q
        
    # For each r in 0..M-1, calculate the number of i in 1..200 
    # such that p^2 + i is divisible by at least one q in M_primes.
    # Note: p^2 mod q is fixed for a given r mod q.
    # We need to aggregate this over the CRT components.
    
    # Actually, we can just iterate r from 0 to M-1 and check the condition directly.
    # M is 30030, which is small.
    
    scores = [0] * M
    # Precompute p^2 mod q for all r? 
    # p mod M determines p mod q.
    
    # Let's compute the score for each residue r modulo M.
    # score[r] = count of i in 1..200 such that exists q in M_primes with (r*r + i) % q == 0
    
    # Optimization: 
    # For a fixed q, let R_q be the set of residues r mod q such that r^2 = -i mod q for some i.
    # Actually, for a fixed r, the set of i that are "covered" by q is { (q - r^2) % q }.
    # So for a given r, the covered residues modulo q are just one value: c_q = (q - r^2) % q.
    # The numbers i in 1..200 that are divisible by q AND i = c_q (mod q) are c_q, c_q+q, c_q+2q...
    # Wait, p^2 + i is divisible by q means p^2 = -i (mod q).
    # So i = -p^2 (mod q).
    # So for a fixed p (residue r), the "bad" (i.e., good for us, as they have small factor) 
    # i's are those where i % q == (-r^2) % q.
    
    # Let's calculate the score for each r in 0..M-1.
    # We can do this efficiently.
    
    # For each r, we check i=1..200.
    # 30030 * 200 = 6 million operations. Very fast.
    
    for r in range(M):
        s = 0
        for i in range(1, 201):
            covered = False
            for q in M_primes:
                if (r * r + i) % q == 0:
                    covered = True
                    break
            if covered:
                s += 1
        scores[r] = s
        
    # Sort residues by score descending.
    # High score means many i have a small factor <= 13.
    # These are the best candidates for having a long p-smooth run, 
    # because the "hard" part of the factorization is reduced.
    
    sorted_residues = sorted(range(M), key=lambda x: scores[x], reverse=True)
    
    # We will now scan primes p of the form p = r + k*M.
    # We prioritize residues with high scores.
    
    # We also need p to be a prime.
    # And p > sqrt(p^2+200) is always true for p>1.
    # And p must be <= 10^9.
    
    # Let's iterate through the top residues.
    # For each residue r, we check p = r + k*M for k=1, 2, ...
    # We stop if p > 10^9.
    
    # To avoid checking too many, we can limit the number of k's per residue
    # or interleave residues.
    
    # Let's take the top 1000 residues.
    top_residues = sorted_residues[:1000]
    
    # We'll use a max-heap or just round-robin to balance the search.
    # Let's just iterate through top_residues and for each, scan a few k's.
    
    # We need to be careful with time.
    # Checking is_p_smooth is the expensive part.
    # But with the pre-filter, many i will fail fast or pass fast?
    # Actually, if i is covered, p^2+i is divisible by a small q.
    # We can divide p^2+i by all small q's in M_primes first.
    # Then check if the remainder is 1 or p-smooth.
    
    def is_p_smooth_fast(n, p, pre_div_primes):
        temp = n
        for q in pre_div_primes:
            while temp % q == 0:
                temp //= q
        if temp == 1: return True
        # Now temp has no factors in pre_div_primes.
        # Check if temp is prime and <= p
        if is_prime_mr(temp):
            return temp <= p
        # Or factor it
        factors = factor(temp)
        return all(f <= p for f in factors)

    # We'll use the primes up to 1000 for the fast check, not just M_primes.
    # But the CRT only guarantees divisibility by M_primes.
    # So we should only use M_primes for the "guaranteed" division, 
    # but we can still try dividing by other small primes in the fast check.
    
    # Let's refine the fast check to use all small_primes up to 100.
    # This is safe because if p^2+i has a factor q <= 100, we remove it.
    
    def is_p_smooth_very_fast(n, p):
        temp = n
        for q in small_primes:
            if q > p: break
            if q * q > temp: break
            while temp % q == 0:
                temp //= q
        if temp == 1: return True
        if is_prime_mr(temp): return temp <= p
        factors = factor(temp)
        return all(f <= p for f in factors)

    # Now, the search loop.
    # We will iterate over top_residues.
    # For each residue, we will check a limited number of k's.
    # How many? 
    # Total time budget is ~110s.
    # Let's say we check 5000 primes total in this phase.
    # 5000 / 1000 residues = 5 k's per residue.
    
    k_per_residue = 10
    
    for r in top_residues:
        if time.time() > deadline:
            break
            
        # Start from the smallest p > 10^4 (to avoid small primes already checked)
        # p = r + k*M
        # We want p to be a prime.
        
        # Find the first k such that p > 10000
        start_k = max(1, (10000 - r) // M + 1)
        
        for k in range(start_k, start_k + k_per_residue):
            if time.time() > deadline:
                break
                
            p = r + k * M
            if p > 10**9:
                break
            if p < 2:
                continue
            if not is_prime_mr(p):
                continue
                
            # Check the run
            base = p * p
            run_k = 0
            for offset in range(1, 201):
                if time.time() > deadline:
                    break
                n = base + offset
                if is_p_smooth_very_fast(n, p):
                    run_k += 1
                else:
                    break
                    
            if run_k > best_k:
                best_k = run_k
                best_p = p
                
            # Optional: If we found a very good one, we could dig deeper around this p?
            # No, let's stick to the systematic search.

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
