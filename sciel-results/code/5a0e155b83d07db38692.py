import math
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # 10s margin from 150s limit
    
    # Deterministic Miller-Rabin bases for n < 2^64
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        if n % 3 == 0:
            return n == 3
        
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
            
        for a in MR_BASES:
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
            
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factorize(n):
        factors = []
        # Trial division by very small primes first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                while n % p == 0:
                    factors.append(p)
                    n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        
        # Pollard Rho for remaining
        stack = [n]
        while stack:
            x = stack.pop()
            if is_prime(x):
                factors.append(x)
            else:
                d = pollard_rho(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    def check_run_slow(p):
        """
        Fallback: Direct factorization check.
        """
        k = 0
        for offset in range(1, 201):
            n = p * p + offset
            if is_prime(n):
                break
            factors = factorize(n)
            if max(factors) > p:
                break
            k += 1
        return k

    def check_run_fast(p, bad_primes):
        """
        Fast check: Verify p^2 + i is not divisible by any q in bad_primes.
        If it passes, we still need to be sure there are no OTHER large factors.
        However, if p is large enough, the probability of a factor q > p 
        that is NOT in our precomputed list (which covers q up to ~10^6) 
        is low but non-zero. 
        Actually, if q > 10^6, and q divides p^2+i, then q > p is possible only if p < q.
        Wait, the condition is: all prime factors of p^2+i must be <= p.
        So if q > p, it's bad.
        Our bad_primes list contains q such that q can divide p^2+i for SOME p.
        Specifically, for a fixed p, we check if there exists i in [1,200] such that p^2 + i has a factor q > p.
        
        The precomputation logic:
        A prime q is "bad" for a specific p if q > p and q divides p^2 + i for some i in [1, 200].
        This is equivalent to p^2 ≡ -i (mod q).
        
        We can precompute for each q (up to some limit Q_max), the set of residues r such that r^2 ≡ -i (mod q) for some i in [1, 200].
        If p mod q is in that set, then p^2 + i is divisible by q for some i.
        If q > p, then p^2 + i has a factor q > p, which is forbidden.
        
        So, for a candidate p:
        1. Check for all q in [p+1, Q_max]: if p mod q is a "bad residue" for q, then p is invalid (for that specific run length? No, it means p^2+i is divisible by q).
           Actually, if p^2+i is divisible by q, and q > p, then p^2+i is NOT p-smooth.
           So if ANY such q exists, the run breaks at that i.
           
        Wait, the task requires p^2+1 ... p^2+k to ALL be p-smooth.
        So if for ANY i in [1, 200], there is a factor q > p, the run length is at most i-1.
        To have k=200, we need NO factor q > p for ANY i in [1, 200].
        
        So the check for k=200 is:
        For all i in [1, 200], p^2 + i has no prime factor q > p.
        
        We can check this by:
        1. Precompute for each q up to Q_max, the set of residues R_q = { r | r^2 ≡ -i (mod q) for some i in [1, 200] }.
        2. For candidate p, check if for any q in [p+1, Q_max], p mod q in R_q.
           If yes, then p^2 + i is divisible by q (where q > p), so p^2 + i is not p-smooth.
           Thus k < 200.
        
        If we pass this check for all q up to Q_max, it is highly likely that k=200, provided p is large enough such that any factor q > Q_max is also > p (which is true if Q_max > p? No, if q > Q_max and q > p, we missed it).
        
        However, if q > p, then q is a "large" factor.
        If we sieve up to Q_max = 10^6, and p is, say, 10^5, then any factor q > 10^6 is definitely > p.
        So if we find no q in [p+1, 10^6] that divides p^2+i, we still have to check if there is a factor q > 10^6.
        But if q > 10^6, then q > p (since p < 10^6 in this phase).
        So we would have missed it.
        
        Therefore, the fast check only works if we are confident that no factor q > Q_max exists.
        This is hard to guarantee without factorization.
        
        Alternative Strategy:
        Use the fast check to FILTER OUT candidates that definitely fail (have a small large factor).
        For candidates that pass the filter, run the slow factorization check to confirm k=200.
        
        This should be much faster because most candidates will have a small factor q > p (e.g., q in [p+1, 10^6]).
        """
        # Check if p is "bad" due to small large factors
        # We precomputed bad_residues[q] = set of r such that r^2 + i ≡ 0 mod q for some i in 1..200
        # We need to check if p mod q is in bad_residues[q] for any q in [p+1, Q_max]
        
        # To make this fast, we can iterate q from p+1 to Q_max.
        # But Q_max can be 10^6. If p is small, this loop is long.
        # However, we can optimize:
        # If p is small, we can just use the slow check?
        # No, the slow check is also slow for many candidates.
        
        # Let's refine the precomputation.
        # We will precompute bad_residues for q up to Q_MAX = 200000.
        # For a candidate p:
        # 1. If p >= Q_MAX, we cannot use this sieve for q > p if q > Q_MAX.
        #    But if p is large (e.g., > 10^6), the chance of a factor q in [p+1, 10^6] is zero (range empty).
        #    So for large p, this sieve is useless.
        
        # So this strategy is best for SMALL p (e.g., p < 10^5).
        # For small p, the range [p+1, Q_MAX] is large, but we can check it.
        
        # Actually, let's just use the slow check but optimize the factorization.
        # The current slow check factorizes p^2+i.
        # p^2 is up to 10^10 (if p=10^5).
        # Factorization of 10^10 is fast with Pollard Rho.
        
        # The bottleneck in the current program is likely the number of primes tested.
        # Sequential scan up to 10^6 takes too long?
        # 10^6 primes * 200 checks * factorization cost.
        
        # Let's try a different approach:
        # 1. Sieve primes up to 10^6.
        # 2. For each prime p, check if p^2+1 ... p^2+200 are p-smooth.
        #    Optimization: Precompute smallest prime factor (SPF) up to 10^7?
        #    p^2 + 200 can be up to 10^12. SPF up to 10^7 is not enough.
        
        # Let's stick to the "Filter then Verify" approach.
        # Precompute bad residues for q up to 10^5.
        # For p < 10^5:
        #   Check if p mod q in BadResidues[q] for q in [p+1, 10^5].
        #   If yes, skip.
        #   If no, run slow check.
        # For p >= 10^5:
        #   Run slow check directly (but fewer of these? No, we have many).
        
        # Actually, for p >= 10^5, p^2 is 10^10.
        # The factors of p^2+i are likely small or large.
        # If a factor is > p, it's "large".
        
        # Let's implement the Filter.
        
        Q_MAX = 100000
        # Precompute bad residues
        # bad_res[q] is a boolean array of size q? No, q is up to 10^5, so array of size 10^5 for each q is too big.
        # Use a set.
        bad_res = [None] * (Q_MAX + 1)
        
        # Precomputation time:
        # For each q up to Q_MAX:
        #   For each i in 1..200:
        #     Find r such that r^2 ≡ -i mod q.
        #     This is solving x^2 ≡ a mod q.
        #     We can just iterate r from 0 to q-1? No, too slow.
        #     We can use Tonelli-Shanks or just check all r?
        #     Since q is small (10^5), iterating r is 10^5 * 200 * 10^5 = 2*10^12. Too slow.
        
        # Better: For each q, iterate i in 1..200.
        #   We want to know if there is r such that (r^2 + i) % q == 0.
        #   This is equivalent to: Is -i a quadratic residue mod q?
        #   If yes, there are 2 solutions r.
        #   We can add these r to bad_res[q].
        
        # How to check if -i is a QR mod q?
        # Euler's criterion: (-i)^(q-1)/2 ≡ 1 mod q.
        # Or just compute pow(-i, (q-1)//2, q).
        
        # But we also need the actual residues r.
        # If we just want to know IF p is bad, we can check:
        # For a given p, is there q in [p+1, Q_MAX] such that (p^2 + i) % q == 0 for some i?
        # This is equivalent to: Is p^2 ≡ -i mod q?
        
        # Let's precompute for each q the set of "bad squares" S_q = { r^2 mod q | r=0..q-1 }? No.
        # We want to know if p^2 mod q is in T_q, where T_q = { (-i) mod q | i in 1..200 }.
        
        # So:
        # 1. Precompute T_q for all q up to Q_MAX.
        #    T_q = set( (-i) % q for i in range(1, 201) )
        # 2. For candidate p:
        #    For q in range(p+1, min(p+200, Q_MAX) + 1)? No, q can be any prime > p.
        #    Actually, if q divides p^2+i, then q can be any prime.
        #    But we only care about q > p.
        #    So we check q in [p+1, Q_MAX].
        #    Check if (p*p) % q in T_q.
        
        # Complexity of check:
        # Number of q to check: Q_MAX - p.
        # If p is small (e.g., 100), we check 10^5 values.
        # 10^5 checks per candidate.
        # If we have 10^4 candidates, that's 10^9 operations. Too slow in Python.
        
        # We need to invert the loop.
        # Instead of checking each p against many q,
        # we can sieve the primes p.
        
        # Create an array valid_p of size P_MAX (e.g., 10^6), initialized to True.
        # For each q in [2, Q_MAX]:
        #   If q is prime:
        #     Compute T_q = set( (-i) % q for i in 1..200 )
        #     For each r in 0..q-1:
        #       if (r*r) % q in T_q:
        #         Then p ≡ r (mod q) is bad.
        #         So for all p in [q, P_MAX] such that p % q == r:
        #            valid_p[p] = False?
        #            Wait, this only invalidates p if q > p.
        #            So we only mark p < q as invalid?
        #            No, the condition is q > p.
        #            So if p < q, and p ≡ r (mod q), then p^2+i is divisible by q.
        #            Since q > p, this is a bad factor.
        #            So p is invalid for k=200.
            
        # So:
        # valid_p = bytearray(b'\x01') * (P_MAX + 1)
        # valid_p[0] = valid_p[1] = 0
        # for q in range(2, Q_MAX + 1):
        #   if is_prime(q):
        #     T_q = set()
        #     for i in range(1, 201):
        #       T_q.add((-i) % q)
        #     # Find r such that r^2 mod q in T_q
        #     # We can iterate r from 0 to q-1
        #     for r in range(q):
        #       if (r * r) % q in T_q:
        #         # Mark all p < q with p % q == r as invalid
        #         # p can be r, r+q, r+2q...
        #         # But we only care about p < q.
        #         # So only p = r is in range [0, q-1].
        #         # So valid_p[r] = 0?
        #         # Wait, p is a prime.
        #         # So if r is prime, valid_p[r] = 0.
        #         # But r < q.
        #         # So we mark r as invalid.
        
        # This logic:
        # If p < q and p^2 + i is divisible by q, then p is invalid.
        # Since p < q, p % q = p.
        # So we just need to check if p^2 mod q in T_q.
        # Which is equivalent to: for each q, find all p < q such that p^2 mod q in T_q.
        
        # This is still O(Q_MAX^2) if we do it naively.
        # Q_MAX = 10^5 -> 10^10. Too slow.
        
        # We need a faster way.
        # Notice that T_q is small (size <= 200).
        # For a fixed q, the number of r such that r^2 mod q in T_q is at most 2 * 200 = 400.
        # So we can find these r efficiently?
        # We can iterate i in 1..200, solve x^2 ≡ -i mod q.
        # If we can solve this fast, we get the bad residues r.
        # Then for each bad r, we mark p=r (if r < q) as invalid.
        
        # Solving x^2 ≡ a mod q:
        # If q is prime, we can use Tonelli-Shanks.
        # But implementing TS for each q and each i might be overhead.
        # Alternatively, since q is small, we can precompute a lookup table for squares mod q?
        # No, q varies.
        
        # Let's try a different angle.
        # The current program scores 2.33.
        # It finds p with k=7? (2.33 / 1.0 = 2.33, maybe k=7 if seed k=3?)
        # Or maybe k is the score.
        
        # Let's just optimize the factorization check.
        # The slow check is:
        # for i in 1..200:
        #   factorize(p^2+i)
        #   if max_factor > p: break
        
        # We can speed up factorize by checking small primes first.
        # And we can stop early if we find a large factor.
        
        # Also, we can parallelize? No, GIL.
        
        # Let's try to improve the search space.
        # Instead of all primes, maybe primes with specific properties?
        
        # Let's just make the current program faster.
        # 1. Use a faster is_prime.
        # 2. Use a faster factorize.
        # 3. Limit the search to a reasonable number of primes.
        
        # The deadline is 150s.
        # Let's run the slow check for as many primes as possible.
        
        # Generate primes up to 10^6.
        limit = 1000000
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        
        primes = [i for i, is_p in enumerate(sieve) if is_p]
        
        best_k = 0
        best_p = 2
        
        # Check primes in increasing order
        for p in primes:
            if time.time() - start_time > DEADLINE - 10:
                break
            
            k = check_run_slow(p)
            if k > best_k:
                best_k = k
                best_p = p
        
        # If we have time, try larger primes randomly
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 20:
            while time.time() - start_time < DEADLINE - 5:
                # Pick random prime in [10^6, 10^8]
                # Approximate: pick random odd, check prime
                c = random.randint(10**6, 10**8)
                if c % 2 == 0: c += 1
                if is_prime(c):
                    k = check_run_slow(c)
                    if k > best_k:
                        best_k = k
                        best_p = c
        
        if best_k < 1:
            best_p = 7
            best_k = 1
            
        return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
