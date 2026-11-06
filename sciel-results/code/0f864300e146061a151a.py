import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Generate small primes up to 2 * 10^6 for factorization and base primes
    # We need primes up to sqrt(10^18) = 10^9 for general factoring, 
    # but for the sieve method, we only need primes up to sqrt(max_candidate).
    # max_candidate ~ p^2 + 200. If p is up to 10^9, sqrt is 10^4.5 ~ 31622.
    # However, we also need primes to test p itself.
    # Let's generate primes up to 10^6 for now, which covers p up to 10^6 easily.
    # For larger p, we can use Miller-Rabin for primality of p.
    
    limit_small = 10**6
    sieve = [True] * (limit_small + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(math.isqrt(limit_small)) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]
    
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

    def is_p_smooth_range(base, k, p):
        """
        Checks if base+1, ..., base+k are all p-smooth.
        Uses a segmented sieve to find the smallest prime factor for each number in the range.
        """
        # We need to check numbers N = base + 1, ..., base + k
        # We need to know if all prime factors of N are <= p.
        
        # If p is small, we can just trial divide by primes <= p.
        # If p is large, the number of primes <= p is large, but N is ~ p^2.
        # The largest prime factor of N can be at most N.
        # If N has a prime factor > p, it must be a prime factor of N that is > p.
        
        # Efficient check:
        # For each N in the range, divide by all primes q <= p.
        # If the remaining factor is 1, it's p-smooth.
        # If the remaining factor is > 1, it must be a prime > p (or composite with factors > p).
        # Wait, if we divide by all primes <= p, any remaining factor must have all its prime factors > p.
        # So we just need to check if the remaining factor is 1.
        
        # Optimization:
        # We can use a sieve to mark the smallest prime factor for the numbers in [base+1, base+k].
        # But we need primes up to sqrt(base+k) to sieve.
        # sqrt(p^2+k) ~ p.
        # So we need primes up to p.
        
        # If p is small (e.g., < 10^5), we have all primes up to p in small_primes.
        # If p is large, we need to generate primes up to p? That's expensive.
        
        # Alternative approach for large p:
        # The condition "all prime factors <= p" is equivalent to:
        # N has no prime factor > p.
        # Let's try to find a prime factor > p.
        # If we can't find one quickly, assume it's p-smooth? No, must be exact.
        
        # Let's stick to the segmented sieve for the specific range [base+1, base+k].
        # We need to divide each number by primes q.
        # Which primes q?
        # If q > p, we don't care about it unless it divides N. But if q > p divides N, then N is NOT p-smooth.
        # So we want to detect if there exists a prime q > p such that q | N.
        
        # This is hard to do directly with a sieve because we don't sieve with primes > p.
        
        # Better strategy:
        # 1. Divide N by all primes q <= p.
        # 2. If the remainder is 1, it's p-smooth.
        # 3. If the remainder is > 1, it is NOT p-smooth.
        
        # So the cost is proportional to the number of primes <= p.
        # For p = 10^6, there are ~78,000 primes.
        # For k=200, that's 200 * 78,000 operations = 15.6 million.
        # This is feasible in Python if optimized? Maybe not fast enough for many p.
        
        # Can we speed this up?
        # We can precompute the product of primes? No.
        
        # Let's use the segmented sieve idea differently.
        # We want to know if N has a prime factor > p.
        # This is equivalent to: gcd(N, M) = 1 where M is the product of all primes > p? No.
        
        # Let's just iterate through primes q <= p and divide.
        # To make it faster, we can stop early if the remainder becomes 1.
        
        # Even better:
        # We can check if N is p-smooth by checking if N divides some p-smooth number?
        
        # Let's implement the direct trial division by primes <= p, but optimized.
        # We will precompute the list of primes <= p.
        # If p > limit_small, we need to generate primes up to p.
        # Generating primes up to 10^7 is fast (~1s). Up to 10^8 is slow (~10s).
        # So for p > 10^7, this method might be too slow per candidate.
        
        # Strategy:
        # Phase 1: p < 10^6. Use precomputed small_primes.
        # Phase 2: p >= 10^6. 
        #   For these, the probability of a long run is very low.
        #   We can use a heuristic: if p is large, the numbers p^2+i are likely to have large prime factors.
        #   We can just test a few random large p's?
        
        # Actually, let's look at the "inspiration" finding:
        # "Extended search range to 5e+06 primes with optimized early-termination factorization Measured: 1 -> 3."
        # This suggests that searching up to 5e6 primes yielded a k=3 (or better?).
        # The current score is 2.3333, which corresponds to k=7 (since 7/3 = 2.3333, assuming seed k=3? Or seed value 3?).
        # The prompt says "score is value/seed". If seed is 3, and value is 7, score is 2.333.
        # So we currently have k=7.
        # We need to find k > 7.
        
        # Let's implement a fast check for p-smoothness.
        
        # If p is large, we can't iterate through all primes <= p for every N.
        # However, if N is p-smooth, its prime factors are small (relative to N).
        # But "small" here means <= p. Since N ~ p^2, the factors can be as large as p.
        
        # Let's use the property:
        # N is p-smooth iff N has no prime factor > p.
        
        # We can use a segmented sieve to find the smallest prime factor (SPF) of each N in [base+1, base+k].
        # To do this, we need primes up to sqrt(base+k) ~ p.
        # If we have primes up to p, we can sieve the range.
        
        # Let's generate primes up to 10^7 once.
        limit_sieve_primes = 10**7
        if limit_sieve_primes > limit_small:
            # Extend sieve
            sieve_large = [True] * (limit_sieve_primes + 1)
            # Copy small sieve
            sieve_large[:limit_small+1] = sieve
            for i in range(int(math.isqrt(limit_small))+1, int(math.isqrt(limit_sieve_primes)) + 1):
                if sieve_large[i]:
                    sieve_large[i*i::i] = [False] * len(sieve_large[i*i::i])
            primes_large = [i for i, is_p in enumerate(sieve_large) if is_p]
        else:
            primes_large = small_primes

        # Find index of p in primes_large
        # Since p is prime, we can binary search.
        
        def get_primes_up_to(p):
            # Returns primes <= p
            # Use binary search on primes_large
            if p > primes_large[-1]:
                # Need to generate more?
                # For now, assume p is within range or use Miller-Rabin for p itself and trial division for N.
                pass
            lo, hi = 0, len(primes_large)
            while lo < hi:
                mid = (lo + hi) // 2
                if primes_large[mid] <= p:
                    lo = mid + 1
                else:
                    hi = mid
            return primes_large[:lo]

        def check_run(base, k, p):
            # Check if base+1 ... base+k are p-smooth
            # Get primes <= p
            # If p is very large, this list is huge.
            # Optimization: If p > sqrt(base+k), then any prime factor > p must be the number itself (if prime) or a large factor.
            # Actually, if N is composite and has a factor > p, it must have a factor < N/p.
            
            # Let's just use the segmented sieve for the range [base+1, base+k] using primes <= p.
            # But if p is large, iterating through primes is slow.
            
            # Alternative:
            # For each N, divide by small primes (e.g., < 10^4).
            # If the remainder is 1, it's p-smooth (since remaining factors must be > 10^4, but if they are <= p, we need to check).
            # Wait, if we divide by all primes < 10^4, the remainder R might be composite.
            # If R is 1, done.
            # If R > 1, we need to check if all prime factors of R are <= p.
            # Since we divided by primes < 10^4, the smallest prime factor of R is >= 10^4.
            # If p < 10^4, then R must be 1.
            # If p >= 10^4, R might have factors in [10^4, p].
            
            # Let's use a hybrid:
            # 1. Trial divide by primes < 10^4.
            # 2. If remainder is 1, return True.
            # 3. If remainder > 1:
            #    - If remainder is prime, check if <= p.
            #    - If remainder is composite, factor it.
            
            # This is basically what the original program did, but the original program was slow because it factored from scratch.
            # The "Segmented Sieve" strategy in the prompt suggests:
            # "precompute the smallest prime factor (SPF) for the entire range [p^2+1, p^2+200] in O(k log log p) time"
            
            # To do this, we need primes up to sqrt(p^2+200) ~ p.
            # If we have primes up to p, we can sieve the range of size 200.
            # Sieving 200 numbers with primes up to p:
            # Number of primes up to p is ~ p / log p.
            # For each prime q, we mark multiples in the range.
            # Total operations ~ sum_{q<=p} (200 / q) ~ 200 * log log p.
            # This is very fast!
            
            # So the plan:
            # 1. Ensure we have primes up to max_p we will test.
            # 2. For each candidate p:
            #    a. Get list of primes <= p.
            #    b. Create an array rem[0..k-1] initialized to base+1 ... base+k.
            #    c. For each prime q in primes <= p:
            #         Find the first multiple of q in [base+1, base+k].
            #         For each multiple m in the range:
            #             while rem[idx] % q == 0:
            #                 rem[idx] //= q
            #    d. Check if all rem[idx] == 1.
            
            # This is O(k * pi(p) / p * something)?
            # Actually, for each q, we do ~ k/q divisions.
            # Sum is ~ k * log log p.
            # This is extremely efficient for k=200.
            
            # Let's implement this.
            
            # We need primes up to p.
            # If p > len(primes_large) * something, we might not have enough primes.
            # Let's generate primes up to 10^7. This covers p up to 10^7.
            # For p > 10^7, we can extend if needed, but likely we won't find better k there.
            
            if p > primes_large[-1]:
                # Fallback to slower method or skip
                # For now, skip p > 10^7 to avoid generating huge sieves
                return 0

            # Get primes <= p
            # Binary search for the index
            lo, hi = 0, len(primes_large)
            while lo < hi:
                mid = (lo + hi) // 2
                if primes_large[mid] <= p:
                    lo = mid + 1
                else:
                    hi = mid
            relevant_primes = primes_large[:lo]
            
            # Segmented sieve for the range
            # rem[i] corresponds to base + 1 + i
            rem = [base + 1 + i for i in range(k)]
            
            for q in relevant_primes:
                # Find start index
                # We want the smallest i such that (base + 1 + i) is divisible by q
                # base + 1 + i = m * q
                # i = m * q - base - 1
                # We need i >= 0 and i < k
                # m * q >= base + 1
                # m >= ceil((base + 1) / q)
                
                start_m = (base + 1 + q - 1) // q
                # Check if start_m * q is in range
                start_val = start_m * q
                if start_val > base + k:
                    continue
                
                # Iterate through multiples
                # idx = start_val - base - 1
                idx = start_val - base - 1
                while idx < k:
                    val = rem[idx]
                    while val % q == 0:
                        val //= q
                    rem[idx] = val
                    if rem[idx] == 1:
                        # Can't get smaller, but we still need to check others
                        pass
                    idx += q
            
            # Check if all are 1
            # Note: If a number was prime and > p, it would not be divided by any q <= p, so it would remain > 1.
            # If a number was composite with a factor > p, it would remain > 1.
            # So if all rem[i] == 1, they are p-smooth.
            
            for i in range(k):
                if rem[i] != 1:
                    return i # Failed at i (0-indexed, so length is i)
            return k

        best_k = 0
        best_p = 3

        # Phase 1: Exhaustive search for small p
        # We can go up to 10^7 with this method?
        # Sieving 200 numbers with ~660k primes (pi(10^7)) takes ~ 200 * log(log(10^7)) ~ 200 * 3 ~ 600 ops?
        # Wait, the loop `for q in relevant_primes` iterates through ALL primes <= p.
        # For p=10^7, that's 660,000 primes.
        # For each prime, we do a few divisions.
        # 660,000 * (small constant) ~ 1-2 million ops per candidate p.
        # If we test 10,000 candidates, that's 20 billion ops. Too slow.
        
        # Optimization:
        # We don't need to check ALL primes <= p.
        # We only need to divide by primes q such that q <= p AND q <= N.
        # But N ~ p^2, so q <= p is always <= N.
        
        # However, if q > sqrt(N), then q can divide N at most once.
        # But we still need to check if q divides N.
        
        # Can we skip primes?
        # If we divide by all primes q <= sqrt(N), the remainder is either 1 or a prime.
        # But we need to check if that prime is <= p.
        
        # Let's refine the sieve:
        # Only sieve with primes q <= sqrt(base+k) ~ p.
        # After sieving with q <= p, the remaining factor R_i is either 1 or a prime > p?
        # No. If N = q1 * q2 with q1, q2 <= p, we divide by both, R=1.
        # If N = q1 * q2 with q1 <= p < q2, we divide by q1, R = q2.
        # Since q2 > p, R > 1. Correct.
        # If N is prime > p, R = N > 1. Correct.
        
        # So the algorithm is correct. The issue is speed for large p.
        
        # To speed up, we can stop sieving if the minimum remaining value in `rem` is 1? No.
        
        # Let's limit the search to p < 10^5 first, then expand if time permits.
        # For p < 10^5, pi(p) ~ 9592.
        # 9592 * 200 ~ 2 million ops per candidate.
        # We can test ~1000 candidates in 1s?
        
        # Let's try testing all primes up to 10^6.
        # pi(10^6) ~ 78000.
        # 78000 * 200 ~ 15 million ops per candidate.
        # Too slow for many candidates.
        
        # We need a better filter.
        # Pre-filter: Check if p^2+1, p^2+2, etc. have small factors.
        
        # Let's just run the sieve method for p up to 10^5.
        # And use the old method (factoring) for a few random larger p's.
        
        # Actually, let's just use the sieve method for p up to 10^4.
        # pi(10^4) ~ 1229.
        # 1229 * 200 ~ 250k ops per candidate.
        # We can test 10,000 candidates in 1s? 2.5 billion ops. Maybe 2-3s in Python.
        
        # Let's test primes up to 10^5.
        
        for p in small_primes:
            if time.time() > deadline:
                break
            if p > 10**5:
                break
            base = p * p
            k_found = check_run(base, 200, p)
            if k_found > best_k:
                best_k = k_found
                best_p = p

        # Phase 2: Random search for larger p
        # Sample random primes between 10^5 and 10^7
        # Use a faster heuristic?
        # Maybe just check the first few numbers?
        
        # Let's try a few random large primes
        for _ in range(100):
            if time.time() > deadline:
                break
            # Random prime in [10^5, 10^7]
            # We don't have a list of primes in this range.
            # Generate a random number and check primality.
            # Or pick from a precomputed list if we had one.
            # Let's just pick random numbers and check.
            
            # Better: Pick a random index in small_primes? No, they are too small.
            
            # Let's generate a random large number and check if prime.
            # This is slow.
            
            # Alternative: Use the fact that if k is large, p must be specific.
            # We already searched up to 10^5.
            
            # Let's try to find p such that p^2+1 is smooth.
            
            # For now, return the best found so far.
            pass

        return {'p': best_p, 'k': best_k}

