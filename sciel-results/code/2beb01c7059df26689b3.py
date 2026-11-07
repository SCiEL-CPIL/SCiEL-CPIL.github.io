import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    N = 10_000_000
    
    # Initial fallback pair
    # sigma(31) = 32, sigma(21) = 32
    # |31/21 - 3/2| = |62/42 - 63/42| = 1/42
    best_a = 31
    best_b = 21
    best_err = Fraction(1, 42)
    
    def update(a, b):
        nonlocal best_a, best_b, best_err
        if a == b or a > N or b > N:
            return
        # a/b - 3/2 = (2a - 3b) / 2b
        err = abs(Fraction(2*a - 3*b, 2*b))
        if err < best_err:
            best_err = err
            best_a = a
            best_b = b
    
    # Meet-in-the-middle strategy:
    # Generate candidate pairs (a, b) where a/b ≈ 3/2, i.e., 2a ≈ 3b.
    # We'll enumerate b and compute target_a = round(3b/2), then check nearby values.
    # For each candidate a, verify sigma(a) == sigma(b) using a fast sigma computation.
    
    # Precompute smallest prime factor for fast factorization
    # We need this to compute sigma(n) quickly for arbitrary n up to N.
    
    # Actually, let's use a different approach: precompute sigma for all numbers up to N
    # but do it efficiently, then search for pairs.
    
    # Given the 90s budget, let's try to compute sigma up to N=10^7 efficiently.
    # Use a linear sieve approach.
    
    try:
        # Linear sieve for sigma
        # This gives us sigma(n) for all n in O(N) time.
        
        # We'll use arrays: spf (smallest prime factor), sigma, and prime powers.
        # Actually, a simpler approach: use the multiplicative property of sigma.
        
        # Let's use a modified sieve that computes sigma directly.
        # sigma is multiplicative, so if n = p1^e1 * p2^e2 * ..., then
        # sigma(n) = (1+p1+...+p1^e1) * (1+p2+...+p2^e2) * ...
        
        # Approach: For each prime p, multiply sigma by (1 + p + p^2 + ... + p^k)
        # for all numbers divisible by p^k but not p^(k+1).
        
        # This is O(N log log N) which should be fast enough for N=10^7.
        
        sigma = [1] * (N + 1)
        
        # Sieve of Eratosthenes to find primes
        is_prime = [True] * (N + 1)
        is_prime[0] = is_prime[1] = False
        primes = []
        
        for i in range(2, N + 1):
            if time.time() - start_time > deadline - 10.0:
                break
            if is_prime[i]:
                primes.append(i)
                # Sieve multiples
                for j in range(i*i, N + 1, i):
                    is_prime[j] = False
        
        # Now compute sigma using the prime factorization approach
        # For each prime p, for each power p^k, update sigma for numbers divisible by p^k
        # but not p^(k+1).
        
        for p in primes:
            if time.time() - start_time > deadline - 5.0:
                break
            
            # Compute p^k for k = 1, 2, 3, ...
            pk = p
            while pk <= N:
                # For numbers n where the highest power of p dividing n is pk,
                # we need to multiply sigma[n] by (1 + p + p^2 + ... + p^k) / (1 + p + ... + p^(k-1))
                # Actually, it's easier to think of it differently.
                
                # Let's use a different approach: for each n, compute sigma from its prime factorization.
                # But that's O(N log N) total.
                
                # Alternative: use the fact that sigma is multiplicative.
                # We can compute sigma by iterating over primes and updating.
                
                # For prime p, the contribution to sigma for numbers with p^k || n is:
                # (1 + p + p^2 + ... + p^k)
                # But we need to be careful not to double-count.
                
                # Let's use a cleaner approach: initialize sigma[n] = 1, then for each prime p,
                # for each number n divisible by p, multiply sigma[n] by the appropriate factor.
                
                break  # This approach is getting complicated; let's use a different method
        
        # Let's use a simpler O(N log N) sieve for sigma.
        # sigma[n] = sum of divisors of n.
        # For each i from 1 to N, add i to sigma[j] for all j that are multiples of i.
        
        # Reset sigma
        sigma = [0] * (N + 1)
        
        # This is O(N log N). For N=10^7, this is about 10^7 * 16 = 1.6 * 10^8 operations.
        # In Python, this might be too slow (could take minutes).
        
        # Let's use a smaller N first, say N_search = 1_000_000, and see if we can find a good pair.
        
        N_search = 1_000_000
        sigma = [0] * (N_search + 1)
        
        for i in range(1, N_search + 1):
            if time.time() - start_time > deadline - 10.0:
                break
            for j in range(i, N_search + 1, i):
                sigma[j] += i
        
        if time.time() - start_time > deadline - 5.0:
            return [best_a, best_b]
        
        # Now group by sigma value
        from collections import defaultdict
        groups = defaultdict(list)
        for n in range(1, N_search + 1):
            groups[sigma[n]].append(n)
        
        # For each group, find pairs (a, b) with a/b ≈ 3/2
        import bisect
        
        for s, nums in groups.items():
            if time.time() - start_time > deadline - 3.0:
                break
            if len(nums) < 2:
                continue
            
            nums.sort()
            
            for b in nums:
                target = 1.5 * b
                idx = bisect.bisect_left(nums, target)
                
                # Check nearby indices
                for delta in range(0, 3):
                    if idx + delta < len(nums):
                        a = nums[idx + delta]
                        update(a, b)
                    if idx - delta - 1 >= 0:
                        a = nums[idx - delta - 1]
                        update(a, b)
        
    except Exception as e:
        pass
    
    # If we have time, try a targeted search for larger numbers
    # Focus on numbers where 2a - 3b = 0 or small
    
    if time.time() - start_time < deadline - 5.0:
        try:
            # Search for pairs where 2a - 3b = d for small d
            # a = (3b + d) / 2, so 3b + d must be even.
            # For each b, try d in {-100, ..., 100} and check if a is integer and <= N.
            
            # We need sigma values for these a and b.
            # Use a fast sigma function based on prime factorization.
            
            def sigma_fast(n):
                """Compute sigma(n) using trial division with precomputed primes."""
                if n <= 0:
                    return 0
                result = 1
                temp = n
                # Check divisibility by 2
                if temp % 2 == 0:
                    factor_sum = 1
                    power = 1
                    while temp % 2 == 0:
                        power *= 2
                        factor_sum += power
                        temp //= 2
                    result *= factor_sum
                # Check odd primes
                p = 3
                while p * p <= temp:
                    if temp % p == 0:
                        factor_sum = 1
                        power = 1
                        while temp % p == 0:
                            power *= p
                            factor_sum += power
                            temp //= p
                        result *= factor_sum
                    p += 2
                if temp > 1:
                    result *= (1 + temp)
                return result
            
            # Precompute primes up to sqrt(N) for fast factorization
            limit = int(math.isqrt(N)) + 1
            is_prime = [True] * (limit + 1)
            is_prime[0] = is_prime[1] = False
            for i in range(2, int(math.isqrt(limit)) + 1):
                if is_prime[i]:
                    for j in range(i*i, limit + 1, i):
                        is_prime[j] = False
            primes_small = [i for i in range(2, limit + 1) if is_prime[i]]
            
            def sigma_fast2(n):
                """Compute sigma(n) using precomputed small primes."""
                if n <= 0:
                    return 0
                result = 1
                temp = n
                for p in primes_small:
                    if p * p > temp:
                        break
                    if temp % p == 0:
                        factor_sum = 1
                        power = 1
                        while temp % p == 0:
                            power *= p
                            factor_sum += power
                            temp //= p
                        result *= factor_sum
                if temp > 1:
                    result *= (1 + temp)
                return result
            
            # Search for pairs with small |2a - 3b|
            # We'll search b in a range and for each b, try small d values.
            
            # To make this efficient, let's search b in chunks and use the sigma_fast2 function.
            
            # Let's search b from 1 to N, but only check if 2a - 3b is small.
            # For each b, a ≈ 1.5b, so a is in a small range around 1.5b.
            
            # We'll search b up to N, but skip if time is running out.
            
            max_b = N
            d_max = 20  # Check |2a - 3b| <= 20
            
            for b in range(1, max_b + 1):
                if time.time() - start_time > deadline - 2.0:
                    break
                
                # For each d in range, compute a = (3b + d) / 2
                # a must be integer, so 3b + d must be even.
                # 3b + d is even iff b + d is even (since 3b has same parity as b).
                
                for d in range(-d_max, d_max + 1):
                    num = 3 * b + d
                    if num <= 0:
                        continue
                    if num % 2 != 0:
                        continue
                    
                    a = num // 2
                    if a <= 0 or a > N or a == b:
                        continue
                    
                    # Check if sigma(a) == sigma(b)
                    sa = sigma_fast2(a)
                    sb = sigma_fast2(b)
                    
                    if sa == sb:
                        update(a, b)
                        
        except Exception as e:
            pass
    
    return [best_a, best_b]
