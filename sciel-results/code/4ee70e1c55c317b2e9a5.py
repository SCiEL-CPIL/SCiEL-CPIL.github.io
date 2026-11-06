import math
def construct():
    N = 123457
    
    # Sieve of Eratosthenes to get primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    is_smooth = [False] * (N + 1)
    is_smooth[1] = True
    smooth_nums = [1]
    
    for k in primes:
        # Generate new smooth numbers
        # We iterate over the current length of smooth_nums
        # to avoid processing numbers added in this same iteration multiple times
        # Actually, we just need to multiply existing smooth numbers by k, k^2, etc.
        # The standard way is to iterate over a snapshot or just be careful.
        # Since we are adding to smooth_nums, we should iterate over the size before adding.
        size = len(smooth_nums)
        for i in range(size):
            num = smooth_nums[i]
            val = num * k
            while val <= N:
                if not is_smooth[val]:
                    is_smooth[val] = True
                    smooth_nums.append(val)
                val *= k
        
        # Check for solution
        # We can optimize this check.
        # We only need to check new numbers?
        # If a pair was formed by two old numbers, it would have been found before.
        # So we only need to check if any NEW number `a` has `N-a` in `is_smooth`.
        # OR if any OLD number `a` has `N-a` being a NEW number?
        # If `N-a` is new, it was just added.
        # So checking only new numbers `a` against `is_smooth[N-a]` covers all new pairs?
        # Case 1: a is new, b is old. N-a is old. is_smooth[N-a] is True. Found.
        # Case 2: a is old, b is new. N-a is new. is_smooth[N-a] is True. Found.
        # Case 3: a is new, b is new. Both added in this step. is_smooth is True for both. Found.
        # So yes, checking only the newly added numbers is sufficient.
        
        # However, we appended to smooth_nums.
        # The new numbers are smooth_nums[size:]
        for a in smooth_nums[size:]:
            b = N - a
            if 1 <= b <= N and is_smooth[b]:
                return [a, b]
                
    # Fallback
    return [1, N - 1]
