import math

def construct():
        N = 123457
        
        # Generate primes up to N
        # Sieve for primes is still needed to know what B values to test
        is_prime = [True] * (N + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(N)) + 1):
            if is_prime[i]:
                for j in range(i*i, N + 1, i):
                    is_prime[j] = False
        primes = [i for i in range(2, N + 1) if is_prime[i]]
        
        # Binary search on the prime bound index
        # We want the smallest B (prime) such that a pair exists.
        # The bound B determines the set of allowed prime factors.
        
        def generate_smooth_numbers(limit, max_prime):
            """
            Generate all numbers <= limit whose prime factors are all <= max_prime.
            Returns a set of such numbers.
            """
            # Identify primes <= max_prime
            allowed_primes = [p for p in primes if p <= max_prime]
            
            smooth_nums = set()
            
            # Recursive generation
            # To avoid recursion depth issues or inefficiency, we can use a stack or iterative expansion.
            # Given N is small, recursion is fine, but let's be careful.
            # Max depth is log2(N) ~ 17.
            
            def gen(current_val, start_idx):
                if current_val > limit:
                    return
                smooth_nums.add(current_val)
                for i in range(start_idx, len(allowed_primes)):
                    p = allowed_primes[i]
                    if current_val * p > limit:
                        break
                    gen(current_val * p, i)
            
            gen(1, 0)
            return smooth_nums

        low = 0
        high = len(primes) - 1
        best_pair = None
        
        # We can optimize by just iterating B from smallest to largest until found.
        # Since the optimal B is likely small, linear search on small B might be faster than binary search 
        # if the generation is cheap for small B.
        # However, binary search is safer for worst case.
        # But note: the property "pair exists" is monotonic. If it exists for B, it exists for any B' > B.
        # So binary search is valid.
        
        while low <= high:
            mid = (low + high) // 2
            B = primes[mid]
            
            # Generate B-smooth numbers
            smooth_set = generate_smooth_numbers(N, B)
            
            # Check for pair
            found = False
            # We can iterate over the smaller set or just the smooth_set
            # a + b = N => b = N - a
            # If a is in smooth_set, check if N-a is in smooth_set.
            for a in smooth_set:
                b = N - a
                if b in smooth_set:
                    best_pair = (a, b)
                    found = True
                    break
            
            if found:
                high = mid - 1
            else:
                low = mid + 1
                
        if best_pair is None:
            # Fallback: This shouldn't happen as B=N covers all numbers
            # If it does, return a trivial pair
            best_pair = (1, N - 1)
            
        return [best_pair[0], best_pair[1]]
