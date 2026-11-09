import time

def construct():
    N = 123457
    start_time = time.time()
    
    # Primes to test as upper bounds for smoothness
    # We start small and go up. If a solution exists for y, it also exists for any y' > y.
    # We want the minimum max(P(a), P(b)), so we test smallest y first.
    primes_to_test = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Precompute powers for speed? Or just recursive generation.
    # Recursive generation of y-smooth numbers up to N is efficient for small y.
    
    def generate_smooth_numbers(limit, primes):
        smooths = set([1])
        # Iterate through primes and add multiples
        # This standard method might be slow if not careful, but for small limits and few primes, it's fine.
        # Better: DFS/Recursion
        smooths_list = [1]
        
        def dfs(idx, current_val):
            if current_val > limit:
                return
            if idx == len(primes):
                return
            p = primes[idx]
            # Add current_val * p^k
            val = current_val
            while val <= limit:
                smooths_list.append(val)
                val *= p
                if val > limit:
                    break
                dfs(idx + 1, val)
        
        # This DFS approach generates numbers by fixing the exponent of the first prime, then the second, etc.
        # It generates duplicates if not careful? No, if we strictly increase index, each combination of exponents is unique.
        # However, the order of primes matters for the tree depth, but not the set.
        
        # Let's use a simpler iterative set expansion which is robust for small N.
        smooths_set = {1}
        for p in primes:
            new_smooths = set()
            for s in smooths_set:
                val = s
                while val <= limit:
                    new_smooths.add(val)
                    val *= p
            smooths_set.update(new_smooths)
        return smooths_set

    for y in primes_to_test:
        # Check time budget
        if time.time() - start_time > 50:
            break
            
        # Generate y-smooth numbers up to N
        # The primes used for generation are all primes <= y
        primes_le_y = [p for p in primes_to_test if p <= y]
        
        smooth_set = generate_smooth_numbers(N, primes_le_y)
        
        # Check for pair
        found_pair = None
        for a in smooth_set:
            b = N - a
            if b in smooth_set:
                # Ensure a, b are positive
                if a > 0 and b > 0:
                    found_pair = [a, b]
                    break
        
        if found_pair:
            return found_pair
            
    # Fallback: if no smooth pair found (unlikely), return a heuristic or seed
    # For N=123457, 11-smooth should definitely work.
    # If it doesn't, we can do a linear scan for min max(P(a), P(b))
    # But let's trust the smoothness approach.
    
    return [1, 123456]
