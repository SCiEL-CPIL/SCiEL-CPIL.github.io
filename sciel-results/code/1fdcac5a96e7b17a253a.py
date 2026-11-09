import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    min_val = 211
    max_val = 10210
    target_kernel = 210
    
    # Precompute square-free kernels for numbers up to max_val
    # This is fast enough and robust.
    # kernel[n] is the square-free part of n.
    # We can compute this by removing all square factors.
    # For n up to 10210, we can just iterate.
    
    # Helper to get square-free part
    def get_square_free_part(n):
        # n is small enough that trial division by primes up to sqrt(n) is fine
        # But we can be smarter: precompute primes up to sqrt(max_val)
        pass # We'll use a precomputed list of primes

    # Primes up to sqrt(10210) ~ 101
    limit = int(math.isqrt(max_val))
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    def sq_free(n):
        # Remove square factors
        res = n
        for p in primes:
            if p * p > res:
                break
            while res % (p * p) == 0:
                res //= (p * p)
        # If res > 1 and res is not divisible by any p^2, it's square-free
        # Note: if n has a prime factor > limit, its exponent is 1, so it stays.
        # Since we only divide by p^2 for p <= limit, any remaining factor is square-free.
        return res

    # We want to find a, b in [211, 10210] such that 210 * a * b is a square.
    # Let K_a = sq_free(210 * a).
    # Then 210 * a = K_a * x^2.
    # We need 210 * a * b = K_a * x^2 * b to be a square.
    # This requires K_a * b to be a square.
    # Since K_a is square-free, b must be of the form K_a * y^2.
    # So b = K_a * y^2.
    # We need to find the smallest b > 210 such that b = K_a * y^2 and b != a.
    # And we want to minimize max(a, b).
    # Since we iterate a from 211 upwards, the first valid b we find for a given a
    # will likely give a small max(a, b).
    # Specifically, if we find b, max(a,b) is a candidate.
    # We should keep track of the global minimum max(a,b).
    
    best_max = float('inf')
    best_pair = None
    
    # Iterate a
    for a in range(min_val, max_val + 1):
        if time.time() > deadline - 5.0:
            break
            
        k_a = sq_free(210 * a)
        
        # We need b = k_a * y^2
        # b must be >= min_val and <= max_val
        # b != a
        
        # Find smallest y such that b = k_a * y^2 >= min_val
        # y^2 >= min_val / k_a
        # y >= sqrt(min_val / k_a)
        
        if k_a == 0: # shouldn't happen
            continue
            
        min_y_sq = min_val / k_a
        y = math.isqrt(min_y_sq)
        if y * y < min_y_sq:
            y += 1
            
        # Check y, y+1, ... until b > max_val
        found_b = None
        while True:
            b = k_a * y * y
            if b > max_val:
                break
            if b >= min_val and b != a:
                # Valid pair (a, b)
                current_max = max(a, b)
                if current_max < best_max:
                    best_max = current_max
                    best_pair = [a, b]
                    # Since we are iterating a in increasing order, 
                    # and for each a we find the smallest valid b,
                    # can we stop early?
                    # Not necessarily, because a later a' might have a smaller b'.
                    # But if best_max == min_val + 1 (i.e., 212), we can't do better than 212 (since a,b > 210 and distinct, min max is 212).
                    # Actually min max is 212 if {211, 212} works.
                    if best_max == 212:
                        # Can't get lower than 212 (distinct ints > 210)
                        # Return immediately? 
                        # Need to verify validity.
                        prod = 210 * a * b
                        s = math.isqrt(prod)
                        if s * s == prod:
                            return [int(a), int(b)]
                # We can break if we found a b such that b <= a? 
                # No, we want to minimize max. 
                # If b > a, max is b. 
                # If we find a b, is it the best for this a? 
                # Yes, smallest b gives smallest max(a,b) for fixed a.
                # So we can break the y-loop for this a.
                break
            y += 1
            
    if best_pair:
        # Final verification
        prod = 210 * best_pair[0] * best_pair[1]
        s = math.isqrt(prod)
        if s * s == prod:
            return [int(best_pair[0]), int(best_pair[1])]
            
    # Fallback: 1 element
    # 210 * 840 = 176400 = 420^2. 
    # 840 is in range.
    return [840]

if __name__ == "__main__":
    res = construct()
    print(res)
    prod = 210
    for x in res:
        prod *= x
    s = math.isqrt(prod)
    print(f"Product: {prod}, Sqrt: {s}, Valid: {s*s == prod}")
    print(f"Value: {max(res) - 210}")
