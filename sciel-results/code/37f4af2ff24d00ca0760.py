import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # The 11 smallest primes ≡ 1 (mod 4)
    primes_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97]
    
    # For each prime p ≡ 1 (mod 4), find x, y such that x^2 + y^2 = p
    def find_gaussian_factor(p):
        for x in range(1, int(math.isqrt(p)) + 1):
            y2 = p - x * x
            y = math.isqrt(y2)
            if y * y == y2 and y >= 0:
                return (x, y)
        return (1, 0)  # fallback
    
    # Precompute Gaussian factors
    gaussian_factors = [find_gaussian_factor(p) for p in primes_1mod4]
    
    # Verify M <= 10^18
    M = 1
    for p in primes_1mod4:
        M *= p
        if M > 10**18:
            break
    
    if M > 10**18:
        num_primes = 10
        M = 1
        for p in primes_1mod4[:num_primes]:
            M *= p
        primes_1mod4 = primes_1mod4[:num_primes]
        gaussian_factors = gaussian_factors[:num_primes]
    
    k = len(primes_1mod4)
    
    # Generate all representations
    pairs = []
    seen = set()
    
    def dfs(idx, re, im):
        if time.time() > deadline - 5:
            return
        
        if idx == k:
            x = abs(re)
            y = abs(im)
            
            # Ensure x <= y
            if x > y:
                x, y = y, x
            
            key = (x, y)
            if key not in seen:
                seen.add(key)
                if len(pairs) < 1000:
                    pairs.append([x * x, y * y])
            
            return
        
        a, b = gaussian_factors[idx]
        
        # Multiply by (a + bi)
        dfs(idx + 1, re * a - im * b, re * b + im * a)
        
        # Multiply by (a - bi)
        dfs(idx + 1, re * a + im * b, im * a - re * b)
        
        if time.time() > deadline - 5:
            return
    
    dfs(0, 1, 0)
    
    if not pairs:
        return {'N': 5, 'pairs': [[1, 4]]}
    
    return {'N': M, 'pairs': pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'])
    print(len(res['pairs']))
