import time

def construct():
    deadline = time.time() + 82.0
    TARGET = 123457
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        lp = 1
        d = 2
        while d * d <= n:
            while n % d == 0:
                lp = d
                n //= d
            d += 1
        if n > 1:
            lp = n
        return lp
    
    def generate_smooth_numbers(max_prime, limit):
        """Generate all max_prime-smooth numbers <= limit using proper DFS"""
        primes = []
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643]:
            if p <= max_prime:
                primes.append(p)
            else:
                break
        
        result = set()
        
        def dfs(idx, current):
            if time.time() > deadline:
                return
            if current > limit:
                return
            if idx == len(primes):
                result.add(current)
                return
            val = current
            while val <= limit:
                dfs(idx + 1, val)
                val *= primes[idx]
        
        dfs(0, 1)
        return sorted(result)
    
    best = None
    best_val = float('inf')
    
    # Try increasing smoothness levels
    for max_p in [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643]:
        if time.time() > deadline:
            break
        
        smooth_nums = generate_smooth_numbers(max_p, TARGET)
        if not smooth_nums:
            continue
        smooth_set = set(smooth_nums)
        
        found_at_level = False
        for a in smooth_nums:
            if time.time() > deadline:
                break
            b = TARGET - a
            if b < 1:
                continue
            if b in smooth_set:
                pa = largest_prime_factor(a)
                pb = largest_prime_factor(b)
                val = max(pa, pb)
                if val < best_val:
                    best_val = val
                    best = [a, b]
                found_at_level = True
        
        # If we found a pair with actual max prime factor < max_p, 
        # we've essentially found a solution at a lower smoothness level.
        # Since we're iterating in order, continue to see if we can do better.
        if best is not None and time.time() > deadline - 3:
            break
    
    if best is None:
        best = [1, 123456]
    
    return best
