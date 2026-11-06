import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin
    
    # 1. Precompute primes up to 10^6
    # sqrt(10^12) = 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    def phi(n, primes):
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            result -= result // temp
        return result

    def check_window(s, k, primes):
        seen = set()
        for i in range(k):
            val = phi(s + i, primes)
            if val in seen:
                return False
            seen.add(val)
        return True

    # 2. Search for a valid window of length 200
    # We aim for k=200.
    k_target = 200
    max_s = 10**12 - k_target + 1
    min_s = 10**11
    
    # Try some deterministic points first, maybe near the top?
    # Or just random.
    # Let's try a few random points.
    random.seed(42)
    
    # Try a few specific ranges or just random
    attempts = 0
    while time.time() - start_time < deadline:
        # Pick a random start point
        s = random.randint(min_s, max_s)
        
        if check_window(s, k_target, primes):
            return {'start': s, 'length': k_target}
        
        attempts += 1
        # If we've tried enough, maybe reduce k? No, we want max k.
        # But if 200 is hard, we might need to fall back.
        # However, distinct totients are common for large n.
        
        # To be safe, if we don't find 200 quickly, we can try smaller k?
        # But the score is k. So we want the largest k.
        # If we can't find 200, we should find the largest k we can.
        # Let's track the best k found.
        
    # If the loop finishes without finding 200, we should return the best found.
    # But the loop above only checks for 200.
    # Let's modify to search for the best k.
    
    # Reset and do a proper search for max k
    # We can do a binary search on k? No, it's not monotonic.
    # We can just try to find 200. If not, try 199, etc.
    # But 200 is very likely to exist.
    
    # Let's refine the search:
    # 1. Try to find 200.
    # 2. If not found in 10s, try 190.
    # 3. If not found in 20s, try 150.
    # etc.
    
    # Actually, the previous loop already ran for most of the time.
    # Let's assume we found it. If not, we return a fallback.
    
    # Fallback: return a small k, e.g., 10, from a small s.
    # But we should try to maximize.
    
    # Let's restructure the search to be adaptive.
    
    # Re-run logic with adaptive k
    start_time = time.time()
    best_k = 1
    best_s = 1
    
    # Try k=200 first
    for k in [200, 190, 180, 150, 100, 50, 10]:
        if time.time() - start_time > deadline - 5:
            break
        found = False
        # Try 50 random points for this k
        for _ in range(50):
            if time.time() - start_time > deadline - 5:
                break
            s = random.randint(min_s, max_s)
            if check_window(s, k, primes):
                best_k = k
                best_s = s
                found = True
                break
        if found:
            # We found a window of length k. 
            # Can we extend it?
            # Try to extend to the right or left?
            # The task requires s >= 1 and s+k-1 <= 10^12.
            # We can try to extend the window.
            # But k is capped at 200.
            # If k < 200, we can try to find a longer one.
            # But if we found 200, we are done.
            if k == 200:
                return {'start': best_s, 'length': best_k}
            # If k < 200, we continue to try larger k? 
            # No, the loop goes 200, 190... so we tried 200 first.
            # If we found 190, it means 200 failed.
            # So we return 190.
            return {'start': best_s, 'length': best_k}
            
    # If nothing found (unlikely), return a trivial result
    # Check small numbers for a small run
    for s in range(1, 1000):
        for k in range(10, 0, -1):
            if check_window(s, k, primes):
                return {'start': s, 'length': k}
                
    return {'start': 1, 'length': 1}
