import time

def construct():
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23)
    
    def smooth_part(x):
        s = 1
        for p in primes:
            while x % p == 0:
                s *= p
                x //= p
        return s

    best_n = 1000
    best_count = 100
    
    start_time = time.time()
    limit = 1000000
    
    # Scan first
    for n in range(1000, limit + 1):
        if time.time() - start_time > 50:
            break
            
        smooths = set()
        for i in range(1, 25):
            smooths.add(smooth_part(n + i))
        
        count = len(smooths)
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 3:
                # Theoretical minimum (K=1,2 impossible)
                return best_n

    return best_n
