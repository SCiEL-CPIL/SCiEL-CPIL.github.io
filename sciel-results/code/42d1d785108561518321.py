import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s margin
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    def compute_score(n):
        factors = set()
        for k in range(1, 25):
            x = n + k
            y = 1
            for p in primes:
                while x % p == 0:
                    y *= p
                    x //= p
            factors.add(y)
        return len(factors)
    
    best_n = 1000
    best_score = compute_score(1000)
    
    # Random search
    # Range 1000 to 1000000
    while time.time() < deadline:
        n = random.randint(1000, 1000000)
        score = compute_score(n)
        if score < best_score:
            best_score = score
            best_n = n
            # If we found 1 or 2, we can stop? 
            # Min possible is likely > 1.
            if best_score == 2: 
                 # Can it be 1? 
                 # If all y_k are same. 
                 # y_k parity must be same. 
                 # But some are even, some odd. 
                 # So y_k cannot all be same. 
                 # So min is at least 2? 
                 # If all even y_k are 2 and all odd y_k are 1? 
                 # Then set is {1, 2}. Size 2.
                 pass

    return best_n
