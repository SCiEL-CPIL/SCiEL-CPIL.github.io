import math
import time

def construct():
    # Primes up to 50
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_n = 2
    best_score = 0
    limit = 10**18
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    for k in range(1, len(primes) + 1):
        if time.time() > deadline:
            break
            
        n = 1
        for i in range(k):
            n *= primes[i]
            if n > limit:
                break
        if n > limit:
            break
            
        # Calculate score for n
        # Generate divisors
        divs = [1]
        for i in range(k):
            p = primes[i]
            new_divs = [d * p for d in divs]
            divs.extend(new_divs)
        
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
                
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n
