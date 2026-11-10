import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    limit = 10**18
    # Primes up to ~100 are sufficient since product of first 15 primes is ~6e17
    # We can include a few more just in case, but exponents will be 1 or 0.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    best_n = 614889782588491410 # 15-prime primorial
    best_tau = 32768

    def search(idx, current_n, current_tau, max_exp):
        nonlocal best_n, best_tau
        if time.time() > deadline:
            return
        
        p = primes[idx]
        # e goes from 0 to max_exp
        # We need current_n * p^e <= limit
        # and e <= max_exp
        
        for e in range(max_exp + 1):
            # Check overflow/limit
            # If e==0, val = current_n
            # Else val = current_n * p^e
            # To be safe and fast, we can break if current_n * p^e > limit
            # We can maintain a running product or just compute p**e.
            # Since e is small (max ~60 for 2, less for others), pow is fast.
            
            val = current_n * (p ** e)
            if val > limit:
                break
                
            new_tau = current_tau * (e + 1)
            
            # Pruning: 
            # The maximum additional divisors we can get from remaining primes is 2^(number of remaining primes)
            # This is a weak upper bound. 
            # A better bound: if new_tau * 2^(len(primes)-idx-1) <= best_tau, we can prune?
            # Not strictly, because exponents can be >1, but usually they are 1 for large primes.
            # Let's just rely on the limit check and time limit.
            
            if idx + 1 >= len(primes):
                if new_tau > best_tau:
                    best_tau = new_tau
                    best_n = val
            else:
                # Next max exponent is e (since exponents must be non-increasing)
                search(idx + 1, val, new_tau, e)
                
            if time.time() > deadline:
                return

    search(0, 1, 1, 60)
    
    return best_n
