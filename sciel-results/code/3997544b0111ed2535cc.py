import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
        
    primes = sieve(1000)
    n_primes = len(primes)
    
    MAX_L = 100000 # Cap for score, but we can check up to this
    # To be safe, if we cover 100000, we might as well check a bit more to ensure stability?
    # No, if we cover 1..100000, score is 100000.
    # If we cover 1..100001, score is still 100000.
    # So we just need to find if 100000 is covered.
    # But we want to maximize L. If we can't reach 100000, we want the max L.
    # So we should search up to a limit where we expect failure.
    # Let's use 200000 for the array to be safe, but return min(L, 100000) logic is handled by evaluator?
    # "The score is the largest L ... capped at 10^5".
    # So if my L is 100001, score is 100000.
    # If my L is 50000, score is 50000.
    # So I should try to maximize L, but once L >= 100000, I can stop optimizing for higher L.
    
    SEARCH_LIMIT = 200000
    coverage = np.zeros(SEARCH_LIMIT + 1, dtype=np.int32)
    
    # Initialize residues
    # A better initialization: 
    # For small primes, residues matter more.
    # Random is okay.
    residues = [random.randint(0, p-1) for p in primes]
    
    for i, p in enumerate(primes):
        r = residues[i]
        coverage[r::p] += 1
        
    def get_L():
        invalid = np.where(coverage[1:] < 2)[0]
        if len(invalid) == 0:
            return SEARCH_LIMIT
        return int(invalid[0])
        
    current_L = get_L()
    best_L = current_L
    best_residues = residues.copy()
    
    if best_L >= 100000:
        return best_residues
        
    T = 100.0
    cooling = 0.9995
    min_T = 0.5
    
    # To speed up, we can batch updates? No, SA needs step-by-step.
    
    while time.time() < deadline:
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = residues[i]
        new_r = random.randint(0, p - 1)
        
        if old_r == new_r:
            continue
            
        coverage[old_r::p] -= 1
        coverage[new_r::p] += 1
        
        new_L = get_L()
        
        if new_L > current_L:
            current_L = new_L
            residues[i] = new_r
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
                if best_L >= 100000:
                    break
        else:
            delta = new_L - current_L
            if delta < 0:
                # Avoid log(0) or underflow
                if delta < -500:
                    prob = 0.0
                else:
                    prob = np.exp(delta / T)
                if random.random() < prob:
                    current_L = new_L
                    residues[i] = new_r
                else:
                    coverage[new_r::p] -= 1
                    coverage[old_r::p] += 1
            else:
                current_L = new_L
                residues[i] = new_r
                
        T *= cooling
        if T < min_T:
            T = min_T

    return best_residues
