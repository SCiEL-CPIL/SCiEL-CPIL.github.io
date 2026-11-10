import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    N = 512
    max_a = N - 1
    
    # Precompute prime factors
    prime_factors = [set() for _ in range(max_a + 1)]
    for p in range(2, max_a + 1):
        # Check if p is prime
        is_p_prime = True
        for i in range(2, int(math.sqrt(p)) + 1):
            if p % i == 0:
                is_p_prime = False
                break
        if is_p_prime:
            for multiple in range(p, max_a + 1, p):
                prime_factors[multiple].add(p)
                
    # Calculate values
    values = [0.0] * (max_a + 1)
    for a in range(1, max_a + 1):
        values[a] = 1.0 / (N - a)
        
    # Greedy by Value
    used_primes = set()
    selected = []
    available = list(range(1, max_a + 1))
    # Sort available by value descending
    available.sort(key=lambda x: values[x], reverse=True)
    
    for a in available:
        if time.time() > deadline:
            break
            
        if prime_factors[a].isdisjoint(used_primes):
            selected.append(a)
            used_primes.update(prime_factors[a])
            
    return selected
