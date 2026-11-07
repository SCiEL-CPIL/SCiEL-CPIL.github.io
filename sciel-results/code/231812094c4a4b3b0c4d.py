import time
import random

def construct():
    N = 256
    deadline = time.time() + 50  # 50s limit
    
    # Precompute divisibility checks for speed?
    # a % b == 0
    
    def is_valid_set(S):
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            for j in range(i+1, n):
                b = S_list[j]
                for k in range(j+1, n):
                    c = S_list[k]
                    if a % (b*c) == 0 or b % (a*c) == 0 or c % (a*b) == 0:
                         # Wait, condition is a | bc. 
                         # a divides b*c.
                         if (b*c) % a == 0:
                            return False
                         if (a*c) % b == 0:
                            return False
                         if (a*b) % c == 0:
                            return False
        return True

    def can_add(x, S):
        # S is a list
        n = len(S)
        for i in range(n):
            b = S[i]
            for j in range(i+1, n):
                c = S[j]
                if (b*c) % x == 0:
                    return False
                if (x*c) % b == 0:
                    return False
                if (x*b) % c == 0:
                    return False
        return True

    best_size = 0
    best_set = []
    
    # Primes are a good starting point
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True
        
    primes = [i for i in range(2, N+1) if is_prime(i)]
    best_size = len(primes)
    best_set = primes[:]
    
    # Try Randomized Greedy
    for _ in range(500): # 500 restarts
        if time.time() > deadline:
            break
            
        candidates = list(range(2, N+1))
        random.shuffle(candidates)
        
        # Bias: Primes first?
        # Sort by "safety"? Primes are safer.
        # Let's just shuffle and add.
        
        S = []
        for x in candidates:
            if can_add(x, S):
                S.append(x)
                
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set
