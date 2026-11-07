import math
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_VAL = 10210
    TARGET = 210
    
    # Sieve for smallest prime factor
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL ** 0.5) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Collect primes
    primes = []
    for i in range(2, MAX_VAL + 1):
        if spf[i] == i:
            primes.append(i)
    
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    def compute_parity_mask(n):
        mask = 0
        while n > 1:
            p = spf[n]
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            if cnt % 2 == 1:
                mask |= (1 << prime_to_bit[p])
        return mask
    
    # Precompute parity masks
    parity = {}
    for i in range(211, MAX_VAL + 1):
        parity[i] = compute_parity_mask(i)
    
    target_mask = compute_parity_mask(TARGET)
    
    best_solution = None
    
    # Try increasing B (max value)
    for B in range(211, min(MAX_VAL + 1, 500)):
        if time.time() > deadline:
            break
        
        numbers = list(range(211, B + 1))
        n = len(numbers)
        
        # BFS over parity masks
        dist = {}  # mask -> (count, tuple_of_numbers)
        dist[0] = (0, ())
        queue = deque([0])
        
        found = False
        solution = None
        
        max_depth = min(n, 6)
        for depth in range(1, max_depth + 1):
            if time.time() > deadline:
                break
            
            new_q = deque()
            while queue:
                mask = queue.popleft()
                _, path = dist[mask]
                
                for idx in range(len(path), n):
                    num = numbers[idx]
                    new_mask = mask ^ parity[num]
                    
                    if new_mask == target_mask:
                        solution = path + (num,)
                        found = True
                        break
                    
                    if new_mask not in dist:
                        dist[new_mask] = (depth, path + (num,))
                        new_q.append(new_mask)
                
                if found:
                    break
            
            if found:
                break
            
            queue = new_q
        
        if found and solution:
            prod = 1
            for x in solution:
                prod *= x
            prod *= TARGET
            k = math.isqrt(prod)
            if k * k == prod:
                best_solution = list(solution)
                break
    
    if best_solution is None:
        best_solution = [840]
    
    return [int(x) for x in best_solution]
