import math
import random
import time

def _get_primes(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    n = 256
    primes = _get_primes(n)
    
    best_sum = 0
    best_set = []
    
    # Run multiple greedy passes with random tie-breaking
    for _ in range(100):
        groups = list(primes)
        
        while True:
            max_gain = -1
            best_pair = None
            candidates = []
            
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    prod = groups[i] * groups[j]
                    if prod <= n:
                        gain = prod - groups[i] - groups[j]
                        if gain > max_gain:
                            max_gain = gain
                            best_pair = (i, j)
                            candidates = [(i, j)]
                        elif gain == max_gain:
                            candidates.append((i, j))
            
            if best_pair is None:
                break
            
            # Random tie-breaking
            i, j = random.choice(candidates)
            
            # Merge j into i
            groups[i] = groups[i] * groups[j]
            groups.pop(j)
            
        current_sum = sum(groups)
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = groups[:]
            
    # Include 1
    best_set.append(1)
    
    return best_set
