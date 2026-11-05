import time
import heapq
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    N = 512
    
    # Sieve for primes up to 511
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return is_prime
    
    is_prime = sieve(511)
    primes = [p for p in range(2, 512) if is_prime[p]]
    
    # Precompute factors for all numbers 1..511
    # factors[n] is a list of prime factors
    factors = [[] for _ in range(512)]
    for i in range(2, 512):
        if is_prime[i]:
            factors[i] = [i]
        else:
            for p in primes:
                if p * p > i:
                    break
                if i % p == 0:
                    factors[i] = [p] + factors[i // p]
                    break
            else:
                factors[i] = [i] # Should not happen for composite i < 512 if primes list is correct, but safe

    # Initial set: 1 and all primes
    current_items = [1] + primes
    
    def get_score(item):
        return 1.0 / (512 - item)
    
    # Greedy Loop with Merge and Split
    while True:
        if time.time() > deadline:
            break
            
        best_gain = 0
        best_action = None # ('merge', i, j) or ('split', i, b, c)
        
        n_items = len(current_items)
        
        # Find best merge
        for i in range(n_items):
            a = current_items[i]
            if a == 1:
                continue # Merging 1 is never beneficial
            for j in range(i + 1, n_items):
                b = current_items[j]
                if b == 1:
                    continue
                prod = a * b
                if prod >= 512:
                    continue
                
                score_ab = get_score(prod)
                score_a = get_score(a)
                score_b = get_score(b)
                
                gain = score_ab - score_a - score_b
                
                if gain > best_gain + 1e-15:
                    best_gain = gain
                    best_action = ('merge', i, j)
        
        # Find best split
        for i in range(n_items):
            a = current_items[i]
            if a == 1:
                continue
            # Try all factorizations a = b * c
            # Since a is a product of distinct primes, we can iterate over subsets of its prime factors
            # Or simply iterate divisors b of a where b > 1 and b < a
            # a is small (<512), so this is fast
            for b in range(2, int(a**0.5) + 1):
                if a % b == 0:
                    c = a // b
                    if c == 1 or c == a:
                        continue
                    # Check if b and c are coprime? 
                    # In our construction, items are always products of distinct primes from the original set.
                    # So any factorization b*c=a will have gcd(b,c)=1 if and only if b and c don't share prime factors.
                    # Since a has distinct prime factors, any divisor b will have a subset of primes, and c will have the complement.
                    # Thus gcd(b,c) is always 1.
                    
                    score_b = get_score(b)
                    score_c = get_score(c)
                    score_a = get_score(a)
                    
                    gain = score_b + score_c - score_a
                    
                    if gain > best_gain + 1e-15:
                        best_gain = gain
                        best_action = ('split', i, b, c)
        
        if best_action is None:
            break
        
        if best_action[0] == 'merge':
            _, i, j = best_action
            a = current_items[i]
            b = current_items[j]
            prod = a * b
            
            # Remove b and a, add prod
            if i > j:
                i, j = j, i
            del current_items[j]
            del current_items[i]
            current_items.append(prod)
            
        else: # split
            _, i, b, c = best_action
            a = current_items[i]
            del current_items[i]
            current_items.append(b)
            current_items.append(c)
            
    return current_items
