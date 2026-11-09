# EVOLVE-BLOCK-START
import time
import itertools
import random
from math import isqrt

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def _prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def construct():
    deadline = time.time() + 85  # Safety margin
    n = 256
    all_nums = list(range(2, n + 1))
    primes = [x for x in all_nums if _is_prime(x)]
    
    # Precompute prime factors
    pf = {x: _prime_factors(x) for x in all_nums}
    
    # Function to check if adding x to S is valid
    def _can_add(S, x):
        # S is a list or set of ints
        # We need to check for all a, b in S, a != b, and a != x, b != x
        # If a | b*x or b | a*x or x | a*b, then invalid
        # Actually, the condition is: for all distinct a,b,c in S_new, a !| b*c
        # New triples involving x:
        # 1. x | a*b for some a,b in S
        # 2. a | x*b for some a,b in S, a!=b
        # 3. b | x*a for some a,b in S, a!=b
        # Note: cases 2 and 3 are symmetric.
        
        S_list = list(S)
        m = len(S_list)
        
        # Check x | a*b
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                if (a * b) % x == 0:
                    return False
        
        # Check a | x*b
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                if (x * b) % a == 0:
                    return False
                if (x * a) % b == 0:
                    return False
                    
        return True

    # Greedy construction: try adding numbers in descending order
    # This tends to pick large numbers which have "rare" prime factors
    current_set = []
    for x in sorted(all_nums, reverse=True):
        if _can_add(current_set, x):
            current_set.append(x)
            
    best_set = current_set[:]
    best_size = len(best_set)
    
    # Local search: try to swap elements to increase size or maintain size with better properties
    # Since we want to maximize size, we primarily look for ways to ADD an element.
    # To add an element y, we might need to REMOVE an element x.
    # If len(S) is fixed, swapping doesn't help size.
    # So we need to find a y that can be added if we remove a small subset of S.
    # Given time constraints, simple swaps that maintain size are not helpful for score.
    # However, the greedy might get stuck.
    
    # Let's try a different greedy order: Randomized restarts
    # The score depends on size.
    # If best_size is 54, we want 55+.
    
    # Try to add composites by removing primes?
    # If we remove p, can we add p^2?
    # p^2 is not in all_nums if p^2 > 256? No, 2^7=128.
    
    # Let's run a few randomized greedy passes
    for _ in range(50):
        if time.time() > deadline - 5:
            break
            
        random.shuffle(all_nums)
        curr = []
        for x in all_nums:
            if _can_add(curr, x):
                curr.append(x)
        if len(curr) > best_size:
            best_size = len(curr)
            best_set = curr[:]
            
    # Try to improve best_set by removing one element and adding two?
    # Or just trying to add one element by removing one.
    # If we remove x from best_set, can we add y?
    # Size stays same.
    # To increase size, we need to add y without removing, or remove 1 add 2.
    
    # Since _can_add is O(N^2), and N=256, it's fast enough.
    # Let's try to find a y that is "almost" addable.
    # For each y not in best_set:
    #   Find conflicts.
    #   If conflicts are caused by a single element x, try swapping x->y? No, size same.
    #   If we can remove x and add y AND z, size +1.
    
    # This is complex. Let's stick to the result of the randomized greedy.
    # The inspirations suggest 55-56 is possible.
    
    # One more heuristic: Start with primes, then try to replace primes with composites 
    # that allow adding another composite?
    
    # Let's just return the best found.
    return list(best_set)

# EVOLVE-BLOCK-END
