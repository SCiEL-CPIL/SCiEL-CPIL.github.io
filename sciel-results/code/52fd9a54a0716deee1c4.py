import math
import time
from itertools import combinations

def construct():
    """
    Returns the largest subset of [1, 255] containing 255 such that 
    gcd(a, b) > 1 for all distinct pairs.
    
    Strategy:
    1. Build the compatibility graph for all numbers in [1, 255].
    2. Use greedy maximum clique heuristic starting from 255:
       At each step, add the element with the highest degree in the 
       remaining compatible subgraph.
    3. Compare with the base set of multiples of 3 (size 85).
    """
    
    n = 255
    deadline = time.time() + 40  # Safety margin for 90s limit
    
    # Base set: multiples of 3 (size 85)
    base_set = list(range(3, n + 1, 3))
    
    # Build compatibility: comp[x] = set of y in [1,255] where gcd(x,y) > 1
    # For efficiency, pre-compute prime factors
    def get_prime_factors(x):
        factors = set()
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors.add(d)
                x //= d
            d += 1
        if x > 1:
            factors.add(x)
        return factors
    
    # Pre-compute prime factors for all numbers
    prime_factors = [set() for _ in range(n + 1)]
    for i in range(2, n + 1):
        prime_factors[i] = get_prime_factors(i)
    
    # Two numbers are compatible if they share at least one prime factor
    # Build adjacency: adj[x] = set of y != x such that gcd(x,y) > 1
    adj = [set() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for y in range(x + 1, n + 1):
            # Check if they share a prime factor
            if prime_factors[x] & prime_factors[y]:
                adj[x].add(y)
                adj[y].add(x)
    
    # Greedy maximum clique starting from 255
    # S = current clique, candidates = elements compatible with all in S
    S = {255}
    # Candidates: all elements compatible with 255, excluding 255 itself
    candidates = set(adj[255])
    candidates.discard(255)
    
    while candidates:
        if time.time() > deadline:
            break
        
        # For each candidate, compute its degree within the candidate set
        # (i.e., how many other candidates it's compatible with)
        # Pick the one with highest degree
        best_candidate = None
        best_degree = -1
        
        for x in candidates:
            # Degree of x within candidates
            deg = len(adj[x] & candidates)
            if deg > best_degree:
                best_degree = deg
                best_candidate = x
        
        if best_candidate is None:
            break
        
        # Add best_candidate to S
        S.add(best_candidate)
        
        # Update candidates: keep only those compatible with best_candidate
        candidates = candidates & adj[best_candidate]
    
    # Compare with base set
    greedy_size = len(S)
    base_size = len(base_set)
    
    if greedy_size > base_size:
        return sorted(S)
    else:
        return base_set
