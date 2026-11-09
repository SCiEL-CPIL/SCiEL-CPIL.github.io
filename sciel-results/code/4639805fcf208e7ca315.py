"""Rewritten program: DIRECT approach to pack distinct multiples of fixed divisors."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

P = {
    'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39],
    'maximum': 1000000
}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    MAX_V = P['maximum']
    deadline = time.time() + 85.0  # safety margin before 90s kill
    
    # Calculate LCM of all divisors to exploit periodicity
    def lcm(a, b):
        return a * b // gcd(a, b)
    lcm_all = reduce(lcm, divisors)
    
    # Kuhn's algorithm for bipartite matching
    def find_assignment_kuhn(candidates):
        # candidates: list of lists of values
        # We want to assign each divisor i to a distinct value from candidates[i]
        
        # Sort divisors by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        
        # match[v] = u  (value v is matched to divisor u)
        match = {}
        
        def try_kuhn(u, visited):
            # u is divisor index
            for v in candidates[u]:
                if v in visited:
                    continue
                visited.add(v)
                # If v is not matched, or if the divisor currently matched to v can be rematched
                if v not in match or try_kuhn(match[v], visited):
                    match[v] = u
                    return True
            return False
        
        # Try to find a matching for each divisor in order
        for u in order:
            visited = set()
            if not try_kuhn(u, visited):
                return None
        
        # Reconstruct assignment
        assignment = [0] * n
        for v, u in match.items():
            assignment[u] = v
            
        return assignment

    # Iterate span W from minimum possible (9) upwards
    # The maximum span is bounded by the divisors, but 1000 is a safe upper bound for this small set
    for W in range(9, 1000):
        if time.time() > deadline:
            break
        
        # Iterate window start L
        # We only need to check L up to lcm_all because the pattern repeats
        for L in range(1, lcm_all + 1):
            if time.time() > deadline:
                break
            
            # Check if window [L, L+W] is within bounds
            if L + W > MAX_V:
                continue

            candidates = []
            valid_window = True
            
            for d in divisors:
                # Find multiples of d in [L, L+W]
                first_k = (L + d - 1) // d
                last_k = (L + W) // d
                
                if first_k > last_k:
                    valid_window = False
                    break
                
                # Generate the list of candidates
                cands = [k * d for k in range(first_k, last_k + 1)]
                candidates.append(cands)
            
            if not valid_window:
                continue
            
            # Try to assign distinct values using Kuhn's algorithm
            result = find_assignment_kuhn(candidates)
            
            if result is not None:
                # Verify the solution
                if len(set(result)) != n:
                    continue
                for i, d in enumerate(divisors):
                    if result[i] % d != 0:
                        continue
                if any(v < 1 or v > MAX_V for v in result):
                    continue
                
                return [int(v) for v in result]
        
        # If we finish all L for a given W without success, increment W
    
    # Fallback
    fallback = [int(d) for d in divisors]
    return fallback

# EVOLVE-BLOCK-END
