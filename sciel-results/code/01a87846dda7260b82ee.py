"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin (90s limit, leave buffer)
    
    n = P['n']
    # Candidates are integers from 2 to 256
    candidates = list(range(2, n + 1))
    
    # Precompute prime factorization for all numbers in range
    # This helps in checking divisibility conditions efficiently if needed,
    # but for now, direct modulo is likely fast enough for small sets.
    
    # Strategy: 
    # 1. Start with a strong seed set (e.g., primes, or large numbers)
    # 2. Greedily add elements that don't violate constraints
    # 3. Use local search (swaps) to improve if stuck
    
    # Initial set: Primes are a good start because they have few divisors.
    # However, a prime p can divide b*c if b*c is a multiple of p.
    # If b and c are not multiples of p, b*c mod p != 0.
    # So primes are "safer" than composites? Not necessarily, but let's try.
    
    # Let's try a different structural approach:
    # Sort candidates in descending order.
    # Add a number if it doesn't conflict with any existing pair.
    # This tends to favor larger numbers, which have fewer multiples in the range.
    
    current_set = set()
    
    # Helper to check if adding x to S is valid
    def can_add(x, S):
        # For all distinct b, c in S, x must not divide b*c
        # And for all b in S, c=x, x must not divide b*x (always false, x divides x*b)
        # Wait, the condition is: for every THREE DISTINCT members a, b, c, a does not divide b*c.
        # If we add x to S, we need to check:
        # 1. For any pair (b, c) in S, x does not divide b*c.
        # 2. For any b in S, x does not divide b*x? No, the triple is (a,b,c) distinct.
        #    If a=x, then b and c must be in S and distinct from x.
        #    So we check: for all distinct b, c in S, x does not divide b*c.
        #    Also, if a is in S, and one of b,c is x:
        #    Let a in S, b=x, c in S (c!=a). Then a must not divide x*c.
        #    Let a in S, b in S (b!=a), c=x. Then a must not divide b*x.
        #    So we need to check:
        #    For all distinct b, c in S: x does not divide b*c
        #    For all distinct a, b in S: a does not divide x*b  (and a does not divide b*x, same)
        
        S_list = list(S)
        m = len(S_list)
        
        # Check 1: x as the divisor 'a'
        # x does not divide b*c for any distinct b, c in S
        for i in range(m):
            b = S_list[i]
            for j in range(i + 1, m):
                c = S_list[j]
                if (b * c) % x == 0:
                    return False
        
        # Check 2: existing elements as divisor 'a', with x as one of the multiplicands
        # For each a in S, and each b in S (b != a), check if a divides x*b
        for i in range(m):
            a = S_list[i]
            for j in range(m):
                if i == j: continue
                b = S_list[j]
                if (x * b) % a == 0:
                    return False
                    # Note: if a divides x*b, then the triple (a, x, b) is invalid.
                    # Since a, x, b are distinct (x not in S, a != b), this is a violation.
        
        return True

    # Greedy construction: try adding numbers in a specific order
    # Order 1: Descending (larger numbers have fewer multiples, might be safer)
    # Order 2: Primes first?
    # Let's try descending first.
    
    # To allow for some exploration, we can try multiple random orders or perturbations.
    # But given the time limit, a simple robust greedy + local search is better.
    
    # Let's use a randomized greedy approach with restarts if time permits.
    best_set = set()
    
    # We will run a few iterations with different random seeds/orders
    for _ in range(5): # Up to 5 restarts
        if time.time() > deadline - 10.0: # Leave 10s for final polish
            break
        
        # Randomize the order of candidates
        order = candidates.copy()
        random.shuffle(order)
        
        # Or use a heuristic: sort by "complexity" (number of divisors) descending?
        # Or just random shuffle is fine for now.
        
        current_set = set()
        
        for x in order:
            if time.time() > deadline - 5.0:
                break
            if can_add(x, current_set):
                current_set.add(x)
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
    
    # Local search on best_set
    # Try to swap elements to increase size?
    # Or just try to add more elements if possible.
    
    # Final polish: try to add any remaining candidate
    for x in candidates:
        if time.time() > deadline - 1.0:
            break
        if x in best_set:
            continue
        if can_add(x, best_set):
            best_set.add(x)
    
    # Return as list of ints
    return list(best_set)

# EVOLVE-BLOCK-END
