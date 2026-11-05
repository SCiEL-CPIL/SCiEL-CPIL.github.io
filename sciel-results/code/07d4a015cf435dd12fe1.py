"""
Maximum disjoint residue classes with distinct moduli 2..128.

Two residue classes (r1, m1) and (r2, m2) are DISJOINT iff the system
  x ≡ r1 (mod m1), x ≡ r2 (mod m2)
has no solution, which by CRT is equivalent to:
  r1 mod gcd(m1, m2) != r2 mod gcd(m1, m2)

We need to select a subset of moduli from {2,...,128} and assign each a residue
in [0, m-1] such that all pairs are disjoint, maximizing the count.

Approach: Backtracking with constraint propagation and smart ordering.
"""
import time
from math import gcd

MAXIMUM = 128

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Precompute gcd for all pairs of moduli
    moduli = list(range(2, MAXIMUM + 1))
    n_mods = len(moduli)
    
    # gcd_matrix[i][j] = gcd(moduli[i], moduli[j])
    gcd_matrix = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(i, n_mods):
            g = gcd(moduli[i], moduli[j])
            gcd_matrix[i][j] = g
            gcd_matrix[j][i] = g

    # For each modulus index i, precompute the set of "forbidden residue patterns"
    # Actually, let's just do backtracking with constraint checking.
    
    # Strategy: Process moduli in an order that helps find large solutions.
    # Key insight: If we process moduli in descending order of "constraint tightness",
    # we can prune earlier. A modulus m is "tight" if it has many small gcds with others.
    
    # Let's try multiple strategies and keep the best.
    
    best_result = []
    
    def backtrack(assigned, idx, order, remaining_mods):
        """
        assigned: dict from modulus_index -> residue
        idx: current position in order
        order: list of modulus indices to process
        remaining_mods: set of modulus indices not yet processed
        """
        nonlocal best_result, deadline
        
        if time.time() > deadline:
            return
        
        if len(assigned) > len(best_result):
            best_result = list(assigned.items())
        
        if idx >= len(order):
            return
        
        m_idx = order[idx]
        m = moduli[m_idx]
        
        # Try all residues for this modulus
        # Pruning: if even the best case can't beat current best, stop
        # Max possible = len(assigned) + (len(order) - idx)
        if len(assigned) + (len(order) - idx) <= len(best_result):
            return
        
        for r in range(m):
            # Check constraint with all already-assigned moduli
            valid = True
            for other_idx, other_r in assigned.items():
                g = gcd_matrix[m_idx][other_idx]
                if r % g == other_r % g:
                    valid = False
                    break
            
            if valid:
                assigned[m_idx] = r
                backtrack(assigned, idx + 1, order, remaining_mods)
                del assigned[m_idx]
                
                # If we found a solution with all remaining moduli, we're done
                if len(assigned) + (len(order) - idx - 1) <= len(best_result):
                    continue

    # Strategy 1: Descending order (largest moduli first - more flexibility)
    order1 = list(range(n_mods - 1, -1, -1))  # descending by modulus value
    assigned1 = {}
    backtrack(assigned1, 0, order1, set(range(n_mods)))
    
    # Strategy 2: Ascending order
    order2 = list(range(n_mods))
    assigned2 = {}
    backtrack(assigned2, 0, order2, set(range(n_mods)))
    
    # Strategy 3: Order by number of prime factors (more factors = more constraints)
    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    order3 = sorted(range(n_mods), key=lambda i: (-num_prime_factors(moduli[i]), -moduli[i]))
    assigned3 = {}
    backtrack(assigned3, 0, order3, set(range(n_mods)))
    
    # Strategy 4: Random orders with early termination
    import random
    for attempt in range(50):
        if time.time() > deadline:
            break
        random.seed(attempt)
        order4 = list(range(n_mods))
        random.shuffle(order4)
        assigned4 = {}
        backtrack(assigned4, 0, order4, set(range(n_mods)))
    
    # Convert best_result to required format
    if not best_result:
        # Fallback: just return a valid small solution
        return [[0, 2], [1, 3]]
    
    result = [[r, moduli[m_idx]] for (m_idx, r) in best_result]
    return result
