import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    
    best_result = []
    best_size = 0
    
    # Define a few deterministic orderings and some random ones
    orderings = [
        sorted(moduli, reverse=True),
        sorted(moduli),
        sorted(moduli, key=lambda x: -x.bit_length()),
        sorted(moduli, key=lambda x: sum(1 for i in range(2, x+1) if x % i == 0)), # Most composite first?
    ]
    
    # Add a few random orderings
    random.seed(42)
    for _ in range(20):
        perm = moduli.copy()
        random.shuffle(perm)
        orderings.append(perm)
    
    def try_greedy(order):
        current_solution = [] # list of (residue, modulus)
        
        for m in order:
            # Check if we can add m
            # Forbidden residues for m
            forbidden = [False] * m
            
            for r_j, m_j in current_solution:
                g = gcd(m, m_j)
                # r must not be congruent to r_j mod g
                # Values of r in [0, m-1] such that r % g == r_j % g
                start = r_j % g
                # r = start, start + g, start + 2g, ... < m
                for r_val in range(start, m, g):
                    forbidden[r_val] = True
            
            # Find an available residue
            available = None
            for r in range(m):
                if not forbidden[r]:
                    available = r
                    break
            
            if available is not None:
                current_solution.append((available, m))
        
        return current_solution

    for order in orderings:
        if time.time() > deadline:
            break
            
        result = try_greedy(order)
        
        if len(result) > best_size:
            best_size = len(result)
            best_result = result
            
        # Early stop if we reach a theoretically high bound?
        # Max possible is 127. 64 is a strong baseline.
        # If we find > 80, we might stop? No, keep trying to maximize.
        
    # Convert to required format [[residue, modulus], ...]
    # Ensure canonical residues
    final_result = [[r, m] for r, m in best_result]
    
    # If no result found (should not happen), fallback to simple
    if not final_result:
        final_result = [[1, 2]]
        
    return final_result
