import time
import random
from math import gcd

def construct():
    """
    Finds the maximum number of pairwise disjoint infinite residue classes
    with distinct moduli in [2, 128].
    Uses a Mask-based MRV (Minimum Remaining Values) Greedy approach with restarts.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    max_modulus = 128
    moduli_list = list(range(2, max_modulus + 1))
    
    best_solution = []
    best_size = 0
    
    # Precompute gcds for all pairs of moduli to speed up updates
    # 127 moduli, so ~8000 pairs.
    gcd_cache = {}
    for i in range(len(moduli_list)):
        m1 = moduli_list[i]
        for j in range(i, len(moduli_list)):
            m2 = moduli_list[j]
            if m1 == m2:
                continue
            g = gcd(m1, m2)
            gcd_cache[(m1, m2)] = g
            gcd_cache[(m2, m1)] = g

    def get_gcd(m1, m2):
        return gcd_cache.get((m1, m2), 1)

    def run_greedy():
        # Initialize masks: bit r is 1 if residue r is valid for modulus m
        masks = {}
        for m in moduli_list:
            masks[m] = (1 << m) - 1
        
        remaining = set(moduli_list)
        selected = []
        
        while remaining:
            # Find modulus with fewest valid residues (MRV heuristic)
            best_m = None
            min_count = float('inf')
            
            for m in remaining:
                count = bin(masks[m]).count('1')
                if count == 0:
                    continue
                if count < min_count:
                    min_count = count
                    best_m = m
                    # Early exit if we find a modulus with only 1 valid residue
                    if count == 1:
                        break
            
            if best_m is None:
                break
            
            # Pick a residue from best_m
            mask = masks[best_m]
            valid_residues = []
            # Extract valid residues
            for r in range(best_m):
                if mask & (1 << r):
                    valid_residues.append(r)
            
            if not valid_residues:
                break
                
            # Random choice for diversity
            r = random.choice(valid_residues)
            
            # Add to selected
            selected.append((r, best_m))
            remaining.remove(best_m)
            
            # Update masks for all remaining moduli
            # For each remaining m, remove residues r' such that r' % gcd(best_m, m) == r % gcd(best_m, m)
            for m in remaining:
                g = get_gcd(best_m, m)
                target = r % g
                # Clear bits in masks[m] where r_prime % g == target
                # Iterate r_prime from target to m-1 step g
                for r_prime in range(target, m, g):
                    masks[m] &= ~(1 << r_prime)
                    
        return selected

    # Run multiple trials
    while time.time() < deadline - 2.0:
        sol = run_greedy()
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = sol.copy()
            
    result = [[r, m] for r, m in best_solution]
    
    # Ensure non-empty
    if not result:
        result = [[0, 2]]
        
    return result

# EVOLVE-BLOCK-END
