import time
from math import gcd

def construct():
    """
    Finds the maximum number of pairwise disjoint infinite residue classes
    with distinct moduli in [2, 128].
    Uses backtracking search on moduli with coprime-pruning and residue feasibility checks.
    """
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    max_modulus = 128
    moduli_list = list(range(2, max_modulus + 1))
    
    # Precompute gcds
    gcd_cache = {}
    for i, m1 in enumerate(moduli_list):
        for j in range(i, len(moduli_list)):
            m2 = moduli_list[j]
            if m1 == m2:
                continue
            g = gcd(m1, m2)
            gcd_cache[(m1, m2)] = g
            gcd_cache[(m2, m1)] = g

    def get_gcd(m1, m2):
        return gcd_cache.get((m1, m2), 1)

    best_size = 0
    best_solution = []
    
    # Sort moduli to improve heuristic? 
    # Let's try sorting by number of divisors descending (more factors = more likely to be non-coprime)
    # Or just keep natural order.
    
    def get_valid_residues(m, current_pairs):
        """
        Returns list of residues r in [0, m) such that
        r is disjoint from all (r', m') in current_pairs.
        """
        valid = []
        for r in range(m):
            is_valid = True
            for r_prime, m_prime in current_pairs:
                g = get_gcd(m, m_prime)
                if r % g == r_prime % g:
                    is_valid = False
                    break
            if is_valid:
                valid.append(r)
        return valid

    def search(available_moduli, current_pairs):
        nonlocal best_size, best_solution, start_time, deadline
        
        if time.time() > deadline - 2.0:
            return
        
        if len(current_pairs) > best_size:
            best_size = len(current_pairs)
            best_solution = current_pairs.copy()
            
        if not available_moduli:
            return
            
        # Heuristic: Pick the modulus with the fewest valid residues (MRV)
        # This helps prune dead ends early.
        best_m = -1
        min_residues = float('inf')
        best_residues = []
        
        for m in available_moduli:
            residues = get_valid_residues(m, current_pairs)
            if len(residues) < min_residues:
                min_residues = len(residues)
                best_m = m
                best_residues = residues
                
        if min_residues == 0:
            return
            
        # Remove best_m from available
        new_available = [m for m in available_moduli if m != best_m]
        
        # Prune available moduli that are coprime with best_m
        # If gcd(m, best_m) == 1, they cannot be disjoint, so remove them.
        pruned_available = []
        for m in new_available:
            if get_gcd(m, best_m) > 1:
                pruned_available.append(m)
        
        # Branch on each valid residue for best_m
        for r in best_residues:
            new_pairs = current_pairs + [(r, best_m)]
            search(pruned_available, new_pairs)
            
        # Optional: Branch on excluding best_m?
        # If we exclude best_m, we might find a larger set using other moduli.
        # However, MRV suggests best_m is the most constrained.
        # If we can't place it, we might still place others.
        # But if we CAN place it, we should probably try to include it first.
        # To be safe for maximization, we should also try excluding it.
        
        # Exclude best_m
        search(new_available, current_pairs)

    # Initial call
    search(moduli_list, [])
    
    # Convert best_solution to required format
    result = []
    for r, m in best_solution:
        result.append([r, m])
        
    # Ensure non-empty
    if not result:
        result = [[0, 2]]
        
    return result
