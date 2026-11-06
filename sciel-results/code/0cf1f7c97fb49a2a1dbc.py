"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Moduli in [20, 200]
    all_moduli = list(range(20, 201))
    all_moduli.sort()  # Ascending order to prioritize high 1/m
    
    n_moduli = len(all_moduli)
    
    # Precompute suffix sums of 1/m for pruning
    # suffix_sum[i] = sum(1/m_j for j in range(i, n_moduli))
    suffix_sum = [Fraction(0, 1)] * (n_moduli + 1)
    for i in range(n_moduli - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + Fraction(1, all_moduli[i])
    
    best_score = Fraction(0, 1)
    best_classes = []
    
    def can_assign(m, r, assigned_moduli, assigned_residues):
        """Check if residue r for modulus m is compatible with existing assignments."""
        for j in range(len(assigned_moduli)):
            m_j = assigned_moduli[j]
            r_j = assigned_residues[j]
            g = gcd(m, m_j)
            if (r - r_j) % g == 0:
                return False
        return True

    def get_valid_residues(m, assigned_moduli, assigned_residues):
        """Get list of valid residues for m."""
        valid = []
        for r in range(m):
            if can_assign(m, r, assigned_moduli, assigned_residues):
                valid.append(r)
        return valid

    def dfs(idx, current_score, assigned_moduli, assigned_residues):
        nonlocal best_score, best_classes, start_time, deadline
        
        # Time check
        if time.time() > deadline:
            return
        
        # Pruning: Upper bound
        if current_score + suffix_sum[idx] <= best_score:
            return
        
        if idx == n_moduli:
            if current_score > best_score:
                best_score = current_score
                best_classes = [[assigned_moduli[i], assigned_residues[i]] for i in range(len(assigned_moduli))]
            return
        
        m = all_moduli[idx]
        
        # Option 1: Skip m
        dfs(idx + 1, current_score, assigned_moduli, assigned_residues)
        
        # Option 2: Include m
        # Find valid residues
        valid_residues = get_valid_residues(m, assigned_moduli, assigned_residues)
        
        for r in valid_residues:
            # Add m, r
            assigned_moduli.append(m)
            assigned_residues.append(r)
            
            dfs(idx + 1, current_score + Fraction(1, m), assigned_moduli, assigned_residues)
            
            # Backtrack
            assigned_moduli.pop()
            assigned_residues.pop()

    dfs(0, Fraction(0, 1), [], [])
    
    # Fallback if no solution found (should not happen)
    if not best_classes:
        best_classes = [[20, 0]]
        best_score = Fraction(1, 20)
        
    return best_classes
# EVOLVE-BLOCK-END
