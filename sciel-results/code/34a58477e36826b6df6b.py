"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Build all possible (m, r) pairs
    all_pairs = []
    for m in range(20, 201):
        for r in range(m):
            all_pairs.append((m, r))
    
    n = len(all_pairs)
    
    # Precompute compatibility: compatible[i][j] = True if pair i and j are compatible
    # Two pairs (m1, r1) and (m2, r2) are incompatible if r1 % gcd(m1,m2) == r2 % gcd(m1,m2)
    # We'll build adjacency list where adj[i] contains indices j > i that are incompatible with i
    
    # For efficiency, group by modulus
    # Actually, let's precompute incompatibility using a different approach
    # For each modulus m, and each residue r, we need to check against all selected
    
    # Better approach: use branch and bound with bitsets
    # But n can be up to ~20000, so bitsets of 20000 bits = 2500 bytes each, times 20000 = 50MB
    # That's feasible but let's think about a smarter approach
    
    # Key insight: we can process moduli one at a time
    # For each modulus m, we can choose at most one residue r
    # But the constraint is pairwise: for any two chosen (m1,r1) and (m2,r2), we need r1 != r2 mod gcd(m1,m2)
    
    # Let's use a recursive approach:
    # Process moduli in order of decreasing 1/m (i.e., increasing m)
    # At each step, decide which residue to use for the current modulus, or skip it
    
    # For upper bound: sum of 1/m for all remaining moduli (optimistic - assumes all can be chosen)
    
    moduli = list(range(20, 201))
    moduli.sort(key=lambda m: -1.0/m)  # Process smaller moduli first (higher weight)
    
    # Precompute gcd table for moduli
    gcd_table = {}
    for i in range(len(moduli)):
        for j in range(i+1, len(moduli)):
            gcd_table[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
    
    best_solution = []
    best_score = Fraction(0, 1)
    
    # For tracking which residues are used for each modulus
    # selected_residues[m] = r if we chose residue r for modulus m, None if skipped
    
    def branch_and_bound(idx, selected, current_score, available_residues):
        """
        idx: index into moduli list
        selected: list of [m, r] chosen so far
        current_score: Fraction
        available_residues: dict mapping m -> set of valid residues for that m given current selections
        """
        nonlocal best_solution, best_score
        
        if time.time() > deadline:
            return
        
        # Upper bound: current_score + sum of 1/m for all remaining moduli
        # (optimistic: assumes we can pick one residue for each remaining modulus)
        ub = current_score
        for i in range(idx, len(moduli)):
            ub += Fraction(1, moduli[i])
        
        if ub <= best_score:
            return
        
        if idx == len(moduli):
            if current_score > best_score:
                best_score = current_score
                best_solution = list(selected)
            return
        
        m = moduli[idx]
        
        # Try each valid residue for this modulus
        valid_residues = sorted(available_residues.get(m, range(m)))
        
        # Also try skipping this modulus
        # First try each valid residue
        for r in valid_residues:
            # Check if this residue is compatible with all previously selected
            # Actually, available_residues should already have filtered this
            new_selected = selected + [[m, r]]
            new_score = current_score + Fraction(1, m)
            
            # Update available_residues for future moduli
            new_available = {}
            for future_m in moduli[idx+1:]:
                # Filter residues for future_m that are compatible with (m, r)
                g = math.gcd(m, future_m)
                r_mod_g = r % g
                valid = [fr for fr in range(future_m) if fr % g != r_mod_g]
                # Also need to be compatible with all previously selected
                for prev_m, prev_r in selected:
                    g2 = math.gcd(future_m, prev_m)
                    prev_r_mod_g2 = prev_r % g2
                    valid = [fr for fr in valid if fr % g2 != prev_r_mod_g2]
                new_available[future_m] = set(valid)
            
            # Check if any future modulus has no valid residues - that's ok, we can skip it
            # But if a modulus has no valid residues, we just skip it
            
            branch_and_bound(idx + 1, new_selected, new_score, new_available)
        
        # Try skipping this modulus
        new_available_skip = dict(available_residues)
        branch_and_bound(idx + 1, selected, current_score, new_available_skip)
    
    # Initialize available_residues: for each modulus, all residues 0..m-1 are initially available
    initial_available = {m: set(range(m)) for m in moduli}
    
    branch_and_bound(0, [], Fraction(0, 1), initial_available)
    
    if best_solution:
        return best_solution
    else:
        # Fallback to simple solution
        return [[20 * j, j - 1] for j in range(1, 11)]
# EVOLVE-BLOCK-END
