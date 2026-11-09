import time
import math
import random

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    moduli = list(range(20, 201))
    
    def is_compatible(m, r, assign):
        """Check if [m, r] is disjoint from all in assign."""
        for m2, r2 in assign.items():
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True

    def find_residue(m, assign):
        """Find a valid residue for m given assign. Returns None if not found."""
        # Randomize order to help escape symmetries/local traps in greedy
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            if is_compatible(m, r, assign):
                return r
        return None

    # 1. Greedy Initialization
    assign = {}
    # Try adding moduli in increasing order (smallest 1/m first is not it, smallest m first)
    # Actually, to maximize sum 1/m, we want small m.
    # Greedy: sort by 1/m descending (m ascending)
    for m in sorted(moduli, key=lambda x: 1/x, reverse=True):
        r = find_residue(m, assign)
        if r is not None:
            assign[m] = r
            
    best_assign = dict(assign)
    best_score = sum(1/m for m in assign)

    # 2. Local Search (Hill Climbing with Swaps)
    # We try to improve by swapping a larger modulus for a smaller one, 
    # or adding a modulus if space is available.
    
    # Helper to calculate score
    def calc_score(a):
        return sum(1/m for m in a)

    # Iterative improvement
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try to add any missing modulus
        # Sort missing by 1/m descending
        missing = [m for m in moduli if m not in assign]
        for m in sorted(missing, key=lambda x: 1/x, reverse=True):
            r = find_residue(m, assign)
            if r is not None:
                assign[m] = r
                improved = True
                # Continue trying to add more? Or restart loop?
                # Restarting loop ensures we always try the best available adds
                break
        
        if improved:
            # Update best if better
            curr_score = calc_score(assign)
            if curr_score > best_score:
                best_score = curr_score
                best_assign = dict(assign)
            continue
        
        # If no simple add works, try swaps
        # For each modulus in assign, try to remove it and see if we can add a better one
        # To make it efficient, we only consider swaps where 1/m_in > 1/m_out
        # i.e., m_in < m_out
        
        # Randomize order of m_out to avoid deterministic cycles
        items = list(assign.items())
        random.shuffle(items)
        
        for m_out, r_out in items:
            if (time.time() - start_time) > time_limit:
                break
                
            # Temporarily remove m_out
            del assign[m_out]
            
            # Try to add any m_in < m_out that fits
            # We iterate m_in from 20 to m_out-1
            swapped = False
            for m_in in range(20, m_out):
                if m_in in assign:
                    continue
                r_in = find_residue(m_in, assign)
                if r_in is not None:
                    # Check if this swap improves score
                    # 1/m_in > 1/m_out is guaranteed since m_in < m_out
                    assign[m_in] = r_in
                    swapped = True
                    improved = True
                    break
            
            if not swapped:
                # Restore m_out
                assign[m_out] = r_out
            
            if swapped:
                curr_score = calc_score(assign)
                if curr_score > best_score:
                    best_score = curr_score
                    best_assign = dict(assign)
                # Restart the outer loop to try more adds/swaps
                break

    # Convert to required format
    result = [[m, r] for m, r in best_assign.items()]
    return result
