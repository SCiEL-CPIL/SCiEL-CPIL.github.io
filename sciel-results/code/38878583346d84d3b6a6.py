"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Candidate moduli: all integers from 20 to 200
    candidates = list(range(20, 201))
    
    def is_compatible(m_new, r_new, existing):
        """Check if (m_new, r_new) is disjoint from all in existing."""
        for (m_old, r_old) in existing:
            g = gcd(m_new, m_old)
            if r_new % g == r_old % g:
                return False
        return True
        
    def compute_score(classes):
        return sum(1.0 / m for m, r in classes)

    def find_valid_residue(m, existing):
        """Find any valid residue for m given existing set. Returns r or None."""
        # Randomize order of residues to escape local minima in different runs
        residues = list(range(m))
        random.shuffle(residues)
        for r in residues:
            if is_compatible(m, r, existing):
                return r
        return None

    best_set = []
    best_score = 0.0

    # Run multiple random restarts with greedy + local search
    num_restarts = 200
    
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
            
        current_set = []
        current_score = 0.0
        used_moduli = set()
        
        # 1. Greedy Construction
        # Sort by 1/m descending, but add some randomization to order to explore different structures
        # A small random perturbation helps escape greedy traps
        greedy_candidates = sorted(candidates, key=lambda m: (-1.0/m, random.random()))
        
        for m in greedy_candidates:
            if time.time() > deadline:
                break
            r = find_valid_residue(m, current_set)
            if r is not None:
                current_set.append((m, r))
                used_moduli.add(m)
                current_score += 1.0 / m
                
        # 2. Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any unselected modulus
            unselected = [m for m in candidates if m not in used_moduli]
            random.shuffle(unselected)
            
            for m in unselected:
                if time.time() > deadline:
                    break
                r = find_valid_residue(m, current_set)
                if r is not None:
                    current_set.append((m, r))
                    used_moduli.add(m)
                    current_score += 1.0 / m
                    improved = True
                    # Don't break, try to add more if possible
                    
            # Try to swap: remove one, add one if score improves
            # This is O(N^2) but N is small (~50-100)
            # We only do this if no additions were made or periodically
            if not improved:
                for i in range(len(current_set)):
                    if time.time() > deadline:
                        break
                    m_out, r_out = current_set[i]
                    # Try to replace m_out with any unselected m_in
                    # Only consider m_in if 1/m_in > 1/m_out
                    for m_in in unselected:
                        if 1.0 / m_in <= 1.0 / m_out:
                            continue
                        
                        # Temporarily remove m_out
                        current_set.pop(i)
                        used_moduli.remove(m_out)
                        temp_score = current_score - 1.0 / m_out
                        
                        # Check if m_in fits
                        r_in = find_valid_residue(m_in, current_set)
                        if r_in is not None:
                            # Accept the swap
                            current_set.append((m_in, r_in))
                            used_moduli.add(m_in)
                            current_score = temp_score + 1.0 / m_in
                            improved = True
                            break # Restart the swap loop as set changed
                        else:
                            # Revert
                            current_set.insert(i, (m_out, r_out))
                            used_moduli.add(m_out)
                            current_score = temp_score + 1.0 / m_out
                            
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_set = list(current_set)
            
    # Convert to required format
    result = [[m, r] for m, r in best_set]
    return result

# EVOLVE-BLOCK-END
