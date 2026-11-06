import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    max_mod = 128
    # All possible moduli
    moduli = list(range(2, max_mod + 1))
    
    # Precompute gcd for all pairs of moduli to speed up compatibility checks
    # gcd_cache[m1][m2] = gcd(m1, m2)
    # Actually, we only need gcd(m, m') for m, m' in moduli.
    # Let's just compute on the fly or use a small cache.
    
    def is_compatible(m1, r1, m2, r2):
        if m1 == m2:
            return False
        g = math.gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    # State for backtracking
    # selected: list of (m, r)
    # used_moduli: set of moduli
    best_solution = []
    
    # Heuristic ordering for moduli to try first?
    # MRV will handle the dynamic ordering.
    
    def backtrack(selected, used_moduli, current_size):
        nonlocal best_solution
        if time.time() > deadline:
            return
        
        if current_size > len(best_solution):
            best_solution = selected[:]
        
        # If we have used all moduli, stop
        if len(used_moduli) == len(moduli):
            return
            
        # Find the modulus with the fewest compatible residues (MRV)
        best_m = None
        best_candidates = None
        min_count = float('inf')
        
        # Optimization: If we can't beat the current best, prune.
        # Max possible additional classes = len(moduli) - len(used_moduli)
        # If current_size + (len(moduli) - len(used_moduli)) <= len(best_solution):
        #     return
        
        for m in moduli:
            if m in used_moduli:
                continue
            
            # Count compatible residues for m
            candidates = []
            for r in range(m):
                compatible = True
                for (m2, r2) in selected:
                    if not is_compatible(m, r, m2, r2):
                        compatible = False
                        break
                if compatible:
                    candidates.append(r)
            
            if len(candidates) == 0:
                # Dead end for this modulus, but maybe other moduli are still possible?
                # No, if a modulus has 0 candidates, it just can't be added.
                # But we want to maximize count. We can skip it.
                # However, if *all* remaining moduli have 0 candidates, we are stuck.
                # MRV picks the one with min count. If min count is 0, we can't add any more?
                # Not necessarily. We can skip moduli with 0 candidates.
                # So we only consider moduli with > 0 candidates for the "next" pick.
                # If no moduli have > 0 candidates, we stop.
                pass
            else:
                if len(candidates) < min_count:
                    min_count = len(candidates)
                    best_m = m
                    best_candidates = candidates
                    # Early termination if we found a modulus with only 1 option
                    if min_count == 1:
                        break
        
        if best_m is None or min_count == 0:
            return
        
        # Try each candidate residue for best_m
        # Shuffle candidates to introduce some randomness if we want, or keep deterministic
        # Deterministic is usually fine for small search spaces.
        for r in best_candidates:
            new_selected = selected + [(best_m, r)]
            new_used = used_moduli | {best_m}
            backtrack(new_selected, new_used, current_size + 1)
            if time.time() > deadline:
                return
        
    # Initial call
    # We can start with an empty set.
    backtrack([], set(), 0)
    
    # Convert list of tuples to list of lists [r, m] as required
    # The task asks for [residue, modulus] pairs
    result = [[r, m] for m, r in best_solution]
    
    # Ensure non-empty
    if not result:
        # Fallback to seed-like solution
        result = [[(1<<k)-1, 1<<(k+1)] for k in range(7)]
        
    return result
