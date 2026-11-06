"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Greedy initial construction: Iterate moduli in order of 1/m (descending), adding a modulus if a valid residue exists.
    2. Local Search (Hill Climbing): Try to improve the set of moduli by:
       a. Adding a missing modulus if a valid residue can be found.
       b. Swapping a selected modulus (with lower 1/m) for a missing modulus (with higher 1/m) if the swap results in a valid assignment and higher total score.
       c. Reassigning residues to allow more moduli to be added.
    3. Backtracking search for the best subset of moduli.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Moduli range
    mod_range = list(range(20, 201))
    
    # Sort by 1/m descending (i.e., m ascending)
    sorted_mods = sorted(mod_range, key=lambda m: -1.0/m)
    
    def find_valid_residue(m, existing):
        """
        Find a residue r in [0, m) such that for all (m_j, r_j) in existing:
        r != r_j mod gcd(m, m_j)
        """
        g = {}
        for m_j, r_j in existing:
            g_j = gcd(m, m_j)
            if g_j not in g:
                g[g_j] = set()
            g[g_j].add(r_j % g_j)
        
        # For each residue r in [0, m), check if it's valid
        # Optimization: precompute constraints
        # For a given r, it's invalid if for any m_j, r % gcd(m, m_j) is in the forbidden set for that gcd
        
        # We can iterate r from 0 to m-1
        # But m can be up to 200, so this is fast
        
        # Let's precompute the forbidden residues for each possible gcd
        # Actually, the constraint is: for each m_j, r mod gcd(m, m_j) != r_j mod gcd(m, m_j)
        
        # We can check each r
        for r in range(m):
            valid = True
            for m_j, r_j in existing:
                g_j = gcd(m, m_j)
                if r % g_j == r_j % g_j:
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def is_valid_configuration(config):
        """Check if a list of [m, r] is pairwise disjoint."""
        for i in range(len(config)):
            for j in range(i+1, len(config)):
                m1, r1 = config[i]
                m2, r2 = config[j]
                g = gcd(m1, m2)
                if r1 % g == r2 % g:
                    return False
        return True
    
    def calc_score(config):
        return sum(1.0/m for m, r in config)
    
    # Greedy initial construction
    config = []
    for m in sorted_mods:
        if time.time() > deadline:
            break
        r = find_valid_residue(m, config)
        if r is not None:
            config.append([m, r])
    
    best_config = config[:]
    best_score = calc_score(config)
    
    # Local search
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        # Try to add a missing modulus
        for m in sorted_mods:
            if time.time() > deadline:
                break
            if any(cm == m for cm, cr in config):
                continue
            r = find_valid_residue(m, config)
            if r is not None:
                new_config = config + [[m, r]]
                new_score = calc_score(new_config)
                if new_score > best_score:
                    best_config = new_config
                    best_score = new_score
                    config = new_config
                    improved = True
                    break
        
        # Try to swap a selected modulus for a missing one
        if time.time() > deadline:
            break
        selected_mods = set(cm for cm, cr in config)
        missing_mods = [m for m in sorted_mods if m not in selected_mods]
        
        for m_new in missing_mods:
            if time.time() > deadline:
                break
            if 1.0/m_new <= 1.0/max(selected_mods) if selected_mods else False:
                continue
            # Try to swap out a selected modulus
            for idx, (m_old, r_old) in enumerate(config):
                if 1.0/m_new <= 1.0/m_old:
                    continue
                # Remove m_old
                config_without = [c for i, c in enumerate(config) if i != idx]
                r_new = find_valid_residue(m_new, config_without)
                if r_new is not None:
                    new_config = config_without + [[m_new, r_new]]
                    new_score = calc_score(new_config)
                    if new_score > best_score:
                        best_config = new_config
                        best_score = new_score
                        config = new_config
                        improved = True
                        break
            if improved:
                break
    
    # Backtracking search for the best subset of moduli
    # This is expensive, so we do it only if we have time
    if time.time() < deadline - 30:
        # Try to find a better configuration using backtracking
        # We'll use a recursive search with pruning
        def backtrack_search(mods, current_config, current_score, depth=0):
            if time.time() > deadline:
                return
            if not mods:
                if current_score > best_score:
                    nonlocal best_config, best_score
                    best_config = current_config[:]
                    best_score = current_score
                return
            
            m = mods[0]
            remaining = mods[1:]
            
            # Option 1: Skip m
            backtrack_search(remaining, current_config, current_score, depth+1)
            
            # Option 2: Add m
            r = find_valid_residue(m, current_config)
            if r is not None:
                backtrack_search(remaining, current_config + [[m, r]], current_score + 1.0/m, depth+1)
        
        # Only try backtracking on a subset of moduli to keep it feasible
        # We'll focus on the smaller moduli which have higher 1/m
        # Let's try the first 30 smallest moduli
        small_mods = sorted_mods[:30]
        backtrack_search(small_mods, [], 0.0)
    
    return best_config
# EVOLVE-BLOCK-END
