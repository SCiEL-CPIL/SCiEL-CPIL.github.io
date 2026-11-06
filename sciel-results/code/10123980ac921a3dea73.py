"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd
from fractions import Fraction

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Precompute conflict data.
    2. Multi-start greedy with diverse orderings.
    3. Branch-and-bound MWIS on conflict graph.
    4. Local search polish.
    """
    deadline = time.time() + 145.0
    random.seed(42)
    
    moduli = list(range(20, 201))
    n_mods = len(moduli)
    mod_to_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute gcd table
    gcd_table = [[0]*n_mods for _ in range(n_mods)]
    for i, m1 in enumerate(moduli):
        for j, m2 in enumerate(moduli):
            gcd_table[i][j] = gcd(m1, m2)
    
    def pair_ok(m1, r1, m2, r2):
        """Check if two residue classes are disjoint."""
        g = gcd_table[mod_to_idx[m1]][mod_to_idx[m2]]
        return (r1 % g) != (r2 % g)
    
    def is_valid(classes):
        """Check if all pairs are disjoint."""
        n = len(classes)
        for i in range(n):
            for j in range(i+1, n):
                m1, r1 = classes[i]
                m2, r2 = classes[j]
                if not pair_ok(m1, r1, m2, r2):
                    return False
        return True
    
    def score(classes):
        return sum(1.0/m for m, r in classes)
    
    def find_valid_residue(m, classes):
        """Find a valid residue for modulus m given existing classes."""
        idx_m = mod_to_idx[m]
        for r in range(m):
            valid = True
            for m2, r2 in classes:
                idx_m2 = mod_to_idx[m2]
                g = gcd_table[idx_m][idx_m2]
                if (r % g) == (r2 % g):
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def greedy_build(order):
        """Build solution greedily in given order."""
        classes = []
        for m in order:
            if time.time() > deadline - 15.0:
                break
            r = find_valid_residue(m, classes)
            if r is not None:
                classes.append([m, r])
        return classes
    
    # Multiple greedy initializations with diverse orderings
    best_classes = []
    best_score = 0.0
    
    # Strategy 1: Smallest moduli first (highest weight)
    order1 = sorted(moduli, key=lambda m: 1.0/m, reverse=True)
    # Strategy 2: Most composite numbers first (more conflicts, harder to place)
    def divisor_count(m):
        return sum(1 for i in range(2, m) if m % i == 0)
    order2 = sorted(moduli, key=lambda m: -divisor_count(m))
    # Strategy 3: Random
    order3 = moduli[:]
    random.shuffle(order3)
    # Strategy 4: Prime moduli first (fewest conflicts)
    def is_prime(m):
        if m < 2: return False
        for i in range(2, int(math.sqrt(m))+1):
            if m % i == 0:
                return False
        return True
    order4 = sorted(moduli, key=lambda m: (not is_prime(m), m))
    
    for order in [order1, order2, order3, order4]:
        if time.time() > deadline - 20.0:
            break
        classes = greedy_build(order)
        s = score(classes)
        if s > best_score:
            best_score = s
            best_classes = classes[:]
    
    # Branch and bound for MWIS
    # Nodes are (m, r) pairs. Edge if incompatible.
    # We want max weight independent set where weight = 1/m.
    # But we can only pick at most one r per m.
    
    def branch_and_bound(initial_classes, time_limit=30.0):
        """
        Branch and bound to improve the solution.
        We model it as: for each modulus, we either include it (with some valid r) or not.
        """
        start_time = time.time()
        current_classes = initial_classes[:]
        current_score = score(current_classes)
        best_classes = current_classes[:]
        best_score = current_score
        
        # Get moduli not in current solution
        current_moduli = set(c[0] for c in current_classes)
        missing_moduli = [m for m in moduli if m not in current_moduli]
        
        # Sort missing by weight (1/m) descending
        missing_moduli.sort(key=lambda m: -1.0/m)
        
        # Try to add moduli one by one, with backtracking
        def try_add_moduli(classes, moduli_to_try, depth=0):
            nonlocal best_classes, best_score
            if time.time() - start_time > time_limit:
                return classes
            
            if depth > 0 and time.time() - start_time > time_limit:
                return classes
            
            if not moduli_to_try:
                return classes
            
            m = moduli_to_try[0]
            rest = moduli_to_try[1:]
            
            # Option 1: Skip m
            result1 = try_add_moduli(classes, rest, depth+1)
            
            # Option 2: Try to add m
            r = find_valid_residue(m, classes)
            if r is not None:
                new_classes = classes + [[m, r]]
                new_score = score(new_classes)
                if new_score > best_score:
                    best_score = new_score
                    best_classes = new_classes[:]
                result2 = try_add_moduli(new_classes, rest, depth+1)
                
                # Choose better result
                if score(result2) > score(result1):
                    return result2
            
            return result1
        
        # Use a more efficient approach: iterative improvement with limited backtracking
        improved = True
        while improved and time.time() - start_time < time_limit:
            improved = False
            current_moduli = set(c[0] for c in current_classes)
            missing = [m for m in moduli if m not in current_moduli]
            missing.sort(key=lambda m: -1.0/m)
            
            current_score = score(current_classes)
            
            # Try adding each missing modulus
            for m_add in missing:
                if time.time() - start_time > time_limit:
                    break
                r = find_valid_residue(m_add, current_classes)
                if r is not None:
                    new_classes = current_classes + [[m_add, r]]
                    new_score = score(new_classes)
                    if new_score > current_score + 1e-12:
                        current_classes = new_classes
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_classes = current_classes[:]
                        improved = True
            
            # Try swapping: remove one, add another
            if time.time() - start_time > time_limit:
                break
            current_moduli = set(c[0] for c in current_classes)
            missing = [m for m in moduli if m not in current_moduli]
            missing.sort(key=lambda m: -1.0/m)
            
            for m_add in missing[:10]:  # Limit to top 10 candidates
                if time.time() - start_time > time_limit:
                    break
                for m_remove in list(current_moduli):
                    if 1.0/m_add <= 1.0/m_remove + 1e-12:
                        continue
                    new_classes = [c for c in current_classes if c[0] != m_remove]
                    r = find_valid_residue(m_add, new_classes)
                    if r is not None:
                        new_classes.append([m_add, r])
                        new_score = score(new_classes)
                        if new_score > current_score + 1e-12:
                            current_classes = new_classes
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_classes = current_classes[:]
                            improved = True
                            break
        
        return best_classes, best_score
    
    # Run B&B from best initial
    if time.time() < deadline - 25.0:
        bb_classes, bb_score = branch_and_bound(best_classes, time_limit=25.0)
        if bb_score > best_score:
            best_classes = bb_classes
            best_score = bb_score
    
    # Final local search polish
    def local_search(classes, time_limit=5.0):
        start = time.time()
        current = classes[:]
        current_score = score(current)
        best = current[:]
        best_score = current_score
        
        while time.time() - start < time_limit:
            improved = False
            current_moduli = set(c[0] for c in current)
            missing = [m for m in moduli if m not in current_moduli]
            missing.sort(key=lambda m: -1.0/m)
            
            # Try adds
            for m in missing:
                if time.time() - start > time_limit:
                    break
                r = find_valid_residue(m, current)
                if r is not None:
                    new_classes = current + [[m, r]]
                    new_score = score(new_classes)
                    if new_score > current_score + 1e-12:
                        current = new_classes
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best = current[:]
                        improved = True
            
            # Try swaps
            if time.time() - start > time_limit:
                break
            current_moduli = set(c[0] for c in current)
            missing = [m for m in moduli if m not in current_moduli]
            missing.sort(key=lambda m: -1.0/m)
            
            for m_add in missing[:15]:
                if time.time() - start > time_limit:
                    break
                for m_remove in current_moduli:
                    if 1.0/m_add <= 1.0/m_remove + 1e-12:
                        continue
                    new_classes = [c for c in current if c[0] != m_remove]
                    r = find_valid_residue(m_add, new_classes)
                    if r is not None:
                        new_classes.append([m_add, r])
                        new_score = score(new_classes)
                        if new_score > current_score + 1e-12:
                            current = new_classes
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best = current[:]
                            improved = True
                            break
            
            if not improved:
                break
        
        return best, best_score
    
    if time.time() < deadline - 5.0:
        ls_classes, ls_score = local_search(best_classes, time_limit=5.0)
        if ls_score > best_score:
            best_classes = ls_classes
            best_score = ls_score
    
    return best_classes
# EVOLVE-BLOCK-END
