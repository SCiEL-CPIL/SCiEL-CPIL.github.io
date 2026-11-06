import math
import time
import random

MAX_MOD = 128
DEADLINE = 60.0
MARGIN = 5.0

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - MARGIN
    
    all_mods = list(range(2, MAX_MOD + 1))
    n_mods = len(all_mods)
    
    # Pre-compute gcds
    gcd_table = {}
    for m1 in all_mods:
        for m2 in all_mods:
            if m1 != m2:
                gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    def fast_disjoint(r1, m1, r2, m2):
        g = gcd_table[(m1, m2)]
        return (r2 - r1) % g != 0
    
    def is_valid_assignment(assignment, moduli):
        """Check if assignment is valid for the given moduli set."""
        n = len(moduli)
        for i in range(n):
            for j in range(i+1, n):
                if not fast_disjoint(assignment[i], moduli[i], assignment[j], moduli[j]):
                    return False
        return True
    
    def greedy_initial(moduli, seed_val):
        """Greedy assignment."""
        random.seed(seed_val)
        n = len(moduli)
        assignment = [None] * n
        order = list(range(n))
        random.shuffle(order)
        
        for idx in order:
            m = moduli[idx]
            residues_to_try = list(range(m))
            random.shuffle(residues_to_try)
            for r in residues_to_try:
                valid = True
                for j in range(n):
                    if j == idx:
                        continue
                    if assignment[j] is not None:
                        if not fast_disjoint(r, m, assignment[j], moduli[j]):
                            valid = False
                            break
                if valid:
                    assignment[idx] = r
                    break
        return assignment
    
    def local_search(assignment, moduli, t_limit, seed_val):
        """Local search to improve the assignment."""
        random.seed(seed_val)
        n = len(moduli)
        
        # Count how many are assigned
        def count_assigned():
            return sum(1 for a in assignment if a is not None)
        
        best_count = count_assigned()
        
        while time.time() < t_limit:
            # Pick a random unassigned modulus
            unassigned = [i for i in range(n) if assignment[i] is None]
            if not unassigned:
                break
            
            random.shuffle(unassigned)
            improved = False
            
            for idx in unassigned:
                m = moduli[idx]
                # Try all residues
                for r in range(m):
                    valid = True
                    for j in range(n):
                        if j == idx:
                            continue
                        if assignment[j] is not None:
                            if not fast_disjoint(r, m, assignment[j], moduli[j]):
                                valid = False
                                break
                    if valid:
                        assignment[idx] = r
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Try to reassign an existing modulus to free up space
                assigned = [i for i in range(n) if assignment[i] is not None]
                if not assigned:
                    break
                random.shuffle(assigned)
                
                for idx in assigned:
                    m = moduli[idx]
                    old_r = assignment[idx]
                    # Try to move this modulus to a different residue
                    for new_r in range(m):
                        if new_r == old_r:
                            continue
                        # Check if new_r is valid with all others
                        valid = True
                        for j in range(n):
                            if j == idx:
                                continue
                            if assignment[j] is not None:
                                if not fast_disjoint(new_r, m, assignment[j], moduli[j]):
                                    valid = False
                                    break
                        if valid:
                            # Now check if any unassigned modulus can be placed
                            # Try to place one unassigned modulus
                            unassigned = [i for i in range(n) if assignment[i] is None]
                            if unassigned:
                                random.shuffle(unassigned)
                                for uidx in unassigned:
                                    um = moduli[uidx]
                                    for ur in range(um):
                                        valid2 = True
                                        for j in range(n):
                                            if j == uidx:
                                                continue
                                            if j == idx:
                                                # Use new_r for idx
                                                if not fast_disjoint(ur, um, new_r, m):
                                                    valid2 = False
                                                    break
                                            elif assignment[j] is not None:
                                                if not fast_disjoint(ur, um, assignment[j], moduli[j]):
                                                    valid2 = False
                                                    break
                                        if valid2:
                                            assignment[idx] = new_r
                                            assignment[uidx] = ur
                                            improved = True
                                            break
                                    if improved:
                                        break
                            if not improved:
                                # Just move idx to new_r if it helps (it doesn't directly, but might help later)
                                assignment[idx] = new_r
                                improved = True
                            break
                    if improved:
                        break
                
                if not improved:
                    break
            
            cur_count = count_assigned()
            if cur_count > best_count:
                best_count = cur_count
        
        return assignment
    
    # Strategy:
    # 1. Try greedy on all moduli with different seeds
    # 2. Then local search to improve
    # 3. Keep the best
    
    best_assignment = None
    best_count = 0
    
    # Try a few seeds for greedy
    for seed in range(20):
        if time.time() > deadline - 10.0:
            break
        
        assignment = greedy_initial(all_mods, seed)
        count = sum(1 for a in assignment if a is not None)
        
        if count > best_count:
            best_assignment = assignment[:]
            best_count = count
        
        # Local search
        if time.time() < deadline - 5.0:
            assignment_ls = local_search(assignment[:], all_mods, deadline - 2.0, seed)
            count_ls = sum(1 for a in assignment_ls if a is not None)
            if count_ls > best_count:
                best_assignment = assignment_ls
                best_count = count_ls
    
    if best_assignment is None:
        # Fallback
        return [[0, 2]]
    
    result = []
    for i in range(len(all_mods)):
        if best_assignment[i] is not None:
            result.append([best_assignment[i], all_mods[i]])
    
    return result
