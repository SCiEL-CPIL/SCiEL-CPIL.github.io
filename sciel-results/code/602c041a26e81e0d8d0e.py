"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    MAX_MOD = 128
    
    # All candidate moduli
    moduli = list(range(2, MAX_MOD + 1))
    
    # Precompute gcd for all pairs
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def is_disjoint(r1, m1, r2, m2):
        g = gcd(m1, m2)
        if g == 1:
            return True
        return (r1 - r2) % g != 0
    
    # For a given set of (r,m) pairs, check if adding (r_new, m_new) is valid
    def can_add(r_new, m_new, existing):
        for (r, m) in existing:
            if not is_disjoint(r_new, m_new, r, m):
                return False
        return True
    
    # Strategy: Most-constrained-first greedy + local search
    # A modulus is "more constrained" if it shares factors with more other moduli
    
    def constraint_score(m):
        # Count how many other moduli share a non-trivial gcd with m
        count = 0
        for other in moduli:
            if other != m and gcd(m, other) > 1:
                count += 1
        return count
    
    # Sort moduli by constraint (most constrained first)
    sorted_moduli = sorted(moduli, key=lambda m: constraint_score(m), reverse=True)
    
    def greedy_construct(order):
        selected = []  # list of (r, m)
        for m in order:
            # Try residues 0..m-1, pick first valid one
            found = False
            for r in range(m):
                if can_add(r, m, selected):
                    selected.append((r, m))
                    found = True
                    break
            # If no residue works, skip this modulus
        return selected
    
    # Initial greedy
    best = greedy_construct(sorted_moduli)
    best_size = len(best)
    
    # Local search: try swapping out a selected class for an unselected one
    # Or try reassigning residues
    
    def local_search(current, iterations=20000):
        current = list(current)
        current_size = len(current)
        selected_moduli = set(m for (r, m) in current)
        unselected = [m for m in moduli if m not in selected_moduli]
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            improved = False
            
            # Try 1-for-1 swap: remove one, add another
            if unselected:
                # Try adding an unselected modulus (if possible)
                for m_new in unselected:
                    for r_new in range(m_new):
                        if can_add(r_new, m_new, current):
                            current.append((r_new, m_new))
                            selected_moduli.add(m_new)
                            unselected.remove(m_new)
                            improved = True
                            break
                    if improved:
                        break
                
                # If we couldn't add, try swap
                if not improved and unselected:
                    for i, (r_old, m_old) in enumerate(current):
                        for m_new in unselected:
                            # Try to find r_new for m_new that's disjoint with all except m_old
                            test_list = [(r, m) for j, (r, m) in enumerate(current) if j != i]
                            for r_new in range(m_new):
                                if can_add(r_new, m_new, test_list):
                                    current[i] = (r_new, m_new)
                                    selected_moduli.discard(m_old)
                                    selected_moduli.add(m_new)
                                    unselected.remove(m_new)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                
                # Try reassigning a residue to enable more additions
                if not improved:
                    for i, (r_old, m_old) in enumerate(current):
                        for r_new in range(m_old):
                            if r_new == r_old:
                                continue
                            test_list = [(r, m) for j, (r, m) in enumerate(current) if j != i]
                            if can_add(r_new, m_old, test_list):
                                # Check if this enables adding something new
                                current[i] = (r_new, m_old)
                                # Try to add
                                added = False
                                for m_add in unselected:
                                    for r_add in range(m_add):
                                        if can_add(r_add, m_add, current):
                                            current.append((r_add, m_add))
                                            selected_moduli.add(m_add)
                                            unselected.remove(m_add)
                                            added = True
                                            break
                                    if added:
                                        break
                                if not added:
                                    current[i] = (r_old, m_old)  # revert
                                else:
                                    improved = True
                                break
                        if improved:
                            break
            
            if not improved:
                break
        
        return current
    
    # Run local search
    best = local_search(best, iterations=50000)
    best_size = len(best)
    
    # Try multiple random orderings for greedy
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        random_order = moduli[:]
        random.shuffle(random_order)
        candidate = greedy_construct(random_order)
        candidate = local_search(candidate, iterations=10000)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
    
    # Convert to required format
    result = [[r, m] for (r, m) in best]
    return result

# EVOLVE-BLOCK-END
