"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _check_pair(r1, m1, r2, m2):
    """Return True if classes are disjoint (gcd(m1,m2) does not divide r1-r2)."""
    g = gcd(m1, m2)
    return (r1 - r2) % g != 0

def _is_valid_set(classes):
    """Check if all pairs are disjoint."""
    n = len(classes)
    for i in range(n):
        for j in range(i+1, n):
            r1, m1 = classes[i]
            r2, m2 = classes[j]
            if not _check_pair(r1, m1, r2, m2):
                return False
    return True

def _csp_solve(moduli, deadline):
    """
    Solve the CSP for a given set of moduli.
    Returns a dict {modulus: residue} if solvable, else None.
    Uses backtracking with forward checking.
    """
    n = len(moduli)
    if n == 0:
        return {}
    
    # Precompute gcds and constraints
    # constraints[i] is a list of (j, g) for all j != i
    constraints = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            g = gcd(moduli[i], moduli[j])
            if g > 1: # If g=1, no constraint
                constraints[i].append((j, g))
                constraints[j].append((i, g))
    
    # Order variables by degree (most constrained first)
    var_order = sorted(range(n), key=lambda i: len(constraints[i]), reverse=True)
    
    # Domains: list of available residues for each variable
    domains = [list(range(moduli[i])) for i in range(n)]
    
    # Assignment dict
    assignment = {}
    
    def is_consistent(var, val):
        """Check if assigning val to var is consistent with current assignment."""
        for neighbor, g in constraints[var]:
            if neighbor in assignment:
                if (val - assignment[neighbor]) % g == 0:
                    return False
        return True

    def forward_check(var, val):
        """
        Check if assigning val to var leaves any neighbor with an empty domain.
        Returns a list of (neighbor, removed_values) to restore, or None if failure.
        """
        changes = []
        for neighbor, g in constraints[var]:
            if neighbor in assignment:
                continue
            original_domain = domains[neighbor][:]
            new_domain = [v for v in domains[neighbor] if (v - val) % g != 0]
            if not new_domain:
                # Restore changes made so far
                for n_idx, rem_vals in changes:
                    domains[n_idx] = rem_vals
                return None
            changes.append((neighbor, original_domain))
            domains[neighbor] = new_domain
        return changes

    def backtrack(index):
        if index == n:
            return True
        
        if time.time() > deadline:
            return False
            
        var = var_order[index]
        
        # Least Constraining Value (LCV) heuristic: try values that rule out fewest options for neighbors
        # Simplified: just try values in random order or sorted
        values = domains[var][:]
        random.shuffle(values)
        
        for val in values:
            if not is_consistent(var, val):
                continue
            
            assignment[var] = val
            fc_changes = forward_check(var, val)
            
            if fc_changes is not None:
                if backtrack(index + 1):
                    return True
            
            # Undo assignment and forward checking
            assignment.pop(var)
            if fc_changes is not None:
                for n_idx, orig_dom in fc_changes:
                    domains[n_idx] = orig_dom
                    
        return False

    if backtrack(0):
        # Map back to moduli
        result = {moduli[i]: assignment[i] for i in range(n)}
        return result
    else:
        return None

def _find_max_classes(max_mod=128, deadline=None):
    if deadline is None:
        deadline = time.time() + 85  # Safety margin
        
    best_set = []
    best_size = 0
    
    all_moduli = list(range(2, max_mod+1))
    
    # Multi-start randomized greedy with CSP solver
    for _ in range(500): # Cap iterations
        if time.time() > deadline:
            break
            
        moduli = all_moduli[:]
        random.shuffle(moduli)
        
        current_moduli = []
        current_solution = {}
        
        for m in moduli:
            # Try to add m to current_moduli
            new_moduli = current_moduli + [m]
            
            # Check if CSP is solvable
            # Optimization: Only run full CSP if the new modulus has conflicts
            # For now, just run it. It's fast for small sets.
            sol = _csp_solve(new_moduli, deadline)
            
            if sol is not None:
                current_moduli = new_moduli
                current_solution = sol
                
        if len(current_moduli) > best_size:
            best_size = len(current_moduli)
            best_set = [[r, m] for m, r in current_solution.items()]
            
    return best_set

def construct():
    result = _find_max_classes(max_mod=128)
    return result
# EVOLVE-BLOCK-END
