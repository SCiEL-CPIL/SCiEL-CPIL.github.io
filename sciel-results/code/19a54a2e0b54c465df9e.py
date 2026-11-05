import math
import time

def construct():
    moduli = list(range(20, 201))
    # Sort by weight 1/m descending (smaller m has higher weight)
    moduli.sort(key=lambda x: -1.0/x)
    
    n_moduli = len(moduli)
    
    # Precompute suffix sums of weights for upper bound
    # suffix_sum[i] = sum(1/m for m in moduli[i:])
    suffix_sum = [0.0] * (n_moduli + 1)
    for i in range(n_moduli - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + 1.0 / moduli[i]
        
    best_score = 0.0
    best_solution = []
    
    # Time limit
    start_time = time.time()
    time_limit = 115.0

    def is_compatible(selection, m, r):
        """
        Check if residue r for modulus m is compatible with all existing selections.
        selection is a list of (m2, r2).
        """
        for m2, r2 in selection:
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True

    def get_valid_residues(selection, m):
        """
        Get list of valid residues for modulus m given current selection.
        """
        valid = []
        # Optimization: if selection is empty, all are valid
        if not selection:
            return list(range(m))
            
        for r in range(m):
            if is_compatible(selection, m, r):
                valid.append(r)
        return valid

    def branch_and_bound(index, current_score, current_selection):
        nonlocal best_score, best_solution
        
        # Check time
        if time.time() - start_time > time_limit:
            return

        # Pruning: Upper bound
        if current_score + suffix_sum[index] <= best_score:
            return

        # Base case
        if index == n_moduli:
            if current_score > best_score:
                best_score = current_score
                best_solution = list(current_selection)
            return

        m = moduli[index]
        
        # Get valid residues for this modulus
        valid_residues = get_valid_residues(current_selection, m)
        
        # Branch 1: Include one of the valid residues
        for r in valid_residues:
            new_selection = current_selection + [(m, r)]
            new_score = current_score + 1.0 / m
            branch_and_bound(index + 1, new_score, new_selection)
            
        # Branch 2: Skip this modulus
        branch_and_bound(index + 1, current_score, current_selection)

    # Initial greedy solution to set a good lower bound (incumbent)
    greedy_sel = []
    for m in moduli:
        valids = get_valid_residues(greedy_sel, m)
        if valids:
            r = valids[0]
            greedy_sel.append((m, r))
            
    best_score = sum(1.0/m for m, r in greedy_sel)
    best_solution = list(greedy_sel)
    
    # Run B&B
    branch_and_bound(0, 0.0, [])
    
    return [[m, r] for m, r in best_solution]
