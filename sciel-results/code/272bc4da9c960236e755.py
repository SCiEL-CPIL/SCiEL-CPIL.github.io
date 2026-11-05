import math
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 115  # Safety margin before 150s kill
    
    # Precompute gcd values for all pairs of moduli
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    gcd_table = {}
    for i in range(n_mod):
        for j in range(i+1, n_mod):
            gcd_table[(moduli[i], moduli[j])] = math.gcd(moduli[i], moduli[j])
    
    # For each modulus m, precompute which residues are valid given a set of already-chosen (m2, r2)
    # We'll use a bitset approach for speed
    
    # Strategy: Branch and bound with greedy upper bound
    # Upper bound: sum of 1/m for all remaining moduli that could potentially be added
    # (ignoring conflicts) — this is a valid upper bound
    
    # We process moduli in decreasing order of 1/m (i.e., increasing m) for better pruning
    # Actually, for branch and bound, we want to commit to high-value items first
    # So process in order of decreasing 1/m = increasing m
    
    # Let's use a recursive branch and bound
    # State: current set of selected (m, r) pairs, next modulus to consider
    # We'll iterate over moduli in a fixed order and for each, decide which residue to pick (or skip)
    
    # For efficiency, precompute for each modulus m and each possible residue r,
    # which other moduli m2 have gcd(m, m2) = g, and r % g
    
    # Better approach: process moduli one at a time in a fixed order.
    # For modulus m, the valid residues depend on all previously chosen (m2, r2).
    # We can maintain a "forbidden" array for each modulus m: forbidden[m][r] = True if r is forbidden
    
    # Let's use a different approach: iterative deepening with memoization-like pruning
    
    # Actually, let's do a proper branch and bound:
    # Order moduli by decreasing 1/m (smallest m first)
    # For each modulus, try all valid residues, or skip it
    # Upper bound at node = current_score + sum(1/m for remaining moduli)
    
    # Precompute weights
    weights = {m: Fraction(1, m) for m in moduli}
    
    # Process moduli in order of decreasing weight (increasing m)
    # Wait, 1/m is larger for smaller m, so process smaller m first
    ordered_moduli = sorted(moduli, key=lambda m: -1.0/m)  # decreasing 1/m
    
    best_score = Fraction(0, 1)
    best_solution = []
    
    # Precompute for each pair (m1, m2) the gcd
    # And for each m, list of (m2, g) for all m2 != m
    
    # For speed, precompute conflict structure
    # conflict[m][r][m2] = r2 % gcd(m, m2) is the forbidden residue class for m2
    # Actually, it's simpler to just check on the fly with precomputed gcds
    
    # Let's precompute gcd for all pairs
    gcd_cache = {}
    for m1 in moduli:
        for m2 in moduli:
            if m1 != m2:
                gcd_cache[(m1, m2)] = math.gcd(m1, m2)
    
    # Branch and bound
    # selected: list of (m, r) chosen so far
    # idx: index in ordered_moduli of next modulus to consider
    # current_score: Fraction
    
    def upper_bound(idx, current_score):
        """Upper bound: current_score + sum of 1/m for all moduli from idx onwards"""
        ub = current_score
        for i in range(idx, len(ordered_moduli)):
            ub += weights[ordered_moduli[i]]
        return ub
    
    def is_compatible(m_new, r_new, selected):
        """Check if (m_new, r_new) is compatible with all in selected"""
        for (m2, r2) in selected:
            g = gcd_cache.get((m_new, m2), math.gcd(m_new, m2))
            if r_new % g == r2 % g:
                return False
        return True
    
    def branch_and_bound(idx, selected, current_score):
        nonlocal best_score, best_solution
        
        if time.time() > deadline:
            return
        
        if idx == len(ordered_moduli):
            if current_score > best_score:
                best_score = current_score
                best_solution = list(selected)
            return
        
        # Upper bound check
        ub = upper_bound(idx, current_score)
        if ub <= best_score:
            return
        
        m = ordered_moduli[idx]
        
        # Try all residues for m
        for r in range(m):
            if is_compatible(m, r, selected):
                selected.append((m, r))
                branch_and_bound(idx + 1, selected, current_score + weights[m])
                selected.pop()
        
        # Also try skipping m
        branch_and_bound(idx + 1, selected, current_score)
    
    # This might be too slow for full branch and bound
    # Let's use a greedy initialization first, then local search
    
    # Greedy: process moduli in order of decreasing 1/m, pick first valid residue
    selected_greedy = []
    for m in ordered_moduli:
        for r in range(m):
            if is_compatible(m, r, selected_greedy):
                selected_greedy.append((m, r))
                break
    
    greedy_score = sum(weights[m] for (m, r) in selected_greedy)
    best_score = greedy_score
    best_solution = list(selected_greedy)
    
    # Now do local search: try to improve by swapping residues or adding/removing moduli
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try changing residue of each selected modulus
        for i in range(len(best_solution)):
            m_i, r_i = best_solution[i]
            saved = best_solution.pop(i)
            
            for r in range(m_i):
                if is_compatible(m_i, r, best_solution):
                    if r != r_i:
                        best_solution.insert(i, (m_i, r))
                        improved = True
                    else:
                        best_solution.insert(i, saved)
                    break
            else:
                best_solution.insert(i, saved)
        
        # Try adding a modulus that was skipped
        selected_mods = set(m for (m, r) in best_solution)
        for m in ordered_moduli:
            if m in selected_mods:
                continue
            for r in range(m):
                if is_compatible(m, r, best_solution):
                    best_solution.append((m, r))
                    improved = True
                    break
    
    return [[m, r] for (m, r) in best_solution]
