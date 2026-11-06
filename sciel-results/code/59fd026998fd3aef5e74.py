import time
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def time_up():
        return elapsed() > deadline
    
    # All candidate moduli in [20, 200]
    all_moduli = list(range(20, 201))
    
    # Precompute gcd pairs
    def get_gcd(m1, m2):
        return gcd(m1, m2)
    
    # Score function
    def score(moduli_set):
        return sum(1.0 / m for m in moduli_set)
    
    # Check if a set of moduli with given residues is valid
    def is_valid(moduli, residues):
        n = len(moduli)
        for i in range(n):
            for j in range(i + 1, n):
                g = get_gcd(moduli[i], moduli[j])
                if residues[i] % g == residues[j] % g:
                    return False
        return True
    
    # Try to find valid residues for a given set of moduli using backtracking
    def find_residues(moduli, timeout=5.0):
        start = time.time()
        n = len(moduli)
        residues = [-1] * n
        
        def backtrack(idx):
            if time.time() - start > timeout:
                return False
            if idx == n:
                return True
            
            m = moduli[idx]
            # Try residues in order of preference (smaller residues first for determinism)
            for r in range(m):
                # Check compatibility with already assigned residues
                compatible = True
                for j in range(idx):
                    g = get_gcd(moduli[idx], moduli[j])
                    if r % g == residues[j] % g:
                        compatible = False
                        break
                if compatible:
                    residues[idx] = r
                    if backtrack(idx + 1):
                        return True
                    residues[idx] = -1
            
            return False
        
        if backtrack(0):
            return residues
        return None
    
    # Greedy construction: add moduli in order of decreasing 1/m
    def greedy_construct():
        selected = []
        residues = []
        
        # Sort by 1/m descending (i.e., m ascending)
        for m in sorted(all_moduli, key=lambda x: x):
            if time_up():
                break
            # Try to find a residue for m that's compatible with all selected
            found = False
            for r in range(m):
                compatible = True
                for i, sm in enumerate(selected):
                    g = get_gcd(m, sm)
                    if r % g == residues[i] % g:
                        compatible = False
                        break
                if compatible:
                    selected.append(m)
                    residues.append(r)
                    found = True
                    break
        
        return selected, residues
    
    # Local search improvements
    def local_search(selected, residues, max_iters=10000):
        current_score = score(selected)
        best_selected = list(selected)
        best_residues = list(residues)
        best_score = current_score
        
        for iteration in range(max_iters):
            if time_up():
                break
            
            improved = False
            
            # Try adding a modulus not in selected
            selected_set = set(selected)
            candidates_to_add = [m for m in all_moduli if m not in selected_set]
            
            if candidates_to_add:
                # Try adding each candidate
                for m_add in candidates_to_add:
                    if time_up():
                        break
                    # Try to find a residue for m_add
                    found_r = None
                    for r in range(m_add):
                        compatible = True
                        for i, sm in enumerate(selected):
                            g = get_gcd(m_add, sm)
                            if r % g == residues[i] % g:
                                compatible = False
                                break
                        if compatible:
                            found_r = r
                            break
                    
                    if found_r is not None:
                        new_selected = selected + [m_add]
                        new_residues = residues + [found_r]
                        new_score = score(new_selected)
                        if new_score > best_score:
                            best_selected = new_selected
                            best_residues = new_residues
                            best_score = new_score
                            improved = True
                            break
            
            # Try swapping: remove one selected, add one not selected
            if not improved:
                for i, m_remove in enumerate(selected):
                    if time_up():
                        break
                    
                    # Try removing m_remove and adding each candidate
                    remaining = selected[:i] + selected[i+1:]
                    remaining_res = residues[:i] + residues[i+1:]
                    remaining_set = set(remaining)
                    
                    candidates = [m for m in all_moduli if m not in remaining_set]
                    
                    for m_add in candidates:
                        if time_up():
                            break
                        # Try to find a residue for m_add compatible with remaining
                        found_r = None
                        for r in range(m_add):
                            compatible = True
                            for j, sm in enumerate(remaining):
                                g = get_gcd(m_add, sm)
                                if r % g == remaining_res[j] % g:
                                    compatible = False
                                    break
                            if compatible:
                                found_r = r
                                break
                        
                        if found_r is not None:
                            new_selected = remaining + [m_add]
                            new_residues = remaining_res + [found_r]
                            new_score = score(new_selected)
                            if new_score > best_score:
                                best_selected = new_selected
                                best_residues = new_residues
                                best_score = new_score
                                improved = True
                                break
                    
                    if improved:
                        break
            
            if not improved:
                # No improvement, try reassigning residues to enable additions
                # Try random reassignment of one residue
                if selected:
                    idx = random.randint(0, len(selected) - 1)
                    m = selected[idx]
                    old_r = residues[idx]
                    
                    # Try all other residues for this modulus
                    for r in range(m):
                        if r == old_r:
                            continue
                        new_res = residues[:]
                        new_res[idx] = r
                        if is_valid(selected, new_res):
                            # Check if this enables adding a new modulus
                            selected_set = set(selected)
                            for m_add in all_moduli:
                                if m_add in selected_set:
                                    continue
                                found_r2 = None
                                for r2 in range(m_add):
                                    compatible = True
                                    for j, sm in enumerate(selected):
                                        g = get_gcd(m_add, sm)
                                        if r2 % g == new_res[j] % g:
                                            compatible = False
                                            break
                                    if compatible:
                                        found_r2 = r2
                                        break
                                if found_r2 is not None:
                                    new_selected = selected + [m_add]
                                    new_residues = new_res + [found_r2]
                                    new_score = score(new_selected)
                                    if new_score > best_score:
                                        best_selected = new_selected
                                        best_residues = new_residues
                                        best_score = new_score
                                        improved = True
                                        break
                        if improved:
                            break
                    
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_selected, best_residues, best_score
    
    # Initial greedy construction
    selected, residues = greedy_construct()
    
    if not selected:
        # Fallback: return the seed solution
        return [[20 * j, j - 1] for j in range(1, 11)]
    
    # Local search
    selected, residues, best_score = local_search(selected, residues)
    
    # Final validation
    if not is_valid(selected, residues):
        # Fallback
        return [[20 * j, j - 1] for j in range(1, 11)]
    
    return [[m, r] for m, r in zip(selected, residues)]
