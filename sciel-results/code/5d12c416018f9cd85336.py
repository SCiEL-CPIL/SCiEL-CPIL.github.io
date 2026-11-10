import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_mod = len(moduli)
    
    # Precompute gcd table
    gcd_table = [[0] * (max_mod + 1) for _ in range(max_mod + 1)]
    for i in range(2, max_mod + 1):
        for j in range(2, max_mod + 1):
            gcd_table[i][j] = gcd(i, j)
    
    # Precompute conflict info: for each pair of moduli, which residue pairs conflict
    # Two classes (r1, m1) and (r2, m2) conflict iff r1 % gcd(m1,m2) == r2 % gcd(m1,m2)
    
    def num_divisors(n):
        count = 1
        i = 2
        while i * i <= n:
            if n % i == 0:
                count += 1
                if i * i != n:
                    count += 1
            i += 1
        return count
    
    # Sort moduli by descending divisor count (more constrained first)
    moduli_desc = sorted(moduli, key=num_divisors, reverse=True)
    moduli_asc = sorted(moduli, key=num_divisors)
    
    def check_valid(selected, r, m):
        """Check if (r, m) is compatible with all in selected."""
        for r_j, m_j in selected:
            g = gcd_table[m][m_j]
            if r % g == r_j % g:
                return False
        return True
    
    def find_valid_residue(selected, m):
        """Find any valid residue for m given selected classes."""
        for r in range(m):
            if check_valid(selected, r, m):
                return r
        return None
    
    def find_all_valid_residues(selected, m):
        """Find all valid residues for m given selected classes."""
        valid = []
        for r in range(m):
            if check_valid(selected, r, m):
                valid.append(r)
        return valid
    
    def greedy(order):
        """Greedy: process moduli in order, add if possible."""
        selected = []
        selected_set = set()
        for m in order:
            if time.time() > deadline:
                break
            r = find_valid_residue(selected, m)
            if r is not None:
                selected.append((r, m))
                selected_set.add(m)
        return selected, selected_set
    
    def local_search_remove_one_add_two(selected, selected_set):
        """Try remove-one-add-two moves to escape local optima."""
        best = selected[:]
        best_set = selected_set.copy()
        best_len = len(best)
        
        # Build list of missing moduli
        missing = [m for m in moduli if m not in best_set]
        
        # Try each pair of missing moduli: can we remove one existing and add two?
        # For efficiency, try random subsets
        random.shuffle(missing)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(missing)
            
            for i in range(len(missing)):
                if time.time() > deadline:
                    break
                m1 = missing[i]
                for j in range(i+1, len(missing)):
                    if time.time() > deadline:
                        break
                    m2 = missing[j]
                    
                    # Try removing each existing modulus and adding m1, m2
                    for k in range(len(best)):
                        if time.time() > deadline:
                            break
                        
                        removed_r, removed_m = best[k]
                        # Temporarily remove
                        temp_selected = best[:k] + best[k+1:]
                        temp_set = best_set - {removed_m}
                        
                        # Try to add m1
                        r1 = find_valid_residue(temp_selected, m1)
                        if r1 is None:
                            continue
                        temp_selected2 = temp_selected + [(r1, m1)]
                        temp_set2 = temp_set | {m1}
                        
                        # Try to add m2
                        r2 = find_valid_residue(temp_selected2, m2)
                        if r2 is None:
                            continue
                        
                        # Valid move: remove 1, add 2, net +1
                        new_selected = temp_selected2 + [(r2, m2)]
                        new_set = temp_set2 | {m2}
                        
                        if len(new_selected) > best_len:
                            best = new_selected
                            best_set = new_set
                            best_len = len(best)
                            improved = True
                            # Update missing list
                            missing = [m for m in moduli if m not in best_set]
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return best, best_set
    
    def local_search_swap(selected, selected_set):
        """Try swap moves: remove one, add one different missing modulus."""
        best = selected[:]
        best_set = selected_set.copy()
        best_len = len(best)
        
        missing = [m for m in moduli if m not in best_set]
        random.shuffle(missing)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(missing)
            
            for m_new in missing:
                if time.time() > deadline:
                    break
                r_new = find_valid_residue(best, m_new)
                if r_new is None:
                    continue
                
                # Try swapping with each existing
                for k in range(len(best)):
                    if time.time() > deadline:
                        break
                    removed_r, removed_m = best[k]
                    temp_selected = best[:k] + [(r_new, m_new)] + best[k+1:]
                    temp_set = best_set - {removed_m} | {m_new}
                    
                    # This doesn't change size, but might enable further additions
                    # Only useful if it leads to improvement later; skip for now
                    pass
        
        return best, best_set
    
    def fast_greedy_fill(selected, selected_set):
        """Quickly try to add any missing moduli."""
        best = selected[:]
        best_set = selected_set.copy()
        
        missing = [m for m in moduli if m not in best_set]
        random.shuffle(missing)
        
        for m in missing:
            if time.time() > deadline:
                break
            r = find_valid_residue(best, m)
            if r is not None:
                best.append((r, m))
                best_set.add(m)
        
        return best, best_set
    
    best_solution = []
    best_set = set()
    best_len = 0
    
    strategies_done = 0
    while time.time() < deadline:
        strategies_done += 1
        
        # Use various orders
        if strategies_done % 7 == 1:
            order = moduli_desc
        elif strategies_done % 7 == 2:
            order = moduli_asc
        elif strategies_done % 7 == 3:
            order = moduli[:]
            random.shuffle(order)
        else:
            order = moduli[:]
            random.shuffle(order)
            
        sol, sol_set = greedy(order)
        
        # Fast fill
        sol, sol_set = fast_greedy_fill(sol, sol_set)
        
        # Remove-one-add-two local search
        sol, sol_set = local_search_remove_one_add_two(sol, sol_set)
        
        # Final fast fill
        sol, sol_set = fast_greedy_fill(sol, sol_set)
        
        if len(sol) > best_len:
            best_len = len(sol)
            best_solution = sol
            best_set = sol_set
            
        # Early exit if we have a very good solution
        if best_len >= 60:
            # Keep searching a bit more but with tighter budget
            pass
    
    if not best_solution:
        best_solution = [(0, 2)]
    
    return [[r, m] for r, m in best_solution]
