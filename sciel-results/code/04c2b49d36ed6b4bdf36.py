"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s limit
    
    moduli = list(range(20, 201))
    
    def is_compatible(m_new, r_new, selected):
        """Check if (m_new, r_new) is disjoint from all in selected."""
        for m_old, r_old in selected:
            g = math.gcd(m_new, m_old)
            if (r_new - r_old) % g == 0:
                return False
        return True
    
    def score_solution(selected):
        return sum(1.0 / m for m, r in selected)
    
    def try_add_modulus(selected, m):
        """Try to add modulus m by finding a valid residue."""
        for r in range(m):
            if is_compatible(m, r, selected):
                return r
        return None
    
    def greedy_initial(strategy=0, seed=0):
        """Greedy: pick moduli in order, choose residue based on strategy."""
        rng = random.Random(seed)
        selected = []
        
        if strategy == 0:
            # Smallest first, first valid residue
            order = moduli[:]
        elif strategy == 1:
            # Smallest first, random valid residue
            order = moduli[:]
        elif strategy == 2:
            # Random order, first valid residue
            order = moduli[:]
            rng.shuffle(order)
        else:
            # Random order, random valid residue
            order = moduli[:]
            rng.shuffle(order)
        
        for m in order:
            if time.time() > deadline:
                break
            valid_r = []
            for r in range(m):
                if is_compatible(m, r, selected):
                    valid_r.append(r)
            if valid_r:
                if strategy in (1, 3):
                    r = rng.choice(valid_r)
                else:
                    r = valid_r[0]
                selected.append([m, r])
        
        return selected
    
    def local_search(selected, max_time=None):
        """
        Enhanced local search with multiple improvement strategies:
        1. Adding missing moduli (smallest first for max gain)
        2. Single residue reassignment to free space
        3. Pairwise residue reassignment (coordinated changes)
        4. Remove-and-retry: remove a large modulus, try to add smaller ones
        """
        if max_time is None:
            max_time = deadline
        selected = [list(x) for x in selected]
        current_score = score_solution(selected)
        selected_set = set(m for m, r in selected)
        improved = True
        iter_count = 0
        
        while improved and time.time() < max_time:
            improved = False
            iter_count += 1
            
            # 1. Try to add any missing modulus (smallest first for max 1/m gain)
            for m in sorted(moduli, key=lambda x: x):  # ascending = smallest first
                if m in selected_set:
                    continue
                if time.time() > max_time:
                    return selected
                r = try_add_modulus(selected, m)
                if r is not None:
                    selected.append([m, r])
                    selected_set.add(m)
                    current_score = score_solution(selected)
                    improved = True
                    break
            
            # 2. If can't add directly, try changing residues to free up space
            if not improved:
                missing = [m for m in moduli if m not in selected_set]
                if not missing:
                    break
                
                # Try to unblock the smallest missing modulus
                for m_add in sorted(missing, key=lambda x: x):
                    if time.time() > max_time:
                        return selected
                    
                    # Strategy A: Single residue change
                    found_change = False
                    for idx in range(len(selected)):
                        if time.time() > max_time:
                            return selected
                        m_old, r_old = selected[idx]
                        
                        for r_new in range(m_old):
                            if r_new == r_old:
                                continue
                            compatible = True
                            for jdx, (m_j, r_j) in enumerate(selected):
                                if jdx == idx:
                                    continue
                                g = math.gcd(m_old, m_j)
                                if (r_new - r_j) % g == 0:
                                    compatible = False
                                    break
                            if not compatible:
                                continue
                            
                            old_r = selected[idx][1]
                            selected[idx][1] = r_new
                            
                            r_add = try_add_modulus(selected, m_add)
                            if r_add is not None:
                                selected.append([m_add, r_add])
                                selected_set.add(m_add)
                                new_score = score_solution(selected)
                                if new_score > current_score:
                                    current_score = new_score
                                    improved = True
                                    found_change = True
                                    break
                                else:
                                    selected.pop()
                                    selected_set.discard(m_add)
                            selected[idx][1] = old_r
                        
                        if found_change:
                            break
                    
                    # Strategy B: Pairwise residue changes (if single didn't work)
                    if not found_change and time.time() < max_time - 5.0:
                        for idx1 in range(len(selected)):
                            if found_change or time.time() > max_time:
                                break
                            m1, r1_old = selected[idx1]
                            for idx2 in range(idx1 + 1, len(selected)):
                                if found_change or time.time() > max_time:
                                    break
                                m2, r2_old = selected[idx2]
                                
                                # Try changing both residues
                                for r1_new in range(m1):
                                    if r1_new == r1_old:
                                        continue
                                    # Check r1_new compatible with all except idx2
                                    compat1 = True
                                    for jdx, (m_j, r_j) in enumerate(selected):
                                        if jdx == idx1 or jdx == idx2:
                                            continue
                                        g = math.gcd(m1, m_j)
                                        if (r1_new - r_j) % g == 0:
                                            compat1 = False
                                            break
                                    if not compat1:
                                        continue
                                    
                                    for r2_new in range(m2):
                                        if r2_new == r2_old:
                                            continue
                                        # Check r2_new compatible with all except idx1
                                        compat2 = True
                                        for jdx, (m_j, r_j) in enumerate(selected):
                                            if jdx == idx1 or jdx == idx2:
                                                continue
                                            g = math.gcd(m2, m_j)
                                            if (r2_new - r_j) % g == 0:
                                                compat2 = False
                                                break
                                        if not compat2:
                                            continue
                                        
                                        # Check r1_new and r2_new compatible with each other
                                        g12 = math.gcd(m1, m2)
                                        if (r1_new - r2_new) % g12 == 0:
                                            continue
                                        
                                        # Apply both changes
                                        old_r1 = selected[idx1][1]
                                        old_r2 = selected[idx2][1]
                                        selected[idx1][1] = r1_new
                                        selected[idx2][1] = r2_new
                                        
                                        r_add = try_add_modulus(selected, m_add)
                                        if r_add is not None:
                                            selected.append([m_add, r_add])
                                            selected_set.add(m_add)
                                            new_score = score_solution(selected)
                                            if new_score > current_score:
                                                current_score = new_score
                                                improved = True
                                                found_change = True
                                                break
                                            else:
                                                selected.pop()
                                                selected_set.discard(m_add)
                                        
                                        # Revert
                                        selected[idx1][1] = old_r1
                                        selected[idx2][1] = old_r2
                                        
                                        if found_change:
                                            break
                                    
                                    if found_change:
                                        break
                    
                    if found_change:
                        break
            
            # 3. Remove-and-retry: if stuck, try removing a large modulus and adding smaller ones
            if not improved and time.time() < max_time - 10.0:
                # Try removing the largest modulus and see if we can add multiple smaller ones
                if len(selected) > 5:
                    max_idx = max(range(len(selected)), key=lambda i: selected[i][0])
                    m_remove = selected[max_idx][0]
                    
                    others = [selected[i] for i in range(len(selected)) if i != max_idx]
                    
                    # Try to add as many small moduli as possible after removal
                    new_selected = [list(x) for x in others]
                    new_set = set(m for m, r in new_selected)
                    added_any = False
                    
                    for m in sorted(moduli):
                        if m in new_set or m == m_remove:
                            continue
                        r = try_add_modulus(new_selected, m)
                        if r is not None:
                            new_selected.append([m, r])
                            new_set.add(m)
                            added_any = True
                    
                    if added_any:
                        new_score = score_solution(new_selected)
                        if new_score > current_score:
                            selected = new_selected
                            selected_set = new_set
                            current_score = new_score
                            improved = True
        
        return selected
    
    def final_polish(selected, max_time=None):
        """
        High-precision polish: exhaustive search over residue assignments
        for a subset of moduli to find optimal configuration.
        """
        if max_time is None:
            max_time = deadline
        selected = [list(x) for x in selected]
        current_score = score_solution(selected)
        selected_set = set(m for m, r in selected)
        
        # Try to find a better configuration by reassigning residues
        # for all moduli simultaneously using a smarter approach
        
        # First, try to add any remaining moduli
        for m in sorted(moduli):
            if m in selected_set:
                continue
            if time.time() > max_time:
                break
            r = try_add_modulus(selected, m)
            if r is not None:
                selected.append([m, r])
                selected_set.add(m)
        
        # Now try to improve by reassigning residues
        # Use a greedy approach: for each modulus, try all residues and keep best
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for idx in range(len(selected)):
                if time.time() > max_time:
                    break
                m, r_old = selected[idx]
                best_r = r_old
                best_score = current_score
                
                for r_new in range(m):
                    if r_new == r_old:
                        continue
                    # Check compatibility
                    compatible = True
                    for jdx, (m_j, r_j) in enumerate(selected):
                        if jdx == idx:
                            continue
                        g = math.gcd(m, m_j)
                        if (r_new - r_j) % g == 0:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    
                    # Temporarily change and see if we can add more moduli
                    old_r = selected[idx][1]
                    selected[idx][1] = r_new
                    
                    # Try to add any missing moduli
                    temp_selected = [list(x) for x in selected]
                    temp_set = set(m for m, r in temp_selected)
                    temp_score = score_solution(temp_selected)
                    
                    for m_add in sorted(moduli):
                        if m_add in temp_set:
                            continue
                        r_add = try_add_modulus(temp_selected, m_add)
                        if r_add is not None:
                            temp_selected.append([m_add, r_add])
                            temp_set.add(m_add)
                            temp_score = score_solution(temp_selected)
                    
                    if temp_score > best_score:
                        best_score = temp_score
                        best_r = r_new
                        # Keep this change
                        selected = temp_selected
                        selected_set = temp_set
                        current_score = temp_score
                        improved = True
                        break
                    else:
                        # Revert
                        selected[idx][1] = old_r
                
                if improved:
                    break
        
        return selected
    
    def swap_improve(selected, max_time=None):
        """Try swapping a large modulus for a smaller one."""
        if max_time is None:
            max_time = deadline
        selected = [list(x) for x in selected]
        current_score = score_solution(selected)
        selected_set = set(m for m, r in selected)
        improved = True
        
        while improved and time.time() < max_time:
            improved = False
            
            # Find the largest modulus in our selection
            if not selected:
                break
            max_idx = max(range(len(selected)), key=lambda i: selected[i][0])
            m_remove = selected[max_idx][0]
            
            # Try to find a smaller unused modulus that can replace it
            for m_new in sorted(moduli, key=lambda x: x):
                if m_new in selected_set or m_new >= m_remove:
                    continue
                if time.time() > max_time:
                    return selected
                
                # Check if we can add m_new after removing m_remove
                others = [selected[i] for i in range(len(selected)) if i != max_idx]
                r_new = try_add_modulus(others, m_new)
                if r_new is not None:
                    new_selected = others + [[m_new, r_new]]
                    new_score = score_solution(new_selected)
                    if new_score > current_score:
                        selected = new_selected
                        selected_set = set(m for m, r in selected)
                        current_score = new_score
                        improved = True
                        break
        
        return selected
    
    # Run multiple strategies
    best_solution = []
    best_score = 0.0
    
    strategies = [
        (0, 0), (0, 1), (0, 42),
        (1, 0), (1, 1), (1, 42),
        (2, 0), (2, 1), (2, 42),
        (3, 0), (3, 1), (3, 42),
    ]
    
    for strategy, seed in strategies:
        if time.time() > deadline - 15.0:
            break
        
        sol = greedy_initial(strategy, seed)
        sc = score_solution(sol)
        
        # Apply enhanced local search
        if time.time() < deadline - 15.0:
            sol = local_search(sol, max_time=deadline - 12.0)
            sc = score_solution(sol)
        
        # Apply swap improvement
        if time.time() < deadline - 10.0:
            sol = swap_improve(sol, max_time=deadline - 8.0)
            sc = score_solution(sol)
        
        if sc > best_score:
            best_score = sc
            best_solution = sol
    
    # Final polish on best solution
    if best_solution and time.time() < deadline - 3.0:
        best_solution = local_search(best_solution, max_time=deadline - 1.0)
        best_score = score_solution(best_solution)
    
    # Ensure we have a valid solution
    if not best_solution:
        best_solution = [[20, 0]]
    
    return best_solution
# EVOLVE-BLOCK-END
