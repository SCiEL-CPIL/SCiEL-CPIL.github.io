import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    moduli = list(range(20, 201))
    n_mods = len(moduli)
    
    # Precompute gcd matrix
    gcd_mat = [[0] * n_mods for _ in range(n_mods)]
    for i in range(n_mods):
        for j in range(n_mods):
            gcd_mat[i][j] = math.gcd(moduli[i], moduli[j])
            
    # Precompute inverse moduli for scoring
    inv_m = [1.0 / m for m in moduli]

    def is_compatible(idx_m, r, selected):
        """
        selected: list of (idx_prime, r_prime)
        """
        for idx_prime, r_prime in selected:
            g = gcd_mat[idx_m][idx_prime]
            if r % g == r_prime % g:
                return False
        return True

    def get_score(selected_indices):
        return sum(inv_m[i] for i in selected_indices)

    def greedy_fill():
        """
        Greedy fill processing moduli in increasing order of m.
        For each m, choose the residue r that maximizes the number of 
        future moduli that can still be satisfied.
        """
        selected = [] # list of (idx, r)
        selected_set = set()
        
        # Process in order of increasing m (index 0 to n_mods-1)
        for idx_m in range(n_mods):
            m = moduli[idx_m]
            valid_residues = []
            
            # Find all valid residues for this m
            for r in range(m):
                if is_compatible(idx_m, r, selected):
                    valid_residues.append(r)
                    
            if not valid_residues:
                continue
                
            # If only one, take it
            if len(valid_residues) == 1:
                r_choice = valid_residues[0]
            else:
                # Evaluate flexibility: for each candidate r, count how many 
                # unprocessed moduli idx_future > idx_m have at least one 
                # valid residue given (idx_m, r_choice) is added.
                best_r = valid_residues[0]
                best_score = -1
                
                # To speed up, we can sample or limit the check. 
                # But N is small, so full check is okay.
                for r_cand in valid_residues:
                    # Temporary state: selected + (idx_m, r_cand)
                    temp_selected = selected + [(idx_m, r_cand)]
                    
                    flexibility = 0
                    for idx_future in range(idx_m + 1, n_mods):
                        # Check if there exists ANY r_future compatible with temp_selected
                        # We don't need to find which one, just if one exists.
                        found = False
                        # Optimization: iterate residues? m_future up to 200.
                        # This inner loop is expensive if done naively.
                        # Let's approximate: just check if the set of residues 
                        # blocked by temp_selected doesn't cover all residues of m_future.
                        
                        # Blocked residues for idx_future:
                        # r_future is blocked if r_future % gcd(m_future, m_sel) == r_sel % gcd(...)
                        # for any (m_sel, r_sel) in temp_selected.
                        
                        # Instead of iterating all r_future, we can check if the union of blocked residues covers [0, m_future).
                        # This is a set cover problem, hard to do exactly fast.
                        
                        # Heuristic: Just check if r_future = 0 is compatible? No.
                        # Let's just pick the first compatible residue found for the future modulus to test existence.
                        # If we find one, it's flexible.
                        
                        # To make it faster, we can just check a few random residues or 0.
                        # Or simply: assume it's flexible unless we prove it's not.
                        # Let's do a quick check: is there a residue in 0..min(10, m_future) that works?
                        
                        compatible_found = False
                        # Check 0 first, then random few
                        test_residues = [0]
                        # Add a few randoms
                        for _ in range(5):
                            test_residues.append(random.randint(0, moduli[idx_future]-1))
                            
                        for r_test in test_residues:
                            if is_compatible(idx_future, r_test, temp_selected):
                                compatible_found = True
                                break
                                
                        if compatible_found:
                            flexibility += 1
                            
                    if flexibility > best_score:
                        best_score = flexibility
                        best_r = r_cand
                        
                r_choice = best_r
                
            selected.append((idx_m, r_choice))
            selected_set.add(idx_m)
            
        return selected, selected_set

    def local_search(selected, selected_set, max_iters=500):
        """
        Local search to improve the solution.
        Moves:
        1. Try to insert a missing modulus by adjusting residues of existing ones.
        2. Try to swap residues to enable insertion.
        """
        best_score = get_score(list(selected_set))
        best_selected = list(selected)
        best_set = set(selected_set)
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            # Identify missing moduli, sorted by weight (1/m) descending (m ascending)
            missing = [i for i in range(n_mods) if i not in best_set]
            if not missing:
                break
            
            missing.sort(key=lambda i: inv_m[i], reverse=True)
            
            # Try to insert the most valuable missing modulus
            target_idx = missing[0]
            target_m = moduli[target_idx]
            
            # Can we insert it as is?
            valid_residues = []
            for r in range(target_m):
                if is_compatible(target_idx, r, best_selected):
                    valid_residues.append(r)
                    
            if valid_residues:
                # Pick one (e.g., first or most flexible, but for local search just pick first)
                new_selected = best_selected + [(target_idx, valid_residues[0])]
                new_set = best_set | {target_idx}
                new_score = get_score(list(new_set))
                if new_score > best_score + 1e-15:
                    best_score = new_score
                    best_selected = new_selected
                    best_set = new_set
                continue
                
            # If not directly insertable, try to adjust one existing modulus's residue
            # to make room for target_idx.
            # Pick a random existing modulus to adjust
            if not best_selected:
                continue
                
            idx_to_adjust = random.choice(best_selected)[0]
            m_adj = moduli[idx_to_adjust]
            
            # Try all residues for idx_to_adjust
            for r_new in range(m_adj):
                # Check if r_new is compatible with all OTHER selected
                others = [(i, r) for i, r in best_selected if i != idx_to_adjust]
                if not is_compatible(idx_to_adjust, r_new, others):
                    continue
                
                # Now check if target_idx can be inserted with this new configuration
                temp_selected = others + [(idx_to_adjust, r_new)]
                
                # Find valid residue for target
                target_valid = False
                for r_t in range(target_m):
                    if is_compatible(target_idx, r_t, temp_selected):
                        target_valid = True
                        # Construct new solution
                        new_selected = temp_selected + [(target_idx, r_t)]
                        new_set = best_set | {target_idx}
                        # Note: idx_to_adjust was already in best_set
                        new_score = get_score(list(new_set))
                        
                        if new_score > best_score + 1e-15:
                            best_score = new_score
                            best_selected = new_selected
                            best_set = new_set
                        break
                if target_valid:
                    break # Move applied, break residue loop

        return best_selected, best_set

    # Main Loop
    best_solution = []
    best_score = -1
    
    # Run Greedy
    selected, selected_set = greedy_fill()
    score = get_score(list(selected_set))
    
    if score > best_score:
        best_score = score
        best_solution = selected
    
    # Run Local Search
    if time.time() < deadline - 10.0:
        ls_selected, ls_set = local_search(selected, selected_set, max_iters=2000)
        ls_score = get_score(list(ls_set))
        if ls_score > best_score:
            best_score = ls_score
            best_solution = ls_selected

    # Convert to required format [[m, r], ...]
    result = []
    for idx, r in best_solution:
        result.append([moduli[idx], r])
        
    return result
