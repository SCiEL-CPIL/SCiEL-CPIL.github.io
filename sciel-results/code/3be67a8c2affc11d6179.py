import math
import time
import random
import sys

def construct():
    deadline = time.time() + 140  # 150s wall-clock limit, keep 10s margin
    
    max_m = 200
    min_m = 20
    
    # ---------- Precomputation ----------
    # gcd table
    gcd_table = [[0] * (max_m + 1) for _ in range(max_m + 1)]
    for i in range(min_m, max_m + 1):
        for j in range(min_m, max_m + 1):
            gcd_table[i][j] = math.gcd(i, j)

    # List of all moduli
    moduli = list(range(min_m, max_m + 1))
    
    # Precompute 1/m for sorting
    weights = {m: 1.0 / m for m in moduli}

    def is_valid(m, r, selected, skip_idx=None):
        """Check if (m, r) is valid against selected classes."""
        for idx in range(len(selected)):
            if skip_idx is not None and idx == skip_idx:
                continue
            m2, r2 = selected[idx]
            g = gcd_table[m][m2]
            if r % g == r2 % g:
                return False
        return True

    def get_valid_residues(m, selected):
        """Return list of valid residues for m given selected classes."""
        valid = []
        for r in range(m):
            if is_valid(m, r, selected):
                valid.append(r)
        return valid

    def calc_score(selected):
        return sum(1.0 / m for m, _ in selected)

    def randomized_greedy(deadline):
        """
        Build a solution using randomized greedy.
        We shuffle the order of moduli to explore different structures.
        """
        selected = []
        # Use a random permutation of moduli
        mods = moduli[:]
        random.shuffle(mods)
        
        for m in mods:
            if time.time() > deadline:
                break
            # Try a few random residues first, then all if needed?
            # For speed, just pick the first valid one in a random order
            residues = list(range(m))
            random.shuffle(residues)
            for r in residues:
                if is_valid(m, r, selected):
                    selected.append([m, r])
                    break
        return selected

    def local_search_polish(selected, deadline):
        """
        Intensive local search.
        1. Try to add missing moduli.
        2. Try to change residues to enable adding high-value moduli.
        3. Try to swap low-value moduli for high-value ones.
        """
        current_sol = [list(x) for x in selected]
        current_score = calc_score(current_sol)
        improved = True
        no_improve_count = 0
        
        while improved and time.time() < deadline:
            improved = False
            no_improve_count += 1
            if no_improve_count > 5:
                break # Heuristic: if stuck, stop
            
            # 1. Try adding any missing modulus (smallest m first for max gain)
            added_any = False
            for m in moduli:
                if time.time() > deadline:
                    return current_sol, current_score
                if any(m == m_sel for m_sel, _ in current_sol):
                    continue
                valid_res = get_valid_residues(m, current_sol)
                if valid_res:
                    # Pick the first one (smallest r) for determinism in polish, 
                    # or random? Deterministic is safer for local optima stability.
                    r = valid_res[0]
                    current_sol.append([m, r])
                    current_score += 1.0 / m
                    improved = True
                    added_any = True
                    break # Restart loop to check for new additions
            
            if added_any:
                continue

            # 2. Try to "unlock" a missing high-value modulus by changing a residue
            # Identify missing moduli, sorted by value desc
            selected_mods = set(m for m, _ in current_sol)
            missing_mods = [m for m in moduli if m not in selected_mods]
            missing_mods.sort(key=lambda x: -1.0/x) # Smallest m first (highest value)
            
            unlocked = False
            for m_new in missing_mods[:10]: # Only try to unlock top 10 missing
                if time.time() > deadline:
                    return current_sol, current_score
                
                # For each existing class, try changing its residue to allow m_new
                for idx in range(len(current_sol)):
                    m_chg, old_r = current_sol[idx]
                    
                    # Get valid residues for m_chg if it were not in the set
                    # i.e., valid against current_sol excluding idx
                    others = [current_sol[j] for j in range(len(current_sol)) if j != idx]
                    valid_res_chg = get_valid_residues(m_chg, others)
                    
                    for new_r in valid_res_chg:
                        if new_r == old_r:
                            continue
                        
                        # Temporarily swap
                        current_sol[idx] = [m_chg, new_r]
                        
                        # Check if m_new can now be added
                        valid_res_new = get_valid_residues(m_new, current_sol)
                        if valid_res_new:
                            # Success! Keep the swap and add m_new
                            r_new = valid_res_new[0]
                            current_sol.append([m_new, r_new])
                            current_score += 1.0 / m_new
                            improved = True
                            unlocked = True
                            break
                        else:
                            # Revert
                            current_sol[idx] = [m_chg, old_r]
                    
                    if unlocked:
                        break
                
                if unlocked:
                    break

            if unlocked:
                continue

            # 3. Try swapping a low-value class for a high-value missing one
            # Sort current_sol by value asc (largest m first, lowest value)
            indices_by_value = sorted(range(len(current_sol)), key=lambda i: current_sol[i][0])
            
            swapped = False
            for idx_old in indices_by_value:
                if time.time() > deadline:
                    return current_sol, current_score
                
                m_old, r_old = current_sol[idx_old]
                
                # Try to remove m_old and add any missing m_new with 1/m_new > 1/m_old
                for m_new in missing_mods:
                    if 1.0 / m_new <= 1.0 / m_old:
                        break # No more better moduli
                    
                    # Check if m_new can be added to (current_sol \ {m_old})
                    others = [current_sol[j] for j in range(len(current_sol)) if j != idx_old]
                    valid_res_new = get_valid_residues(m_new, others)
                    if valid_res_new:
                        # Perform swap
                        r_new = valid_res_new[0]
                        current_sol[idx_old] = [m_new, r_new]
                        current_score += 1.0 / m_new - 1.0 / m_old
                        improved = True
                        swapped = True
                        break
                
                if swapped:
                    break
            
            # Reset no_improve counter if we made any move
            if improved:
                no_improve_count = 0

        return current_sol, current_score

    # ---------- Main Search Loop ----------
    best_solution = []
    best_score = 0.0
    
    # Run multiple restarts with randomized greedy + local search
    # Allocate time for restarts
    restart_time_budget = deadline - 10 # Leave 10s for final polish
    
    num_restarts = 0
    while time.time() < restart_time_budget:
        num_restarts += 1
        
        # Generate initial solution
        init_sol = randomized_greedy(deadline - 5)
        
        # Polish
        polished_sol, score = local_search_polish(init_sol, deadline - 5)
        
        if score > best_score:
            best_score = score
            best_solution = polished_sol
        
        # If we have a very good score, we might spend less time on subsequent restarts?
        # For now, just keep going until time runs out.
        
        # Safety check: if we've done enough restarts and score is high, break?
        # No, let it run to fill time.

    # Final polish on the best solution found
    if time.time() < deadline - 2:
        final_sol, final_score = local_search_polish(best_solution, deadline - 2)
        if final_score > best_score:
            best_score = final_score
            best_solution = final_sol

    # Ensure return format
    return [[m, r] for m, r in best_solution]

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = sum(1.0 / m for m, _ in res)
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
    valid = True
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            m1, r1 = res[i]
            m2, r2 = res[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                valid = False
                print(f"Invalid: {res[i]} and {res[j]}")
    print(f"Valid: {valid}")
