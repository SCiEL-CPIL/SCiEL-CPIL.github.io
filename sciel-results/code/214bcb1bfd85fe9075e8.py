import time
import sys
import random
import numpy as np

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 110
    
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_score = max(max(r) for r in best_solution)
    
    while time.time() < deadline - 5:
        # Perturb the best solution using VALID moves
        candidate = [r[:] for r in best_solution]
        
        # Perform a few random valid moves (perturbation)
        n_perturb = random.randint(2, 6)
        for _ in range(n_perturb):
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            
            new_val = _find_random_valid_value_fast(candidate, r_idx, pos, n_rows, k)
            if new_val is not None and new_val != candidate[r_idx][pos]:
                candidate[r_idx][pos] = new_val
        
        # Local search to minimize scope
        improved = _local_search_minimize_fast(candidate, n_rows, k, deadline)
        
        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
    
    # Final polish: try to squeeze out more improvement
    if time.time() < deadline - 2:
        final = _local_search_minimize_fast([r[:] for r in best_solution], n_rows, k, deadline)
        if final:
            final_score = max(max(r) for r in final)
            if final_score < best_score:
                best_solution = final
                best_score = final_score
    
    return best_solution

def _find_random_valid_value_fast(rows, r_idx, pos, n_rows, k):
    # Find a random valid value for rows[r_idx][pos] using precomputed diff sets
    row = rows[r_idx]
    old_val = row[pos]
    n = len(row)
    
    # Compute all diffs in the system
    all_diffs = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                all_diffs.add(r[j] - r[i])
    
    # Diffs involving old_val at position pos
    diffs_involving_old = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)
    
    forbidden_diffs = all_diffs - diffs_involving_old
    
    # Determine search range
    lower_bound = row[pos-1] + 1
    upper_bound = 300
    if pos + 1 < n:
        upper_bound = min(upper_bound, row[pos+1] - 1)
    
    # Collect valid candidates
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue
        
        valid = True
        new_diffs = set()
        for i in range(n):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val
            
            if d in forbidden_diffs or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
        
        if valid:
            valid_candidates.append(new_val)
    
    if not valid_candidates:
        return None
    
    return random.choice(valid_candidates)

def _local_search_minimize_fast(rows, n_rows, k, deadline):
    # Fast local search with incremental diff tracking
    best_rows = [r[:] for r in rows]
    n = k + 1
    
    # Build initial diff set
    all_diffs = set()
    for r in best_rows:
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.add(r[j] - r[i])
    
    while time.time() < deadline - 3:
        progress = False
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        random.shuffle(indices)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 3:
                break
                
            row = best_rows[r_idx]
            old_val = row[pos]
            
            # Compute diffs involving (r_idx, pos)
            diffs_involving = set()
            for i in range(n):
                if i == pos:
                    continue
                if i < pos:
                    diffs_involving.add(old_val - row[i])
                else:
                    diffs_involving.add(row[i] - old_val)
            
            forbidden_diffs = all_diffs - diffs_involving
            
            # Try to find smaller valid value
            lower_bound = row[pos-1] + 1
            upper_bound = old_val - 1
            if pos + 1 < n:
                upper_bound = min(upper_bound, row[pos+1] - 1)
            
            if upper_bound >= lower_bound:
                found = False
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val
                        
                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        # Update: remove old diffs, add new diffs
                        all_diffs.difference_update(diffs_involving)
                        all_diffs.update(new_diffs)
                        best_rows[r_idx][pos] = new_val
                        progress = True
                        found = True
                        break
                
                if found:
                    continue
            
            # Also try swap moves: swap two elements within the same row
            # or swap values between different rows at same position
            if random.random() < 0.3:
                # Try swapping with another position in the same row
                other_pos = random.randint(1, k)
                if other_pos != pos:
                    other_row = best_rows[r_idx]
                    v1 = other_row[pos]
                    v2 = other_row[other_pos]
                    
                    if v1 != v2 and v1 > other_row[pos-1] and v2 > other_row[other_pos-1]:
                        if pos + 1 < n and v1 < other_row[pos+1] and other_pos + 1 < n and v2 < other_row[other_pos+1]:
                            # Check if swap is valid
                            # This is complex; skip for now and rely on single-position moves
                            pass
    
        if not progress:
            break
            
    return best_rows

def _find_smallest_valid_value(rows, r_idx, pos, old_val, n_rows, k):
    # Kept for compatibility but not used in the fast path
    row = rows[r_idx]
    n = len(row)
    
    all_diffs = set()
    for r in rows:
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.add(r[j] - r[i])
    
    diffs_involving_old = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)
    
    forbidden_diffs = all_diffs - diffs_involving_old
    
    lower_bound = row[pos-1] + 1
    upper_bound = old_val - 1
    if pos + 1 < n:
        upper_bound = min(upper_bound, row[pos+1] - 1)
    
    if upper_bound < lower_bound:
        return None
    
    for new_val in range(lower_bound, upper_bound + 1):
        valid = True
        new_diffs = set()
        for i in range(n):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val
            
            if d in forbidden_diffs or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
        
        if valid:
            return new_val
    
    return None

def _greedy_construct(n_rows, k):
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Try values starting from current_max + 1
            for x in range(current_max + 1, current_max + 500):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                # Try a wider search with some randomization
                for _retry in range(50):
                    x = current_max + 1 + random.randint(0, 100)
                    new_diffs = []
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        row.append(x)
                        used_diffs.update(new_diffs)
                        current_max = x
                        found = True
                        break
                if not found:
                    return None
        rows.append(row)
    return rows

def _local_search_ilS(rows, n_rows, k, deadline):
    # Deprecated - use _local_search_minimize_fast instead
    return _local_search_minimize_fast(rows, n_rows, k, deadline)

def _repair(rows, n_rows, k):
    # If the input rows are invalid (due to perturbation), make them valid.
    # Strategy: Re-generate the problematic rows greedily, trying to fit them into the existing diff set.
    # Or, simpler: Re-run greedy construction from scratch if repair is too hard?
    # Let's try a simple repair:
    # 1. Check if valid. If yes, return.
    # 2. If no, find the row/pos causing conflict.
    # 3. Change that value to a valid one (greedy search for next valid).
    # 4. Repeat.
    
    # This might be slow or fail. Let's just return None if invalid?
    # No, that wastes the ILS restart.
    # Let's try to fix by shifting values.
    
    # Actually, for ILS, if perturbation breaks validity, we can just revert to the best valid solution
    # and perturb again. Or, we can try to repair.
    # Given the time constraint, let's assume perturbation is small enough that we can repair by
    # finding the next valid value for the changed element.
    
    # For now, let's check validity. If invalid, we try to fix the specific changed elements.
    # But we don't know which were changed.
    
    # Alternative: The ILS loop in `construct` should only perturb if the result is likely valid?
    # Or, we can do a "soft" local search that allows invalid states with a penalty?
    # That's more complex.
    
    # Let's stick to: If invalid, try to repair by re-greedy-filling the specific rows that might be broken.
    # Which rows? We don't know.
    
    # Let's simplify: If invalid, return None. The caller will restart from best_solution.
    # This means ILS only works if perturbation keeps it valid? Unlikely.
    
    # Better ILS strategy:
    # 1. Start with valid solution.
    # 2. Pick a random row and position.
    # 3. Change value to a RANDOM valid value (search for it).
    # 4. This ensures validity is maintained!
    # 5. Then do local search to minimize.
    
    # Let's implement `_local_search_ilS` to assume input is valid.
    # And in `construct`, we will perform the perturbation such that it maintains validity.
    
    # Wait, the current `construct` code does:
    # candidate = [r[:] for r in best_solution]
    # ... perturb ...
    # improved = _local_search_ilS(candidate, ...)
    
    # So `_local_search_ilS` receives a potentially invalid `candidate`.
    # Let's change the strategy in `construct` to do "Valid Perturbation".
    
    # But the function signature is fixed.
    # Let's just make `_local_search_ilS` robust.
    # If invalid, try to fix by replacing invalid elements with greedy next-valid.
    
    # For now, let's just check validity. If invalid, return None.
    # And in `construct`, we will modify the perturbation to be "valid".
    
    # How to do valid perturbation?
    # Pick row r, pos p.
    # Find a new value v' such that the row remains valid and global diffs remain valid.
    # This is the same as the local search move!
    # So, "Perturbation" = "Random Valid Move".
    # "Local Search" = "Greedy Valid Move to minimize".
    
    # This is a solid ILS.
    
    # Let's implement `_find_valid_move` that returns a new value for (r, p) that is valid.
    
    return None # Placeholder, we will change strategy in construct
