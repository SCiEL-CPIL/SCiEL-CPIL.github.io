import time
import sys
import random
import numpy as np

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 110
    
    # 1. Generate a pool of candidate rows using randomized greedy
    # We want a diverse set of "good" rows to choose from.
    pool = []
    # Generate rows with varying "tightness" or randomization
    # We'll generate rows one by one, ensuring they are valid internally (Sidon-like within row)
    # and keep them if they don't conflict with each other? 
    # Actually, the problem requires ALL differences across ALL rows to be distinct.
    # So we can't just have a pool of valid rows and pick 7. We need the SET of 7 to be valid.
    # However, we can use a "portfolio" approach:
    # Generate many random valid rows (individually valid, i.e., distinct diffs within row).
    # Then use a local search / backtracking to select a subset of 7 that are mutually compatible.
    # Or, better: Use the pool to seed local search.
    
    # Let's stick to the "construct then optimize" approach but improve the optimizer.
    # The previous local search was weak. Let's try a different representation or move set.
    
    # Strategy:
    # 1. Fast greedy baseline.
    # 2. Iterated Local Search (ILS):
    #    - Start with a solution.
    #    - Perturb: randomly change a few elements.
    #    - Local Search: minimize max entry using a specialized move.
    #    - If improved, accept. If not, restart from best found.
    
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_score = max(max(r) for r in best_solution)
    
    # Pre-compute some constants
    max_possible = 200 # Upper bound for search
    
    while time.time() < deadline - 10:
        # Perturb the best solution using VALID moves
        candidate = [r[:] for r in best_solution]
        
        # Perform a few random valid moves (perturbation)
        for _ in range(5):
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k) # Don't touch 0
            
            # Find a random valid new value for candidate[r_idx][pos]
            new_val = _find_random_valid_value(candidate, r_idx, pos, n_rows, k)
            if new_val is not None and new_val != candidate[r_idx][pos]:
                candidate[r_idx][pos] = new_val
            else:
                # If no valid move found, skip or try another pos
                continue
        
        # Local search to minimize scope
        improved = _local_search_minimize(candidate, n_rows, k, deadline)
        
        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
    
    return best_solution

def _find_random_valid_value(rows, r_idx, pos, n_rows, k):
    # Find a random valid value for rows[r_idx][pos]
    # that is different from the current value.
    
    row = rows[r_idx]
    old_val = row[pos]
    
    # Calculate forbidden diffs (excluding those involving old_val)
    # All diffs in the system
    all_diffs = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                all_diffs.add(r[j] - r[i])
    
    # Diffs involving old_val
    diffs_involving_old = set()
    for i in range(len(row)):
        if i == pos: continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)
            
    forbidden_diffs = all_diffs - diffs_involving_old
    
    # Search for a new value
    # Range: [row[pos-1]+1, row[pos+1]-1] (or max possible)
    lower_bound = row[pos-1] + 1
    upper_bound = 200 # Arbitrary limit
    if pos + 1 < len(row):
        upper_bound = min(upper_bound, row[pos+1] - 1)
        
    # Collect valid candidates
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue
            
        # Check validity
        valid = True
        new_diffs = set()
        for i in range(len(row)):
            if i == pos: continue
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
        
    # Pick randomly
    return random.choice(valid_candidates)

def _local_search_minimize(rows, n_rows, k, deadline):
    # Greedy local search to minimize max entry
    # We iterate and try to decrease each element.
    
    best_rows = [r[:] for r in rows]
    
    while time.time() < deadline - 5:
        progress = False
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        random.shuffle(indices)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 5:
                break
                
            row = best_rows[r_idx]
            old_val = row[pos]
            
            # Try to find the smallest valid value < old_val
            new_val = _find_smallest_valid_value(best_rows, r_idx, pos, old_val, n_rows, k)
            
            if new_val is not None and new_val < old_val:
                best_rows[r_idx][pos] = new_val
                progress = True
                
        if not progress:
            break
            
    return best_rows

def _find_smallest_valid_value(rows, r_idx, pos, old_val, n_rows, k):
    # Find the smallest valid value for rows[r_idx][pos] that is < old_val
    row = rows[r_idx]
    
    # Calculate forbidden diffs
    all_diffs = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                all_diffs.add(r[j] - r[i])
    
    diffs_involving_old = set()
    for i in range(len(row)):
        if i == pos: continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)
            
    forbidden_diffs = all_diffs - diffs_involving_old
    
    lower_bound = row[pos-1] + 1
    upper_bound = old_val - 1
    if pos + 1 < len(row):
        upper_bound = min(upper_bound, row[pos+1] - 1)
        
    if upper_bound < lower_bound:
        return None
        
    # Search for smallest valid value
    for new_val in range(lower_bound, upper_bound + 1):
        valid = True
        new_diffs = set()
        for i in range(len(row)):
            if i == pos: continue
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
            for x in range(current_max + 1, current_max + 1000):
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
    # First, repair the solution if it's invalid
    rows = _repair(rows, n_rows, k)
    if not rows:
        return None
        
    # Local search to minimize max entry
    # We use a "steepest descent" style local search with a specific move:
    # For each element, try to decrease it.
    
    # To make it faster, we maintain a set of used differences.
    # But recalculating it is O(N^2). N is small (7*6=42 elements), so O(N^2) is fine.
    
    best_rows = [r[:] for r in rows]
    best_score = max(max(r) for r in best_rows)
    
    while time.time() < deadline - 5:
        progress = False
        # Randomize order of rows and positions to escape local minima
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        random.shuffle(indices)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 5:
                break
                
            row = best_rows[r_idx]
            old_val = row[pos]
            
            # Try to reduce old_val
            # Constraints:
            # 1. row[pos-1] < new_val < row[pos+1] (if pos+1 exists)
            # 2. Differences with other elements in the row must be distinct
            # 3. Differences with elements in other rows must be distinct
            
            # Calculate forbidden diffs
            # Diffs involving row[pos] in the current solution:
            diffs_involving_pos = set()
            for i in range(len(row)):
                if i == pos: continue
                if i < pos:
                    diffs_involving_pos.add(old_val - row[i])
                else:
                    diffs_involving_pos.add(row[i] - old_val)
            
            # All other diffs in the system
            all_diffs = set()
            for r in best_rows:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d = r[j] - r[i]
                        all_diffs.add(d)
            
            # Diffs that are NOT involving row[pos]
            forbidden_diffs = all_diffs - diffs_involving_pos
            
            # Search for new_val < old_val
            # Lower bound: row[pos-1] + 1
            # Upper bound: old_val - 1 (to ensure decrease)
            # Also must be < row[pos+1] if pos+1 exists
            
            lower_bound = row[pos-1] + 1
            upper_bound = old_val - 1
            if pos + 1 < len(row):
                upper_bound = min(upper_bound, row[pos+1] - 1)
                
            if upper_bound < lower_bound:
                continue
                
            # Try values from lower_bound to upper_bound
            # We want the smallest possible value to minimize scope?
            # Actually, we just want ANY value < old_val that is valid.
            # To maximize chance of future improvements, maybe try random valid values?
            # Or just the smallest? Smallest is greedy and might get stuck.
            # Let's try the smallest valid value first. If found, accept.
            
            found = False
            for new_val in range(lower_bound, upper_bound + 1):
                # Check validity
                valid = True
                new_diffs = set()
                
                # Check diffs within this row
                for i in range(len(row)):
                    if i == pos: continue
                    if i < pos:
                        d = new_val - row[i]
                    else:
                        d = row[i] - new_val
                    
                    if d in forbidden_diffs or d in new_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # If valid, update
                # Remove old diffs, add new diffs
                # We don't need to maintain a global set if we recalculate all_diffs each time,
                # but that's O(N^2) per check.
                # Let's just update the row and accept.
                best_rows[r_idx][pos] = new_val
                progress = True
                found = True
                break
            
            # If we found a smaller value, the "forbidden_diffs" calculation for the next
            # element in this loop is now stale if we don't recalculate.
            # But since we are iterating over a shuffled list of positions, and we only
            # accept improvements, the solution remains valid.
            # The "forbidden_diffs" for the NEXT position will be recalculated at the top
            # of the loop? No, we calculated it for the CURRENT position.
            # Wait, the loop above iterates `indices`. For each index, we calculate `forbidden_diffs`
            # based on the CURRENT state of `best_rows`.
            # Since we update `best_rows` immediately, the next iteration will use the new state.
            # This is correct.
            
        if not progress:
            break
            
    return best_rows

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
