import random
import time
import math
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Total number of differences that must be distinct
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Simple lower bound: need at least total_diffs distinct positive integers
    lower_bound = total_diffs

    # Helper to get max entry from rows
    def get_max_entry(rows):
        return max(max(row) for row in rows) if rows else 0

    # Helper to check if all differences are distinct
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    # Greedy construction: build rows one by one, placing smallest possible values
    def pure_greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    x += 1
                else:
                    x += 1
            rows.append(row)
        return rows

    # VNS Strategy:
    # Represent the state as a list of rows.
    # Objective: Minimize the maximum entry.
    # Constraint: All differences are distinct.
    
    # Since the constraint is hard, we must maintain validity.
    # We use a "repair" or "local move" approach where we only make valid moves.
    # However, VNS typically works on a continuous or discrete space with a cost function.
    # Here, the space is discrete and constrained.
    
    # Alternative VNS formulation:
    # 1. Start with a valid solution (greedy).
    # 2. Define neighborhoods:
    #    N1: Move a single element to a smaller valid position.
    #    N2: Swap two elements in different rows (if valid).
    #    N3: Swap two elements in the same row (if valid).
    # 3. Perturbation: Randomly select a few elements and shift them (repairing validity if possible).
    
    # Given the strict "distinct differences" constraint, random perturbations often break validity.
    # A more robust approach for this specific problem is "Iterated Local Search" (ILS)
    # or a specialized VNS that only accepts valid states.
    
    # Let's implement a VNS that operates on valid states.
    # We define the "scope" as the max entry. We want to minimize it.
    # We will try to "compress" the rows.

    def local_search_compress(rows):
        """
        Perform local search to reduce the maximum entry.
        We iterate through elements and try to move them down if possible.
        """
        current_rows = [row[:] for row in rows]
        improved = True
        max_iter = 1000
        iter_count = 0
        
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            
            # Shuffle order of elements to avoid bias
            indices = []
            for r in range(n_rows):
                for c in range(k, 0, -1):
                    indices.append((r, c))
            random.shuffle(indices)
            
            for r, c in indices:
                if time.time() > deadline:
                    break
                
                old_val = current_rows[r][c]
                min_possible = current_rows[r][c - 1] + 1
                
                # Try to move down
                for new_val in range(min_possible, old_val):
                    # Check if moving current_rows[r][c] to new_val is valid
                    # We need to check all differences in row r
                    # And ensure they don't conflict with other rows
                    
                    # Efficient check:
                    # 1. Compute diffs for the modified row
                    # 2. Check against global used set (excluding current row's diffs)
                    
                    # To be safe and simple, we do a full validation of the modified row
                    # against the rest.
                    
                    # Temporarily modify
                    test_row = current_rows[r][:]
                    test_row[c] = new_val
                    
                    # Check increasing
                    if test_row[c-1] >= test_row[c] or (c < k and test_row[c] >= test_row[c+1]):
                        continue
                        
                    # Check differences in this row
                    row_diffs = set()
                    valid_row = True
                    for i in range(len(test_row)):
                        for j in range(i + 1, len(test_row)):
                            d = test_row[j] - test_row[i]
                            if d in row_diffs:
                                valid_row = False
                                break
                            row_diffs.add(d)
                    if not valid_row:
                        continue
                        
                    # Check against other rows
                    conflict = False
                    for r2 in range(n_rows):
                        if r2 == r:
                            continue
                        for i in range(len(current_rows[r2])):
                            for j in range(i + 1, len(current_rows[r2])):
                                d = current_rows[r2][j] - current_rows[r2][i]
                                if d in row_diffs:
                                    conflict = True
                                    break
                            if conflict:
                                break
                        if conflict:
                            break
                    
                    if not conflict:
                        current_rows[r][c] = new_val
                        improved = True
                        break
                        
        return current_rows

    def perturb(rows, strength):
        """
        Randomly perturb the rows by moving some elements to random valid positions.
        This is tricky because we must maintain validity.
        Instead, we will "rebuild" a few rows using backtracking but with a random seed
        and a tighter bound, or simply shuffle the assignment of difference sets.
        
        A simpler perturbation for constrained problems:
        Pick a row, remove it, and rebuild it using backtracking with a randomized order.
        """
        new_rows = [row[:] for row in rows]
        r_idx = random.randint(0, n_rows - 1)
        
        # Remove row r_idx
        row_to_rebuild = new_rows.pop(r_idx)
        
        # Collect used diffs from other rows
        used_diffs = set()
        for r in range(n_rows - 1):
            for i in range(len(new_rows[r])):
                for j in range(i + 1, len(new_rows[r])):
                    used_diffs.add(new_rows[r][j] - new_rows[r][i])
                    
        # Rebuild row r_idx
        # Use a randomized backtracking
        def _bt_rebuild(used, max_val_limit):
            row = [0]
            def _bt(idx, last_val):
                if time.time() > deadline:
                    return False
                if idx == k:
                    return True
                min_val = last_val + 1
                max_val = min(max_val_limit, min_val + (k - idx - 1))
                
                candidates = list(range(min_val, max_val + 1))
                random.shuffle(candidates)
                
                for v in candidates:
                    new_diffs = set()
                    conflict = False
                    for a in row:
                        d = v - a
                        if d in used:
                            conflict = True
                            break
                        new_diffs.add(d)
                    if conflict:
                        continue
                    row.append(v)
                    used.update(new_diffs)
                    if _bt(idx + 1, v):
                        return True
                    else:
                        row.pop()
                    used.difference_update(new_diffs)
                return False
            
            if _bt(1, 0):
                return row[:]
            return None

        # Try to rebuild with a scope slightly smaller than current max
        current_max = max(max(row) for row in new_rows)
        # Try a few random scopes
        found_row = None
        for _ in range(5):
            target_scope = random.randint(lower_bound, current_max)
            row = _bt_rebuild(used_diffs.copy(), target_scope)
            if row:
                found_row = row
                break
                
        if found_row:
            new_rows.insert(r_idx, found_row)
        else:
            # If rebuild failed, restore original
            new_rows.insert(r_idx, row_to_rebuild)
            
        return new_rows

    # Main VNS Loop
    initial_rows = pure_greedy()
    best_rows = [row[:] for row in initial_rows]
    best_max = get_max_entry(best_rows)
    
    current_rows = [row[:] for row in initial_rows]
    current_max = best_max
    
    vns_iter = 0
    max_vns_iters = 50
    
    while vns_iter < max_vns_iters and time.time() < deadline:
        vns_iter += 1
        
        # Local Search
        polished = local_search_compress(current_rows)
        polished_max = get_max_entry(polished)
        
        if polished_max < current_max:
            current_rows = polished
            current_max = polished_max
            if current_max < best_max:
                best_rows = [row[:] for row in current_rows]
                best_max = current_max
            continue # If improved, stay in same neighborhood
        
        # Perturbation
        # Increase perturbation strength if stuck
        strength = vns_iter % 3 + 1
        perturbed = perturb(current_rows, strength)
        
        if perturbed is None:
            break
            
        perturbed_max = get_max_entry(perturbed)
        
        # Accept if better or equal (to escape local minima, we might accept worse,
        # but for minimization of max entry, accepting worse usually leads to worse final result.
        # However, VNS often accepts worse to jump out of local optima.
        # Given the hard constraint, let's only accept if it leads to a better local optimum
        # after re-polishing, or simply accept if it's better.)
        
        # For now, let's just accept if it's better.
        if perturbed_max < current_max:
            current_rows = perturbed
            current_max = perturbed_max
            if current_max < best_max:
                best_rows = [row[:] for row in current_rows]
                best_max = current_max
        else:
            # If not better, try to polish the perturbed state
            polished_perturbed = local_search_compress(perturbed)
            polished_p_max = get_max_entry(polished_perturbed)
            if polished_p_max < current_max:
                current_rows = polished_perturbed
                current_max = polished_p_max
                if current_max < best_max:
                    best_rows = [row[:] for row in current_rows]
                    best_max = current_max
            else:
                # No improvement, break to try a different strategy or restart
                # In a full VNS, we would change neighborhood. Here, we just break.
                break

    # Final validation
    if not is_valid(best_rows):
        best_rows = pure_greedy()

    return best_rows
# EVOLVE-BLOCK-END
