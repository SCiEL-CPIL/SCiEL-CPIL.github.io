import time
import random
import sys

_START = None

def construct(n_rows, k):
    global _START
    if _START is None:
        _START = time.time()
    
    # Safety margin: The evaluator kills at 150s. 
    # We aim to finish well before to ensure return.
    deadline = _START + 145.0 

    # Parameters from task: 4 rows, 8 integers (k=7 in construct call if k+1=8)
    # The prompt says "Choose 4 rows of 8 increasing integers".
    # construct(n_rows, k) -> rows of k+1 integers.
    # So if evaluator passes n_rows=4, k=7, we get 8 integers.
    
    d = k * (k + 1) // 2
    total_diffs = n_rows * d

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate_scope(rows):
        """Return scope if valid, else None."""
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    rows = greedy_init()
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(rows)

    rng = random.Random(42)

    def compute_diff_map(rows):
        """
        Returns a dictionary: diff -> list of (row_idx, i, j)
        This helps identify conflicts quickly.
        """
        diff_map = {}
        for r_idx, row in enumerate(rows):
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff not in diff_map:
                        diff_map[diff] = []
                    diff_map[diff].append((r_idx, i, j))
        return diff_map

    def find_conflicts(diff_map):
        """Return list of diffs that have more than 1 producer."""
        conflicts = []
        for diff, producers in diff_map.items():
            if len(producers) > 1:
                conflicts.append(diff)
        return conflicts

    def try_move_element(rows, r_idx, pos, new_val):
        """
        Attempt to set rows[r_idx][pos] to new_val.
        Returns True if the resulting rows are valid (no conflicts) and strictly increasing.
        If valid, it updates rows in place.
        If invalid, it reverts.
        """
        old_val = rows[r_idx][pos]
        
        # Check strict increase
        if pos > 0 and new_val <= rows[r_idx][pos - 1]:
            return False
        if pos < len(rows[r_idx]) - 1 and new_val >= rows[r_idx][pos + 1]:
            return False
            
        rows[r_idx][pos] = new_val
        
        # Check validity
        # For efficiency, we only check diffs involving the changed element?
        # No, the change affects diffs with all other elements in the same row.
        # It does NOT affect diffs in other rows.
        # So we only need to check if the new diffs created in row r_idx conflict with:
        # 1. Other diffs in row r_idx (internal consistency)
        # 2. Diffs in all other rows (global consistency)
        
        # To be safe and simple within the time limit, we can do a full check
        # or a partial check. Given n_rows=4, k=7, full check is O(4 * 28) = 112 ops.
        # This is very fast.
        
        scope = full_validate_scope(rows)
        if scope is None:
            rows[r_idx][pos] = old_val
            return False
        return True

    def repair_state(rows, budget_time):
        """
        Attempt to repair an invalid state by resolving conflicts.
        Returns True if repaired, False otherwise.
        Modifies rows in place.
        """
        start_t = time.time()
        max_attempts = 1000
        
        for _ in range(max_attempts):
            if time.time() - start_t > budget_time:
                break
                
            diff_map = compute_diff_map(rows)
            conflicts = find_conflicts(diff_map)
            
            if not conflicts:
                return True
            
            # Pick a random conflict
            diff = rng.choice(conflicts)
            producers = diff_map[diff]
            
            # We have at least 2 producers. Let's pick one to move.
            # We pick a random producer.
            r_idx, i, j = rng.choice(producers)
            
            # We can move row[r_idx][i] or row[r_idx][j].
            # Moving j is usually more flexible (larger range) if it's not the last element?
            # Actually, moving any element changes multiple diffs.
            # Let's try to move the element that is "easier" to move.
            # Heuristic: Try to move the larger index j first, then i.
            
            moved = False
            for pos in [j, i]:
                # Determine range for new value
                lo = rows[r_idx][pos - 1] + 1 if pos > 0 else 1
                hi = rows[r_idx][pos + 1] - 1 if pos < len(rows[r_idx]) - 1 else best_scope + 20
                
                if hi < lo:
                    continue
                
                # Try a few random values in this range
                # To be "conflict directed", we should ideally pick values that don't create new conflicts.
                # But finding such a value requires checking.
                # Let's try a few random candidates.
                candidates = set()
                for _ in range(20):
                    candidates.add(rng.randint(lo, hi))
                
                # Also try values close to the current one to minimize perturbation?
                # Or values that reduce scope?
                # For repair, we just want ANY valid value.
                
                for new_val in candidates:
                    if new_val == rows[r_idx][pos]:
                        continue
                    if try_move_element(rows, r_idx, pos, new_val):
                        moved = True
                        break
                
                if moved:
                    break
            
            if not moved:
                # If we couldn't move this producer, try another conflict or give up
                # For now, just continue to next iteration (might pick same conflict again)
                # To avoid infinite loop on same conflict, we could backtrack, but that's complex.
                # Instead, we rely on the randomness of rng.choice(conflicts) to eventually pick a different one.
                # However, if one conflict is "hard", we might loop.
                # A simple improvement: if we fail to move, we revert the last successful move? 
                # No, try_move_element reverts on failure.
                # If no candidate works, we just try again.
                pass
        
        # Check if repaired
        diff_map = compute_diff_map(rows)
        return len(find_conflicts(diff_map)) == 0

    def local_search_with_repair(rows, budget_time):
        """
        Main search loop.
        1. Try to reduce scope (Hill Climb on valid states).
        2. If stuck, Perturb (make invalid) and Repair.
        """
        start_t = time.time()
        cur_rows = [r[:] for r in rows]
        cur_scope = get_scope(cur_rows)
        best_local_scope = cur_scope
        best_local_rows = [r[:] for r in cur_rows]
        
        no_improve_count = 0
        
        while time.time() - start_t < budget_time:
            # 1. Try to reduce scope
            improved = False
            
            # Find the max element
            max_val = max(max(r) for r in cur_rows)
            # Find all positions with max_val
            max_positions = []
            for r_idx, row in enumerate(cur_rows):
                for pos, val in enumerate(row):
                    if val == max_val:
                        max_positions.append((r_idx, pos))
            
            # Try to lower one of them
            rng.shuffle(max_positions)
            for r_idx, pos in max_positions:
                old_val = cur_rows[r_idx][pos]
                # Try values just below old_val
                # Range: [cur_rows[r_idx][pos-1]+1, old_val-1]
                lo = cur_rows[r_idx][pos - 1] + 1 if pos > 0 else 1
                hi = old_val - 1
                
                if hi < lo:
                    continue
                
                # Try a few candidates, preferring lower values (bigger reduction)
                candidates = set()
                for v in range(lo, min(lo + 10, hi + 1)):
                    candidates.add(v)
                for _ in range(5):
                    candidates.add(rng.randint(lo, hi))
                
                for new_val in sorted(candidates, reverse=True): # Try bigger reductions first? 
                    # Actually, if we want to reduce scope, we want the smallest valid new_val.
                    # But any valid new_val < old_val reduces the max of that row.
                    # If it's the global max, it reduces global scope.
                    if try_move_element(cur_rows, r_idx, pos, new_val):
                        new_scope = get_scope(cur_rows)
                        if new_scope < cur_scope:
                            cur_scope = new_scope
                            improved = True
                            if cur_scope < best_local_scope:
                                best_local_scope = cur_scope
                                best_local_rows = [r[:] for r in cur_rows]
                            break
                if improved:
                    break
            
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
                
                # 2. If stuck, Perturb and Repair
                if no_improve_count > 5:
                    # Perturb: Make a random move that might break validity
                    r_idx = rng.randrange(n_rows)
                    pos = rng.randint(1, len(cur_rows[r_idx]) - 1)
                    old_val = cur_rows[r_idx][pos]
                    
                    # Pick a random new value, possibly outside current range to force change
                    # But must maintain strict increase locally? 
                    # If we break strict increase, try_move_element will reject it.
                    # So we must pick a value that maintains strict increase, 
                    # but might create a global conflict.
                    
                    lo = cur_rows[r_idx][pos - 1] + 1 if pos > 0 else 1
                    hi = cur_rows[r_idx][pos + 1] - 1 if pos < len(cur_rows[r_idx]) - 1 else cur_scope + 50
                    
                    if hi > lo:
                        new_val = rng.randint(lo, hi)
                        # Force the move, ignoring validity check for a moment?
                        # No, try_move_element checks validity.
                        # To perturb into invalid state, we must bypass the validity check.
                        
                        # Let's just set it and see.
                        cur_rows[r_idx][pos] = new_val
                        
                        # Now Repair
                        if repair_state(cur_rows, budget_time=1.0):
                            # If repaired, check scope
                            new_scope = get_scope(cur_rows)
                            if new_scope is not None:
                                if new_scope < cur_scope:
                                    cur_scope = new_scope
                                    if cur_scope < best_local_scope:
                                        best_local_scope = cur_scope
                                        best_local_rows = [r[:] for r in cur_rows]
                                else:
                                    # Same scope, but different structure. Accept to escape local min.
                                    cur_scope = new_scope
                            no_improve_count = 0
                        else:
                            # Repair failed, revert to last valid state
                            # We don't have a stack, so we just restart from best_local_rows?
                            # Or revert the perturbation?
                            # Simplest: revert to best_local_rows
                            cur_rows = [r[:] for r in best_local_rows]
                            cur_scope = best_local_scope
                            no_improve_count = 0 # Reset to try again
        
        return best_local_rows, best_local_scope

    # Main loop: Iterated Local Search
    max_restarts = 50
    time_per_repair = 2.0
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
            
        # Start from best found so far
        cur_rows = [r[:] for r in best_rows]
        
        # Run local search with repair
        ls_rows, ls_scope = local_search_with_repair(cur_rows, budget_time=min(10.0, deadline - time.time() - 5.0))
        
        if ls_scope is not None and ls_scope < best_scope:
            best_scope = ls_scope
            best_rows = ls_rows
        elif ls_rows is not None:
            # Even if scope didn't improve, the structure might be better for future perturbations?
            # Usually we stick to the best scope.
            pass

    return best_rows

# Ensure the function signature matches
# The prompt says: construct(n_rows, k) -> list of n_rows rows
# The current code has: def construct(n_rows, k):
# This matches.
