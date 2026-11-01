"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def get_diffs(row):
        """Return set of all positive differences in the row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            d_set = get_diffs(row)
            if len(d_set) != len(row) * (len(row) - 1) // 2:
                return False # Internal collision
            if d_set & seen:
                return False # Cross-row collision
            seen |= d_set
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline ----------
    def greedy():
        used = set()
        rows = []
        for _ in range(n_rows):
            row, x = [0], 0
            row_diffs = set()
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not (new & used) and not (new & row_diffs):
                    # Check internal uniqueness of new diffs against each other?
                    # Since x is increasing, x-a are distinct.
                    # But we need to ensure new diffs don't collide with existing row_diffs.
                    if len(new | row_diffs) == len(new) + len(row_diffs):
                        row.append(x)
                        row_diffs |= new
                        used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)
    
    # Maintain global used_diffs for fast checking
    all_diffs = set()
    for r in best_rows:
        all_diffs |= get_diffs(r)

    def try_decrease_element(rows, row_idx, elem_idx, max_search=200):
        """
        Try to decrease rows[row_idx][elem_idx] to a smaller valid value.
        Returns True if a change was made.
        """
        row = rows[row_idx]
        old_val = row[elem_idx]
        
        # Determine bounds
        low_bound = row[elem_idx - 1] + 1 if elem_idx > 0 else 0
        high_bound = row[elem_idx + 1] - 1 if elem_idx < k else old_val - 1
        
        if high_bound < low_bound:
            return False

        # Remove old diffs involving this element from all_diffs
        old_diffs = set()
        for i in range(k + 1):
            if i == elem_idx:
                continue
            if i < elem_idx:
                old_diffs.add(old_val - row[i])
            else:
                old_diffs.add(row[i] - old_val)
        
        all_diffs -= old_diffs

        # Try values from low_bound to high_bound (increasing order to find smallest)
        # But we want to minimize the max entry, so we want the smallest valid v.
        # However, if we are not at the last element, reducing it might force later elements to be larger?
        # No, later elements are fixed in this move. We just change this one.
        # If we reduce an early element, it might conflict with later elements' diffs.
        
        found_val = None
        # Search from low_bound upwards? Or from high_bound downwards?
        # To minimize the scope, we generally want to reduce the largest elements first.
        # If we are reducing elem_idx, we want the smallest valid v to keep the row "tight".
        # Let's search from low_bound to high_bound.
        
        # Optimization: If elem_idx is the last one, we just want the smallest valid v.
        # If elem_idx is not the last, reducing it might be good, but we must ensure
        # that the subsequent elements can still fit? No, they are fixed.
        # But if we reduce row[j], the differences to row[j+1..k] increase.
        # This might cause conflicts with global diffs.
        
        for v in range(low_bound, high_bound + 1):
            if v == old_val:
                continue
            
            new_diffs = set()
            valid = True
            for i in range(k + 1):
                if i == elem_idx:
                    continue
                if i < elem_idx:
                    d = v - row[i]
                else:
                    d = row[i] - v
                
                # Check if this diff is already in all_diffs (which excludes old_diffs of this element)
                if d in all_diffs:
                    valid = False
                    break
                
                # Check if this diff collides with other new diffs in this set
                # (i.e., is it a duplicate within the new_diffs set?)
                if d in new_diffs:
                    valid = False
                    break
                
                new_diffs.add(d)
            
            if valid:
                # Also need to check if new_diffs collide with the REMAINING diffs of the row
                # The remaining diffs are (get_diffs(row) - old_diffs).
                # We already checked against all_diffs which contains those remaining diffs?
                # Yes, all_diffs was updated by removing old_diffs.
                # So all_diffs contains:
                # 1. Diffs from other rows.
                # 2. Diffs from this row NOT involving elem_idx.
                # So checking against all_diffs is sufficient.
                
                found_val = v
                # We found the smallest valid v. 
                # Wait, is it always best to take the smallest?
                # If we take a slightly larger v, it might allow further reductions in other elements?
                # For a simple descent, smallest valid is a good heuristic to keep the row compact.
                break

        if found_val is not None:
            row[elem_idx] = found_val
            all_diffs |= new_diffs
            return True
        else:
            # Revert: add old_diffs back
            all_diffs |= old_diffs
            return False

    def local_search(rows):
        """
        Perform local search by trying to decrease elements in rows.
        Priority: Rows with largest max, then elements from last to first.
        """
        improved = True
        while improved and time_left() > 5:
            improved = False
            # Sort rows by max value descending
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            
            for r_idx in order:
                if time_left() < 2:
                    break
                # Try to decrease elements from last to first
                # Decreasing the last element reduces scope directly.
                # Decreasing earlier elements might reduce scope if it allows the last to be smaller later,
                # but directly, only the last element's value contributes to the row's max.
                # However, if we decrease an earlier element, it might not change the max immediately,
                # but it might free up difference space for the last element to be decreased in a subsequent step.
                
                # Let's try all elements from last to first.
                for e_idx in range(k, -1, -1):
                    if e_idx == 0:
                        continue # Cannot decrease 0
                    if try_decrease_element(rows, r_idx, e_idx):
                        improved = True
                        # Restart the inner loop? Or just continue?
                        # Continuing might allow further reductions in the same row.
                        pass
        
        return rows

    def kick(rows):
        """
        Perturb the solution.
        Strategy: Pick a random row and a random element (not 0), and increase it slightly,
        or swap two elements from different rows if valid.
        Or, rebuild a random row with a "loose" greedy to escape local minima.
        """
        rows_copy = [list(r) for r in rows]
        
        # Rebuild a random row with high randomness
        r_idx = random.randint(0, n_rows - 1)
        used_other = set()
        for j in range(n_rows):
            if j != r_idx:
                used_other |= get_diffs(rows_copy[j])
        
        old_row = rows_copy[r_idx]
        old_diffs = get_diffs(old_row)
        
        # Try to find a new row that is "different" or "looser"
        # We can try to construct a row that avoids the "tightest" packing.
        # Simple kick: Replace the row with a greedy one that has some randomness.
        row, x = [0], 0
        row_diffs = set()
        success = False
        
        # Try a few times to find a valid row
        for attempt in range(10):
            row, x = [0], 0
            row_diffs = set()
            valid = True
            while len(row) < k + 1:
                found = False
                # Randomized selection
                # Bias towards smaller values but allow jumps
                search_range = 50 + random.randint(0, 50)
                for trial_x in range(x + 1, x + search_range):
                    new = {trial_x - a for a in row}
                    if not (new & used_other) and not (new & row_diffs):
                        # Check internal uniqueness
                        if len(new | row_diffs) == len(new) + len(row_diffs):
                            # Jitter: skip some valid candidates to explore
                            if random.random() < 0.3 and trial_x < 200:
                                continue
                            row.append(trial_x)
                            row_diffs |= new
                            found = True
                            break
                if not found:
                    valid = False
                    break
                x = row[-1]
            
            if valid:
                rows_copy[r_idx] = row
                success = True
                break
        
        if not success:
            # If rebuild failed, revert to original
            rows_copy[r_idx] = old_row
            
        # Update global all_diffs
        nonlocal all_diffs
        all_diffs -= get_diffs(rows[r_idx])
        all_diffs |= get_diffs(rows_copy[r_idx])
        
        return rows_copy

    # Main Loop
    # 1. Initial Local Search
    if time_left() > 10:
        best_rows = local_search(best_rows)
        best_scope = scope(best_rows)
        # Update all_diffs
        all_diffs = set()
        for r in best_rows:
            all_diffs |= get_diffs(r)

    # 2. Iterated Local Search
    no_improve_count = 0
    while time_left() > 15:
        # Local Search
        polished = [list(r) for r in best_rows]
        # Re-sync all_diffs with polished
        temp_diffs = set()
        for r in polished:
            temp_diffs |= get_diffs(r)
        # We need to pass all_diffs to local_search or make it global.
        # Let's refactor local_search to use a global-ish all_diffs or pass it.
        # For simplicity in this block, I'll use the global all_diffs variable.
        # But local_search modifies rows, so we must keep all_diffs in sync.
        
        # Let's just run local_search on best_rows directly, using the global all_diffs.
        # But we need to be careful: if we kick, we change best_rows.
        
        # To be safe, let's make a copy for local search? No, that's expensive.
        # Let's assume local_search modifies in place and keeps all_diffs correct.
        
        old_scope = best_scope
        best_rows = local_search(best_rows)
        new_scope = scope(best_rows)
        
        if new_scope < old_scope:
            best_scope = new_scope
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Kick if stuck
        if no_improve_count >= 5 or random.random() < 0.1:
            if time_left() < 10:
                break
            best_rows = kick(best_rows)
            no_improve_count = 0
            # After kick, do a quick local search to clean up
            if time_left() > 10:
                best_rows = local_search(best_rows)
                best_scope = scope(best_rows)

    # Final Polish
    if time_left() > 5:
        best_rows = local_search(best_rows)
        best_scope = scope(best_rows)

    # Final Validity Check
    if not is_valid(best_rows):
        # Fallback
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
