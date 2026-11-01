"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin under 150s kill

    def time_left():
        return deadline - time.time()

    def row_diffs_set(row):
        """All positive differences in a row (as a set)."""
        diffs = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                diffs.add(row[j] - ri)
        return diffs

    def is_valid(rows):
        seen = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                ri = row[i]
                for j in range(i + 1, n):
                    d = row[j] - ri
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def scope(rows):
        return max(max(row) for row in rows)

    # ---------- Greedy baseline (fast, deterministic) ----------
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
                    row.append(x)
                    row_diffs |= new
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = scope(best_rows)
    
    # Precompute row diffs for current best
    best_row_diffs = [row_diffs_set(r) for r in best_rows]
    best_used = set()
    for rd in best_row_diffs:
        best_used |= rd

    # ---------- Helper: Rebuild a row given used diffs from others ----------
    def rebuild_row(used_other, jitter=0.0, max_val=None, local_rng=None):
        """Rebuild a single row given differences already used by other rows.
        Returns the row list or None if failed."""
        if local_rng is None:
            local_rng = random
        row = [0]
        row_diffs = set()
        x = 0
        # Heuristic bound: don't go too far beyond current best scope
        search_end = max_val if max_val is not None else (best_scope + 50)
        
        # Ensure we can at least fit k+1 elements
        # A loose bound: max_val should be at least k*(k+1)/2 + something
        if search_end < 100:
             search_end = 100

        while len(row) < k + 1:
            found = False
            # Look ahead a bit to allow skipping
            for trial_x in range(x + 1, search_end + 1):
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    # Apply jitter: sometimes skip a valid candidate to explore
                    if jitter > 0 and local_rng.random() < jitter and trial_x < 300:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    x = trial_x
                    found = True
                    break
            if not found:
                return None
        return row

    # ---------- Simulated Annealing Core ----------
    def simulated_annealing(initial_rows, initial_scope, budget):
        """
        Runs SA to minimize scope.
        State: rows
        Energy: scope
        Moves: 
          1. Shift last element of a random row to a new valid position.
          2. Rebuild a random row with jitter.
        """
        rows = [list(r) for r in initial_rows]
        row_diffs = [row_diffs_set(r) for r in rows]
        used = set()
        for rd in row_diffs:
            used |= rd
        
        current_scope = initial_scope
        best_sa_rows = [list(r) for r in rows]
        best_sa_scope = current_scope
        
        # Cooling schedule
        T_start = 50.0
        T_end = 0.1
        cooling = 0.9995  # Adjust based on budget if needed, but fixed is safer
        
        # Pre-calculate valid positions for shift moves? 
        # For simplicity, we generate candidates on the fly.
        
        steps = 0
        while time_left() > budget + 5:
            if T_start < T_end:
                break
            T_start *= cooling
            
            # Choose a random row
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            rd = row_diffs[r_idx]
            
            # Move Type 1: Shift last element (most likely to reduce scope)
            # We try to move the last element to a smaller valid position, 
            # or a larger one if we want to explore (though less likely to help scope directly).
            # Actually, moving the last element *down* is the primary scope reducer.
            # Moving it *up* increases scope, but might allow other rows to shrink? 
            # In this problem, rows are independent except for diff conflicts.
            # Shrinking one row frees diffs, potentially allowing others to shrink.
            
            # Let's try to find a new position for the last element of row r_idx
            last_idx = len(row) - 1
            old_val = row[last_idx]
            
            # Candidate range: from row[last_idx-1]+1 to old_val+10 (allow small expansion)
            # We prioritize shrinking.
            found_new_val = False
            new_val = old_val
            
            # Try shrinking first
            for trial in range(old_val - 1, row[last_idx-1], -1):
                # Check if trial is valid
                # Remove old diffs, add new diffs
                new_row = row[:last_idx] + [trial]
                new_rd = row_diffs_set(new_row)
                
                # Check conflicts with other rows
                # used_other = used - rd
                # We need: (new_rd - rd) & used_other == empty
                # And: new_rd & rd == empty (internal distinctness, guaranteed by construction if increasing)
                
                # Efficient check:
                # Diffs that disappear: rd - new_rd
                # Diffs that appear: new_rd - rd
                # We need (new_rd - rd) to be disjoint from (used - rd)
                
                # Since n is small (k+1 <= 9 usually), recomputing set is fast enough for SA steps
                # But let's optimize slightly:
                # We know row[last_idx-1] is fixed.
                # New diffs are {trial - a for a in row[:last_idx]}
                # Old diffs were {old_val - a for a in row[:last_idx]}
                
                # Let's just use the set check for robustness in SA, it's fast enough for ~100k steps
                if time_left() < budget + 5:
                    break
                
                # Check internal validity (strictly increasing)
                if trial <= row[last_idx-1]:
                    continue
                
                new_rd = row_diffs_set(new_row)
                
                # Check global validity
                # The set of all diffs in the system would be:
                # (used - rd) U new_rd
                # This is valid iff (new_rd - rd) & (used - rd) is empty
                # Which is equivalent to: new_rd & (used - rd) is empty
                # Note: new_rd might contain diffs that were in rd (if trial == old_val, but we are changing it)
                # Actually, if trial != old_val, the new diffs {trial - a} are distinct from {old_val - a} for a < last_idx?
                # Not necessarily. E.g., row [0, 2, 4], move 4 to 3. Diffs: {2,4,2} -> {2,3,1}. 
                # Old: {2, 4, 2} (wait, row diffs are distinct within row? No, "all differences ... are distinct over all rows".
                # The problem statement says: "all positive differences a_j - a_i (i < j) inside the rows are distinct over all rows".
                # This implies diffs within a row must be distinct, AND diffs across rows must be distinct.
                # So yes, row_diffs_set must be a set.
                
                # So we need:
                # 1. new_row is strictly increasing (checked)
                # 2. new_rd has no duplicates (guaranteed if strictly increasing? No. [0, 1, 3] -> diffs 1, 3, 2. Distinct. [0, 2, 3] -> 2, 3, 1. Distinct.
                #    Actually, for a strictly increasing sequence, are all pairwise differences distinct?
                #    No. [0, 1, 3, 4] -> 1, 3, 4, 2, 3, 1. Dups!
                #    So we MUST check internal distinctness.
                #    row_diffs_set does this. If len(new_rd) < len(new_row)*(len(new_row)-1)//2, it's invalid.
                
                if len(new_rd) != len(new_row) * (len(new_row) - 1) // 2:
                    continue

                # Check against global used
                # used_other = used - rd
                # We need new_rd & used_other to be empty.
                # Note: new_rd might intersect with rd? 
                # If new_rd has an element d that is also in rd, that's fine as long as d is not in used_other.
                # But wait, if d is in new_rd and d is in rd, does that cause a conflict?
                # The global set is (used - rd) U new_rd.
                # If d is in new_rd and d is in rd, then d is in the union.
                # But d was already in used (via rd). So d is in the union.
                # The condition for validity is that the global set has no duplicates.
                # Since used was a set, and we remove rd and add new_rd, 
                # duplicates can only occur if:
                # 1. new_rd has internal duplicates (checked)
                # 2. new_rd contains an element that is in (used - rd).
                
                if not (new_rd & (used - rd)):
                    new_val = trial
                    found_new_val = True
                    # Update state
                    rows[r_idx] = new_row
                    row_diffs[r_idx] = new_rd
                    # Update global used
                    used -= rd
                    used |= new_rd
                    break
            
            if not found_new_val:
                # Try expanding? Or just skip?
                # Expanding increases scope, which we want to minimize.
                # However, expanding might free up tight constraints? 
                # In this problem, expanding a row uses MORE diffs, which makes it HARDER for other rows.
                # So expanding is almost never beneficial for minimizing scope.
                # We should only consider moves that keep scope same or lower, or use SA to accept worse.
                # Let's try a "Rebuild" move instead if shift failed.
                pass

            # Move Type 2: Rebuild a random row (Perturbation)
            # This is a larger move.
            if random.random() < 0.3: # 30% chance to do a rebuild
                # Rebuild row r_idx
                used_other = used - row_diffs[r_idx]
                # Try to rebuild with a max_val slightly larger than current max to allow flexibility
                # Or try to rebuild smaller?
                # Rebuild usually produces a "greedy" row which might be larger or smaller.
                new_row = rebuild_row(used_other, jitter=0.2, max_val=current_scope + 20, local_rng=random)
                if new_row is not None:
                    new_rd = row_diffs_set(new_row)
                    if len(new_rd) == len(new_row) * (len(new_row) - 1) // 2:
                        if not (new_rd & used_other):
                            # Calculate new scope
                            new_scope = max(max(r) for r in rows) # Wait, only row r_idx changed
                            # old max of row r_idx was max(rows[r_idx])
                            # new max is max(new_row)
                            # global scope is max(max(other rows), max(new_row))
                            other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx) if n_rows > 1 else 0
                            new_global_scope = max(other_max, max(new_row))
                            
                            delta = new_global_scope - current_scope
                            
                            if delta <= 0 or (T_start > 0 and random.random() < math.exp(-delta / T_start)):
                                # Accept
                                rows[r_idx] = new_row
                                row_diffs[r_idx] = new_rd
                                used -= row_diffs[r_idx] # Wait, row_diffs[r_idx] was updated? No, we stored it in new_rd
                                # Correct update:
                                # We need to remove OLD rd from used, add NEW rd
                                # I updated row_diffs[r_idx] above? No.
                                
                                # Let's restart the state update logic to be safe
                                old_rd = row_diffs[r_idx]
                                # Check validity again
                                if not (new_rd & (used - old_rd)):
                                    row_diffs[r_idx] = new_rd
                                    used -= old_rd
                                    used |= new_rd
                                    rows[r_idx] = new_row
                                    current_scope = new_global_scope
                                    
                                    if current_scope < best_sa_scope:
                                        best_sa_scope = current_scope
                                        best_sa_rows = [list(r) for r in rows]
            else:
                # If we did a shift move and it was accepted (found_new_val True), we already updated state.
                # If found_new_val was False, we did nothing.
                # We need to handle the case where shift was tried but failed.
                # In the code above, if found_new_val is True, state is updated.
                # If False, state is unchanged.
                # We should update best_sa if current_scope < best_sa_scope after a shift.
                if found_new_val:
                    # State already updated
                    if current_scope < best_sa_scope:
                        best_sa_scope = current_scope
                        best_sa_rows = [list(r) for r in rows]
            
            steps += 1
            # Limit steps per second to avoid infinite loop if time check is slow
            if steps % 1000 == 0 and time_left() < budget + 5:
                break

        return best_sa_rows, best_sa_scope

    # ---------- Local Polish (from current program, very effective) ----------
    def final_polish(rows):
        """High-precision single-element polish: try replacing each element with a smaller valid one."""
        rows = [list(r) for r in rows]
        improved = True
        max_passes = 10
        pass_count = 0
        while improved and time_left() > 3 and pass_count < max_passes:
            improved = False
            pass_count += 1
            for ri in range(n_rows):
                if time_left() < 3:
                    break
                row = rows[ri]
                # Collect all diffs used by other rows
                used_other = set()
                for rj in range(n_rows):
                    if rj != ri:
                        used_other |= row_diffs_set(rows[rj])
                # For each position (except first which is always 0), try to reduce
                for pos in range(len(row) - 1, 0, -1):  # process from end to start
                    if time_left() < 3:
                        break
                    current_val = row[pos]
                    # Try values from current_val - 1 down to row[pos-1] + 1
                    for trial in range(current_val - 1, row[pos - 1], -1):
                        # Check if replacing row[pos] with trial keeps row valid and no diff conflicts
                        new_row = row[:pos] + [trial] + row[pos+1:]
                        # Check strict increasing
                        valid_increasing = True
                        for ii in range(1, len(new_row)):
                            if new_row[ii] <= new_row[ii-1]:
                                valid_increasing = False
                                break
                        if not valid_increasing:
                            continue
                        # Check all diffs in new_row are distinct and don't conflict with used_other
                        new_diffs = set()
                        conflict = False
                        n = len(new_row)
                        for ii in range(n):
                            ri_val = new_row[ii]
                            for jj in range(ii + 1, n):
                                d = new_row[jj] - ri_val
                                if d in new_diffs or d in used_other:
                                    conflict = True
                                    break
                                new_diffs.add(d)
                            if conflict:
                                break
                        if not conflict:
                            # Valid replacement!
                            row[pos] = trial
                            improved = True
                            break  # Move to next position
                rows[ri] = row
        return rows

    # ---------- Main Search Loop ----------
    
    # 1. Initial Polish
    if time_left() > 15:
        polished = final_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # 2. Simulated Annealing Phase
    # Allocate 60% of remaining time to SA
    sa_budget = (deadline - time.time()) * 0.6
    if time_left() > 10:
        sa_rows, sa_scope = simulated_annealing(best_rows, best_scope, sa_budget)
        if sa_scope < best_scope:
            best_rows = sa_rows
            best_scope = sa_scope
    
    # 3. Final Polish on SA result
    if time_left() > 5:
        polished = final_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)
            
    # 4. One more SA run with lower temperature if time permits?
    # Or just polish again.
    if time_left() > 3:
        polished = final_polish(best_rows)
        if scope(polished) < best_scope:
            best_rows = polished
            best_scope = scope(polished)

    # Final validity guarantee.
    if not is_valid(best_rows):
        # Fallback to a fresh greedy (always valid).
        best_rows = greedy()

    return best_rows
# EVOLVE-BLOCK-END
