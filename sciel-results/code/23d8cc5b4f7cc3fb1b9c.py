import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    n_marks = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Helper to get differences for a row
    def get_row_diffs(row):
        diffs = set()
        for i in range(n_marks):
            for j in range(i + 1, n_marks):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper to compute used diffs excluding a specific row
    def get_used_diffs_excluding(rows, exclude_row):
        used = set()
        for r in range(n_rows):
            if r == exclude_row:
                continue
            used |= get_row_diffs(rows[r])
        return used

    # Greedy initialization with "pressure" strategy
    def greedy_init(strategy="pressure"):
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n_marks:
                if strategy == "pressure":
                    # Look ahead a bit to minimize max new difference
                    base = x + 1
                    valid_cands = []
                    # Scan a window of candidates
                    for cand in range(base, base + 50):
                        new = {cand - a for a in row}
                        if not new & used:
                            # Score by max new diff (pressure)
                            valid_cands.append((cand, max(new) if new else 0))
                    if not valid_cands:
                        # Fallback: find next valid
                        cand = base
                        while True:
                            new = {cand - a for a in row}
                            if not new & used:
                                valid_cands.append((cand, max(new) if new else 0))
                                break
                            cand += 1
                    
                    # Pick candidate with smallest max new diff, tie-break by smallest value
                    valid_cands.sort(key=lambda t: (t[1], t[0]))
                    best_cand = valid_cands[0][0]
                    new = {best_cand - a for a in row}
                    row.append(best_cand)
                    used |= new
                    x = best_cand
                else:
                    # Pure greedy
                    x += 1
                    while True:
                        new = {x - a for a in row}
                        if not new & used:
                            break
                        x += 1
                    row.append(x)
                    used |= {x - a for a in row[:-1]}
            rows.append(row)
        return rows

    # Rebuild a single row using backtracking, given other rows are fixed
    def rebuild_row(row_idx, rows, max_val_bound=None):
        if max_val_bound is None:
            max_val_bound = rows[row_idx][-1] + 20
        
        used_diffs = get_used_diffs_excluding(rows, row_idx)
        
        new_row = [0]
        
        def backtrack(pos, last_val):
            if pos == n_marks:
                return True
            
            start_val = last_val + 1
            if start_val > max_val_bound:
                return False
            
            for v in range(start_val, max_val_bound + 1):
                new_diffs = {v - a for a in new_row}
                if new_diffs & used_diffs:
                    continue
                if len(new_diffs) != len(new_row):
                    continue
                
                new_row.append(v)
                if backtrack(pos + 1, v):
                    return True
                new_row.pop()
            return False
        
        if backtrack(1, 0):
            return new_row
        return None

    # State for incremental cost tracking
    # diff_counts: dictionary mapping difference value to count
    # conflicts: number of difference values with count > 1
    # max_entry: maximum value in any row
    
    class SearchState:
        def __init__(self, rows):
            self.rows = [list(r) for r in rows]
            self.n_rows = len(rows)
            self.k = k
            self.n_marks = n_marks
            self.diff_counts = {}
            self.conflicts = 0
            self.max_entry = 0
            self._update_all()
            
        def _update_all(self):
            self.diff_counts = {}
            self.conflicts = 0
            self.max_entry = 0
            for r in range(self.n_rows):
                row = self.rows[r]
                if row[-1] > self.max_entry:
                    self.max_entry = row[-1]
                for i in range(self.n_marks):
                    for j in range(i + 1, self.n_marks):
                        d = row[j] - row[i]
                        self.diff_counts[d] = self.diff_counts.get(d, 0) + 1
            for d, cnt in self.diff_counts.items():
                if cnt > 1:
                    self.conflicts += 1

        def cost(self):
            # Heavily penalize conflicts
            return self.conflicts * 10000 + self.max_entry

        def apply_move(self, r, pos, new_val):
            """
            Apply a move changing rows[r][pos] to new_val.
            Returns the delta in cost.
            Assumes the move is valid in terms of strict increase (caller must check).
            """
            old_val = self.rows[r][pos]
            if old_val == new_val:
                return 0
            
            row = self.rows[r]
            old_cost = self.cost()
            
            # Remove old differences involving pos
            for i in range(self.n_marks):
                if i == pos:
                    continue
                if i < pos:
                    d = row[pos] - row[i]
                else:
                    d = row[i] - row[pos]
                
                # Decrement count for d
                self.diff_counts[d] -= 1
                if self.diff_counts[d] == 0:
                    del self.diff_counts[d]
                # Update conflict count
                if self.diff_counts.get(d, 0) == 1:
                    self.conflicts -= 1
                elif self.diff_counts.get(d, 0) == 2:
                    # Went from 3 to 2, still conflict, no change
                    pass
                # Wait, logic:
                # Before: count = c
                # After: count = c - 1
                # If c-1 == 1 (was 2): conflict removed (was 1 conflict, now 0) -> conflicts -= 1
                # If c-1 == 2 (was 3): still conflict (was 1 conflict, now 1 conflict) -> no change
                # If c-1 == 0 (was 1): no conflict before, no conflict after -> no change
                
                # Let's re-evaluate conflict delta properly
                # c_old = self.diff_counts[d] + 1 (before decrement)
                # c_new = self.diff_counts[d] (after decrement)
                # Conflict exists if c > 1
                # delta_conflicts = (1 if c_new > 1 else 0) - (1 if c_old > 1 else 0)
                
                # I will just recalculate conflicts for the affected diffs at the end of the move to be safe and simple,
                # or track it carefully.
                # Let's track it carefully.
                
            # This manual tracking is error-prone. Let's do a local recalculation of conflicts for affected diffs.
            # Affected diffs are those involving index 'pos' in row 'r'.
            # There are n_marks - 1 such diffs.
            
            # Let's revert and use a simpler incremental approach:
            # 1. Save old values
            # 2. Update diff_counts
            # 3. Recalculate conflicts only for the diffs that changed.
            
            # Re-doing the logic inside apply_move for clarity and correctness.
            
            # Reset state to pre-move for this method? No, we modify in place.
            # Let's just implement a robust `move` that returns delta.
            
            # For simplicity and robustness in this specific implementation, 
            # I will calculate the delta by:
            # 1. Identifying the set of diffs that change: D_old, D_new
            # 2. For each d in D_old: 
            #    - Update count
            #    - Update conflict status
            # 3. For each d in D_new:
            #    - Update count
            #    - Update conflict status
            
            # Let's restart the apply_move logic cleanly.
            
            # Rollback the partial changes above? 
            # Actually, I already modified diff_counts in the loop above. 
            # Let's just write the function correctly from scratch.
            
            # --- CLEAN IMPLEMENTATION OF APPLY MOVE ---
            
            # The previous loop already decremented counts. 
            # I need to finish the conflict update and then add new diffs.
            
            # This is getting messy. I will define a helper to calculate conflict delta for a diff.
            
            def conflict_status(cnt):
                return 1 if cnt > 1 else 0

            # The loop above decremented counts for D_old.
            # I need to update self.conflicts based on the change.
            # Let's assume the loop above is correct in decrementing counts.
            # I need to know the OLD count to determine the OLD conflict status.
            
            # This approach is too fragile. 
            # I will implement `apply_move` by:
            # 1. Calculating D_old and D_new sets.
            # 2. For each d in D_old: 
            #    old_cnt = diff_counts[d]
            #    new_cnt = old_cnt - 1
            #    diff_counts[d] = new_cnt
            #    delta_conf = conflict_status(new_cnt) - conflict_status(old_cnt)
            #    self.conflicts += delta_conf
            # 3. For each d in D_new:
            #    old_cnt = diff_counts.get(d, 0)
            #    new_cnt = old_cnt + 1
            #    diff_counts[d] = new_cnt
            #    delta_conf = conflict_status(new_cnt) - conflict_status(old_cnt)
            #    self.conflicts += delta_conf
            
            # Let's restart the method body to be safe.
            
            # Rollback the previous partial changes? 
            # I will just delete the previous partial code and rewrite.
            
            # Since I can't easily "rollback" in this thought process, I will write the final code block carefully.
            # In the actual code, I will implement `apply_move` correctly.
            
            pass 

        def try_move(self, r, pos, new_val):
            """
            Attempts to move rows[r][pos] to new_val.
            Returns True if accepted (based on SA logic passed in? No, this is just the state update).
            Actually, let's have the SA loop handle the acceptance logic.
            This method just performs the state change and returns the delta cost.
            """
            row = self.rows[r]
            old_val = row[pos]
            if old_val == new_val:
                return 0
            
            # Check strict increase
            if pos > 0 and new_val <= row[pos-1]:
                return None # Invalid
            if pos < self.n_marks - 1 and new_val >= row[pos+1]:
                return None # Invalid
            
            # Calculate affected diffs
            old_diffs = []
            new_diffs = []
            for i in range(self.n_marks):
                if i == pos:
                    continue
                if i < pos:
                    d_old = old_val - row[i]
                    d_new = new_val - row[i]
                else:
                    d_old = row[i] - old_val
                    d_new = row[i] - new_val
                old_diffs.append(d_old)
                new_diffs.append(d_new)
            
            # Update counts and conflicts
            delta_conflicts = 0
            
            # Process old diffs (decrement)
            for d in old_diffs:
                cnt = self.diff_counts.get(d, 0)
                if cnt == 0:
                    # Should not happen if state is consistent
                    continue
                new_cnt = cnt - 1
                self.diff_counts[d] = new_cnt
                if new_cnt == 0:
                    del self.diff_counts[d]
                
                # Conflict change
                old_conf = 1 if cnt > 1 else 0
                new_conf = 1 if new_cnt > 1 else 0
                delta_conflicts += new_conf - old_conf

            # Process new diffs (increment)
            for d in new_diffs:
                cnt = self.diff_counts.get(d, 0)
                new_cnt = cnt + 1
                self.diff_counts[d] = new_cnt
                
                old_conf = 1 if cnt > 1 else 0
                new_conf = 1 if new_cnt > 1 else 0
                delta_conflicts += new_conf - old_conf

            self.conflicts += delta_conflicts
            
            # Update max_entry
            old_max = self.max_entry
            if new_val > self.max_entry:
                self.max_entry = new_val
            # Note: if old_val was the max and we reduce it, we might need to recalc max.
            # For simplicity, if new_val < old_val, we might have lowered the max.
            # Recalculating max from scratch is O(n_rows * n_marks) which is small.
            if new_val < old_val:
                # Check if any row has max > new_max?
                # Actually, if we decrease a value, the max can only stay same or decrease.
                # If old_val was the unique max, max might decrease.
                # Let's just recalculate max if we decreased a value that was close to max.
                # Or simpler: just recalculate max always when decreasing?
                # O(4*9) = 36 ops. Negligible.
                current_max = 0
                for rr in range(self.n_rows):
                    if self.rows[rr][-1] > current_max:
                        current_max = self.rows[rr][-1]
                self.max_entry = current_max
            
            row[pos] = new_val
            
            new_cost = self.cost()
            old_cost = old_conflicts * 10000 + old_max # Wait, I didn't save old_cost correctly in the previous thought.
            # I need to return delta_cost.
            # Let's calculate delta_cost = new_cost - old_cost.
            # I have delta_conflicts and delta_max (new_max - old_max).
            
            # Actually, I can just return the new cost and let the caller compare?
            # No, SA needs delta.
            
            # Let's reconstruct:
            # I didn't save old_cost explicitly in the function scope before modifying.
            # I will refactor `try_move` to return `delta_cost`.
            
            # For the sake of the code block below, I will implement a clean `try_move` that returns delta.
            
            # Since I am writing the final code, I will ensure `try_move` is correct.
            
            # The logic above for delta_conflicts is correct.
            # delta_max = self.max_entry - old_max
            # delta_cost = delta_conflicts * 10000 + delta_max
            
            # I need `old_max` and `old_conflicts` before modification.
            # I will implement it correctly in the final code.
            
            # For now, return 0 as placeholder in this thought.
            return 0

    # Corrected SearchState with proper incremental updates
    class State:
        def __init__(self, rows):
            self.rows = [list(r) for r in rows]
            self.n_rows = len(rows)
            self.n_marks = n_marks
            self.diff_counts = {}
            self.conflicts = 0
            self.max_entry = 0
            self._init_state()

        def _init_state(self):
            self.diff_counts = {}
            self.conflicts = 0
            self.max_entry = 0
            for r in range(self.n_rows):
                row = self.rows[r]
                if row[-1] > self.max_entry:
                    self.max_entry = row[-1]
                for i in range(self.n_marks):
                    for j in range(i + 1, self.n_marks):
                        d = row[j] - row[i]
                        self.diff_counts[d] = self.diff_counts.get(d, 0) + 1
            for cnt in self.diff_counts.values():
                if cnt > 1:
                    self.conflicts += 1

        def cost(self):
            return self.conflicts * 10000 + self.max_entry

        def try_move(self, r, pos, new_val):
            """
            Returns delta_cost if move is valid, None if invalid.
            Modifies state if move is applied.
            Caller must call `undo_move` or keep track if rejected.
            For SA, it's easier to:
            1. Calculate delta
            2. If accepted, apply.
            3. If rejected, do nothing.
            
            So this method should calculate delta WITHOUT applying, or apply and allow undo.
            Let's do: calculate delta, then apply if accepted.
            
            To calculate delta, we need to simulate the change.
            """
            row = self.rows[r]
            old_val = row[pos]
            if old_val == new_val:
                return 0
            
            if pos > 0 and new_val <= row[pos-1]:
                return None
            if pos < self.n_marks - 1 and new_val >= row[pos+1]:
                return None
            
            old_diffs = []
            new_diffs = []
            for i in range(self.n_marks):
                if i == pos:
                    continue
                if i < pos:
                    old_diffs.append(old_val - row[i])
                    new_diffs.append(new_val - row[i])
                else:
                    old_diffs.append(row[i] - old_val)
                    new_diffs.append(row[i] - new_val)
            
            delta_conflicts = 0
            
            # Simulate old diffs removal
            for d in old_diffs:
                cnt = self.diff_counts.get(d, 0)
                new_cnt = cnt - 1
                old_conf = 1 if cnt > 1 else 0
                new_conf = 1 if new_cnt > 1 else 0
                delta_conflicts += new_conf - old_conf

            # Simulate new diffs addition
            for d in new_diffs:
                cnt = self.diff_counts.get(d, 0)
                new_cnt = cnt + 1
                old_conf = 1 if cnt > 1 else 0
                new_conf = 1 if new_cnt > 1 else 0
                delta_conflicts += new_conf - old_conf

            # Delta max
            delta_max = 0
            if new_val > self.max_entry:
                delta_max = new_val - self.max_entry
            elif old_val == self.max_entry and new_val < old_val:
                # Max might decrease. We need to find the new max.
                # This is expensive if done every time.
                # However, max decreases are rare in SA if we start compact.
                # Or we can just check if old_val was the max.
                # If yes, we need to scan.
                # Let's approximate: if we reduce the max, we scan.
                # Scan cost: O(n_rows * n_marks).
                # If this happens often, it's slow.
                # But reducing max is the goal.
                current_max = 0
                for rr in range(self.n_rows):
                    if self.rows[rr][-1] > current_max:
                        current_max = self.rows[rr][-1]
                # Note: if the max was in another row, it stays.
                # If the max was in this row, it might be this row's new last value or another row's max.
                # Since we only change one element, the row's max is either row[-1] (if pos != n_marks-1) or new_val (if pos == n_marks-1).
                # Actually, row is strictly increasing, so row[-1] is the max of the row.
                # If pos < n_marks-1, row[-1] is unchanged.
                # If pos == n_marks-1, row[-1] becomes new_val.
                
                # So:
                if pos == self.n_marks - 1:
                    # This row's max changed
                    new_row_max = new_val
                else:
                    new_row_max = row[-1]
                
                # Find max of all other rows
                other_max = 0
                for rr in range(self.n_rows):
                    if rr == r:
                        continue
                    if self.rows[rr][-1] > other_max:
                        other_max = self.rows[rr][-1]
                
                new_global_max = max(new_row_max, other_max)
                delta_max = new_global_max - self.max_entry
            else:
                # new_val <= self.max_entry and old_val != self.max_entry (or new_val >= old_val)
                # If new_val <= old_val and old_val != max, max unchanged.
                # If new_val > old_val but new_val <= max, max unchanged.
                pass

            delta_cost = delta_conflicts * 10000 + delta_max
            
            # Apply the move to state
            for d in old_diffs:
                cnt = self.diff_counts[d]
                self.diff_counts[d] = cnt - 1
                if self.diff_counts[d] == 0:
                    del self.diff_counts[d]
            for d in new_diffs:
                self.diff_counts[d] = self.diff_counts.get(d, 0) + 1
            
            self.conflicts += delta_conflicts
            self.max_entry += delta_max
            row[pos] = new_val
            
            return delta_cost

        def undo_move(self, r, pos, new_val):
            """Undo the last move at r, pos to new_val."""
            row = self.rows[r]
            old_val = row[pos] # This is the new_val from the perspective of undo? No.
            # We need to know the OLD value. 
            # It's better to have `apply_move` return the old value or store it.
            # Or, simply, don't use undo. Use "try, accept/reject" pattern where reject means "don't change".
            # But `try_move` above ALREADY modifies state.
            # So if we reject, we MUST undo.
            
            # This is a flaw in the design above.
            # Better design: `calculate_delta` (no modify) and `apply` (modify).
            
            # Let's stick to the "Modify and Undo" pattern but ensure we have the old value.
            # I will change the SA loop to:
            # 1. Save old_val
            # 2. Call try_move (which modifies)
            # 3. If rejected, call undo_move (which restores)
            
            # To make undo easy, I will pass the old_val to undo.
            pass

    # Given the complexity of implementing a perfect incremental state with undo in this single block,
    # and the risk of bugs, I will use a slightly less optimal but robust approach:
    # 1. Use the incremental cost for the SA loop, but implement `try_move` to return delta and modify state.
    # 2. Implement `undo_move` carefully.
    
    # Let's refine the State class to be robust.
    
    class RobustState:
        def __init__(self, rows):
            self.rows = [list(r) for r in rows]
            self.n_rows = len(rows)
            self.n_marks = n_marks
            self.diff_counts = {}
            self.conflicts = 0
            self.max_entry = 0
            self._init_state()

        def _init_state(self):
            self.diff_counts = {}
            self.conflicts = 0
            self.max_entry = 0
            for r in range(self.n_rows):
                row = self.rows[r]
                if row[-1] > self.max_entry:
                    self.max_entry = row[-1]
                for i in range(self.n_marks):
                    for j in range(i + 1, self.n_marks):
                        d = row[j] - row[i]
                        self.diff_counts[d] = self.diff_counts.get(d, 0) + 1
            for cnt in self.diff_counts.values():
                if cnt > 1:
                    self.conflicts += 1

        def cost(self):
            return self.conflicts * 10000 + self.max_entry

        def try_move(self, r, pos, new_val):
            """
            Returns (delta_cost, old_val) if valid, (None, None) if invalid.
            Modifies state.
            """
            row = self.rows[r]
            old_val = row[pos]
            if old_val == new_val:
                return 0, old_val
            
            if pos > 0 and new_val <= row[pos-1]:
                return None, None
            if pos < self.n_marks - 1 and new_val >= row[pos+1]:
                return None, None
            
            old_diffs = []
            new_diffs = []
            for i in range(self.n_marks):
                if i == pos:
                    continue
                if i < pos:
                    old_diffs.append(old_val - row[i])
                    new_diffs.append(new_val - row[i])
                else:
                    old_diffs.append(row[i] - old_val)
                    new_diffs.append(row[i] - new_val)
            
            delta_conflicts = 0
            
            for d in old_diffs:
                cnt = self.diff_counts.get(d, 0)
                new_cnt = cnt - 1
                old_conf = 1 if cnt > 1 else 0
                new_conf = 1 if new_cnt > 1 else 0
                delta_conflicts += new_conf - old_conf

            for d in new_diffs:
                cnt = self.diff_counts.get(d, 0)
                new_cnt = cnt + 1
                old_conf = 1 if cnt > 1 else 0
                new_conf = 1 if new_cnt > 1 else 0
                delta_conflicts += new_conf - old_conf

            delta_max = 0
            if new_val > self.max_entry:
                delta_max = new_val - self.max_entry
            elif old_val == self.max_entry and new_val < old_val:
                # Recalculate max
                new_row_max = new_val if pos == self.n_marks - 1 else row[-1]
                other_max = 0
                for rr in range(self.n_rows):
                    if rr != r and self.rows[rr][-1] > other_max:
                        other_max = self.rows[rr][-1]
                new_global_max = max(new_row_max, other_max)
                delta_max = new_global_max - self.max_entry

            delta_cost = delta_conflicts * 10000 + delta_max
            
            # Apply
            for d in old_diffs:
                self.diff_counts[d] -= 1
                if self.diff_counts[d] == 0:
                    del self.diff_counts[d]
            for d in new_diffs:
                self.diff_counts[d] = self.diff_counts.get(d, 0) + 1
            
            self.conflicts += delta_conflicts
            self.max_entry += delta_max
            row[pos] = new_val
            
            return delta_cost, old_val

        def undo_move(self, r, pos, new_val, old_val):
            """
            Undo a move that set rows[r][pos] to new_val (from old_val).
            """
            row = self.rows[r]
            
            # Reverse the diff count changes
            # This is the same as trying to move from new_val to old_val, but we know the state.
            # Actually, it's simpler to just reverse the logic.
            
            old_diffs = [] # These were the "new" diffs in the forward move
            new_diffs = [] # These were the "old" diffs in the forward move
            
            for i in range(self.n_marks):
                if i == pos:
                    continue
                if i < pos:
                    # Forward: new_val - row[i]
                    # Undo: old_val - row[i]
                    old_diffs.append(new_val - row[i])
                    new_diffs.append(old_val - row[i])
                else:
                    old_diffs.append(row[i] - new_val)
                    new_diffs.append(row[i] - old_val)
            
            delta_conflicts = 0
            
            # Remove old_diffs (which were added in forward move)
            for d in old_diffs:
                cnt = self.diff_counts.get(d, 0)
                new_cnt = cnt - 1
                old_conf = 1 if cnt > 1 else 0
                new_conf = 1 if new_cnt > 1 else 0
                delta_conflicts += new_conf - old_conf

            # Add new_diffs (which were removed in forward move)
            for d in new_diffs:
                cnt = self.diff_counts.get(d, 0)
                new_cnt = cnt + 1
                old_conf = 1 if cnt > 1 else 0
                new_conf = 1 if new_cnt > 1 else 0
                delta_conflicts += new_conf - old_conf

            # Reverse max change
            # We need to restore max_entry.
            # If the forward move increased max, we decrease it.
            # If the forward move decreased max, we increase it.
            # Easiest: recalculate max if it was involved.
            
            # Let's just recalculate max_entry always on undo? 
            # It's O(N). N=4*9=36. Very fast.
            current_max = 0
            for rr in range(self.n_rows):
                if self.rows[rr][-1] > current_max:
                    current_max = self.rows[rr][-1]
            
            # Apply counts
            for d in old_diffs:
                self.diff_counts[d] -= 1
                if self.diff_counts[d] == 0:
                    del self.diff_counts[d]
            for d in new_diffs:
                self.diff_counts[d] = self.diff_counts.get(d, 0) + 1
            
            self.conflicts += delta_conflicts
            self.max_entry = current_max
            row[pos] = old_val

    # Simulated Annealing
    def simulated_annealing(rows, time_budget):
        state = RobustState(rows)
        current_cost = state.cost()
        best_state = RobustState(rows)
        best_cost = best_state.cost()
        
        T_start = 500.0
        T_end = 1.0
        steps = 0
        max_steps = 500000 # Higher limit due to speedup
        
        while time.time() < time_budget and steps < max_steps:
            steps += 1
            T = T_start * ((T_end / T_start) ** (steps / max_steps))
            if T < T_end:
                T = T_end
            
            # Pick random move
            r = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            old_val = state.rows[r][pos]
            
            # Determine range for new_val
            min_val = state.rows[r][pos - 1] + 1
            max_val = state.rows[r][pos + 1] - 1 if pos < k else old_val + 20
            
            if min_val > max_val:
                continue
                
            # Randomly choose a value
            # Bias towards smaller values to minimize scope
            new_val = random.randint(min_val, max_val)
            
            delta, old_val_saved = state.try_move(r, pos, new_val)
            if delta is None:
                continue
            
            new_cost = current_cost + delta
            
            if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_state = RobustState(state.rows)
            else:
                state.undo_move(r, pos, new_val, old_val_saved)

        return best_state.rows

    # Main execution
    greedy_rows = greedy_init("pressure")
    
    # Run SA
    sa_rows = simulated_annealing(greedy_rows, time.time() + 100)
    
    # Final validation and cleanup
    for r in range(n_rows):
        for m in range(1, n_marks):
            if sa_rows[r][m] <= sa_rows[r][m - 1]:
                sa_rows[r][m] = sa_rows[r][m - 1] + 1
                
    return sa_rows

