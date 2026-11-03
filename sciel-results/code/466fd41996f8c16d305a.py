import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Number of differences per row
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row

    # --- State Management for Efficient Cost Updates ---
    class SAState:
        def __init__(self, rows):
            self.rows = [r[:] for r in rows]
            self.n_rows = len(rows)
            self.row_len = len(rows[0])
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            self._rebuild()
        
        def _rebuild(self):
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            for r in self.rows:
                if r[-1] > self.max_val:
                    self.max_val = r[-1]
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d = r[j] - r[i]
                        if d in self.diff_count:
                            self.diff_count[d] += 1
                            if self.diff_count[d] == 2:
                                self.collisions += 1
                        else:
                            self.diff_count[d] = 1
        
        def get_cost(self):
            return self.collisions, self.max_val
        
        def is_valid(self):
            return self.collisions == 0
            
        def try_move(self, r_idx, i_idx, new_val):
            """
            Attempt to change rows[r_idx][i_idx] to new_val.
            Returns (old_val, delta_collisions, delta_max) if successful, else None.
            """
            old_val = self.rows[r_idx][i_idx]
            if old_val == new_val:
                return None
            
            # Check bounds
            if i_idx > 0 and new_val <= self.rows[r_idx][i_idx-1]:
                return None
            if i_idx < self.row_len - 1 and new_val >= self.rows[r_idx][i_idx+1]:
                return None
            
            # Calculate delta in collisions and max
            old_row = self.rows[r_idx]
            delta_collisions = 0
            delta_max = 0
            old_max = self.max_val
            
            # Identify affected differences
            # Differences involving index i_idx
            affected_diffs = []
            for j in range(self.row_len):
                if j != i_idx:
                    d_old = abs(old_val - old_row[j])
                    d_new = abs(new_val - old_row[j])
                    affected_diffs.append((d_old, d_new))
            
            # Check if move is feasible regarding collisions
            # We need to ensure that removing old diffs and adding new diffs doesn't create new collisions
            # or resolves existing ones.
            
            # Simulate removal of old diffs
            # We need to know how many times each diff appears.
            # If a diff appears 1 time, removing it reduces count to 0 (no collision change unless it was part of a collision, which is impossible if count 1).
            # If a diff appears 2 times, removing it reduces count to 1 (resolves 1 collision).
            # If a diff appears > 2 times, removing it reduces count by 1 (reduces collisions by 1).
            
            # Simulate addition of new diffs
            # If a diff appears 0 times (after removal), adding it makes 1 (no collision).
            # If a diff appears 1 time (after removal), adding it makes 2 (creates 1 collision).
            # If a diff appears > 1 times (after removal), adding it makes >2 (creates 1 collision).
            
            # We need to be careful: if d_old == d_new, the net effect is 0.
            
            # Let's use a temporary dictionary for the changes
            changes = {}
            for d_old, d_new in affected_diffs:
                if d_old == d_new:
                    continue
                # Track removals
                if d_old not in changes:
                    changes[d_old] = 0
                changes[d_old] -= 1
                # Track additions
                if d_new not in changes:
                    changes[d_new] = 0
                changes[d_new] += 1
            
            # Calculate collision delta
            for d, delta in changes.items():
                current_count = self.diff_count.get(d, 0)
                new_count = current_count + delta
                
                if new_count < 0:
                    # Should not happen if logic is correct, but safe guard
                    return None
                
                # Collision count contribution for a diff with count c is max(0, c - 1)
                old_contrib = max(0, current_count - 1)
                new_contrib = max(0, new_count - 1)
                delta_collisions += (new_contrib - old_contrib)
            
            # Check if any new diff would exceed available space? 
            # The problem doesn't have a hard bound on max_val initially, but we want to minimize it.
            # However, we must ensure the row remains strictly increasing.
            # The bounds check above ensures strict increase.
            
            # Calculate max delta
            if new_val > self.max_val:
                delta_max = new_val - self.max_val
            elif old_val == self.max_val and new_val < self.max_val:
                # Need to find new max
                # This is expensive. We can approximate or just return 0 and let _rebuild handle it if accepted?
                # No, we need accurate cost for SA.
                # Let's compute max of other elements in row and other rows.
                # Optimization: keep a sorted list of maxes? Too complex.
                # For now, if old_val was max, we assume max stays same or decreases.
                # Actually, if old_val was the unique max, max decreases.
                # Let's just set delta_max to 0 for now if it's not increasing, and rely on the fact that 
                # usually we are trying to reduce collisions first. 
                # Wait, the cost is (collisions, max_val). If collisions are 0, we minimize max_val.
                # If we reduce max_val, cost improves.
                # Let's compute new max accurately only if old_val == self.max_val.
                if old_val == self.max_val:
                    new_max_candidate = max(max(r) for r in self.rows) # This is O(N*K), expensive.
                    # Better: max of (current max of other rows, new max of this row)
                    # We don't store max per row.
                    # Let's just set delta_max = 0 if not increasing. 
                    # This might lead to suboptimal max reduction, but collisions are priority 1.
                    # Actually, if we are in a valid state (collisions=0), we want to minimize max.
                    # So we need accurate max.
                    # Let's optimize: store max_per_row.
                    pass # We'll implement max_per_row in the class if needed. For now, assume we don't reduce max in a move that doesn't increase it? No, that's wrong.
                    
            # If delta_collisions is positive, we are creating collisions.
            # If delta_collisions is negative, we are resolving them.
            
            # Apply the move
            self.rows[r_idx][i_idx] = new_val
            
            # Update diff_count safely
            for d_old, d_new in affected_diffs:
                if d_old == d_new:
                    continue
                # Decrement old
                if d_old in self.diff_count:
                    self.diff_count[d_old] -= 1
                    if self.diff_count[d_old] == 0:
                        del self.diff_count[d_old]
                else:
                    # State inconsistency, but avoid crash
                    pass 
                # Increment new
                if d_new in self.diff_count:
                    self.diff_count[d_new] += 1
                else:
                    self.diff_count[d_new] = 1
            
            self.collisions += delta_collisions
            
            # Update max_val
            if new_val > self.max_val:
                self.max_val = new_val
            elif old_val == self.max_val:
                # Recompute max
                self.max_val = max(max(r) for r in self.rows)
            
            return old_val, delta_collisions, (self.max_val - old_max)

        def undo_move(self, r_idx, i_idx, old_val, delta_collisions, delta_max):
            """Undo a move."""
            new_val = self.rows[r_idx][i_idx]
            
            # Revert diff_count safely
            old_row = self.rows[r_idx]
            for j in range(self.row_len):
                if j != i_idx:
                    d_new = abs(new_val - old_row[j])
                    d_old = abs(old_val - old_row[j])
                    if d_new == d_old:
                        continue
                    # Decrement new (current in state)
                    if d_new in self.diff_count:
                        self.diff_count[d_new] -= 1
                        if self.diff_count[d_new] == 0:
                            del self.diff_count[d_new]
                    # Increment old (target in state)
                    if d_old in self.diff_count:
                        self.diff_count[d_old] += 1
                    else:
                        self.diff_count[d_old] = 1
            
            self.rows[r_idx][i_idx] = old_val
            self.collisions -= delta_collisions
            # Recompute max_val to be safe, as delta_max logic is complex
            self.max_val = max(max(r) for r in self.rows)

    # --- Initialization ---
    
    def greedy_construct():
        """Fast greedy construction to get a baseline."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                # Check if x can be added
                valid = True
                for a in row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used.add(x - a)
            rows.append(row)
        return rows
    
    baseline_rows = greedy_construct()
    best_rows = [r[:] for r in baseline_rows]
    best_state = SAState(baseline_rows)
    best_collisions, best_max = best_state.get_cost()
    
    # If greedy already found a valid solution, we can try to optimize it.
    # If not, we need to repair.
    
    # --- Optimization Loop ---
    
    # Strategy:
    # 1. If collisions > 0, use a repair strategy.
    # 2. If collisions == 0, use SA to minimize max_val.
    
    current_state = SAState(baseline_rows)
    current_collisions, current_max = current_state.get_cost()
    
    T = 100.0
    cooling = 0.999
    last_improvement_time = start_time
    
    while time.time() < deadline:
        if time.time() - last_improvement_time > 10.0:
            # Stagnation: Restart with a new random initialization or perturb current
            # Perturb current state
            for _ in range(n_rows * row_len):
                r_idx = random.randint(0, n_rows - 1)
                i_idx = random.randint(1, row_len - 1)
                # Random move
                lower = current_state.rows[r_idx][i_idx - 1] + 1
                upper = current_state.rows[r_idx][i_idx + 1] - 1 if i_idx < row_len - 1 else current_max + 10
                if lower < upper:
                    new_val = random.randint(lower, upper)
                    res = current_state.try_move(r_idx, i_idx, new_val)
                    if res:
                        # Accept if it doesn't increase collisions too much
                        _, d_col, _ = res
                        if d_col <= 0 or random.random() < math.exp(-d_col * 10 / T):
                            pass # Already applied
                        else:
                            current_state.undo_move(r_idx, i_idx, res[0], res[1], res[2])
            last_improvement_time = time.time()
            T = max(T, 50.0)
        
        # Choose move
        r_idx = random.randint(0, n_rows - 1)
        i_idx = random.randint(1, row_len - 1)
        
        # Determine range
        lower = current_state.rows[r_idx][i_idx - 1] + 1
        if i_idx < row_len - 1:
            upper = current_state.rows[r_idx][i_idx + 1] - 1
        else:
            upper = current_max + 5 # Allow small expansion
        
        if lower > upper:
            continue
            
        # Try a few random values
        move_made = False
        for _ in range(5):
            new_val = random.randint(lower, upper)
            if new_val == current_state.rows[r_idx][i_idx]:
                continue
                
            res = current_state.try_move(r_idx, i_idx, new_val)
            if res is None:
                continue
                
            old_val, d_col, d_max = res
            new_collisions = current_collisions + d_col
            new_max = current_max + d_max
            
            # Acceptance Criteria
            # Priority 1: Reduce collisions
            # Priority 2: Reduce max_val (if collisions == 0)
            
            accepted = False
            if new_collisions < current_collisions:
                accepted = True
            elif new_collisions == current_collisions:
                if new_max < current_max:
                    accepted = True
                elif new_max == current_max:
                    accepted = random.random() < 0.5 # Neutral move
                else:
                    # Worse max, but same collisions. Accept with probability
                    if current_collisions == 0:
                        # We are in valid region, try to keep max low
                        delta = new_max - current_max
                        if delta < 10:
                            accepted = random.random() < math.exp(-delta / T)
                    else:
                        # Not valid yet, allow some max increase to escape local minima
                        delta = new_max - current_max
                        if delta < 50:
                            accepted = random.random() < math.exp(-delta / (T * 2))
            else:
                # More collisions. Accept with probability based on T
                delta = new_collisions - current_collisions
                # Penalize collisions heavily
                penalty = delta * 100
                if penalty < 1000:
                    accepted = random.random() < math.exp(-penalty / T)
            
            if accepted:
                current_collisions = new_collisions
                current_max = new_max
                move_made = True
                # Update best if improved
                if (current_collisions, current_max) < (best_collisions, best_max):
                    best_collisions = current_collisions
                    best_max = current_max
                    best_rows = [r[:] for r in current_state.rows]
                    last_improvement_time = time.time()
                break
            else:
                # Undo move
                current_state.undo_move(r_idx, i_idx, old_val, d_col, d_max)
        
        # Cooling
        T *= cooling
        if T < 1.0:
            T = 1.0

    # Return the best valid solution found
    # If no valid solution was found (best_collisions > 0), we should return the one with fewest collisions
    # But the task requires a valid solution. 
    # The greedy baseline is always valid. 
    # Wait, does greedy_construct always produce a valid solution?
    # Yes, because it checks `if d in used`. So `baseline_rows` is always valid.
    # So `best_rows` will always be valid if we initialize best with baseline.
    
    return best_rows

# Quick self-test
if __name__ == "__main__":
    # Test with small values
    rows = construct(7, 5)
    # Verify
    diffs = set()
    valid = True
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in diffs:
                    valid = False
                diffs.add(d)
    max_val = max(max(r) for r in rows)
    print(f"Valid: {valid}, Max: {max_val}")
