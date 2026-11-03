import time
import random
import copy
import math

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0,
    minimizing the maximum entry (scope) such that all pairwise differences
    are distinct across all rows.
    """
    start_time = time.time()
    # Leave 5 seconds margin for safety
    deadline = start_time + 115.0
    
    row_len = k + 1
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    # Number of differences per row
    n_diffs_per_row = k * (k + 1) // 2
    
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
    
    # Get baseline
    baseline_rows = greedy_construct()
    
    # SAState class from inspiration, adapted for speed
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
        
        def get_score(self):
            # Lexicographic minimization: collisions first, then max_val
            # Weight collisions heavily so they are minimized first
            return self.collisions * 10000 + self.max_val

        def try_move(self, r_idx, i_idx, new_val):
            """
            Attempts to change rows[r_idx][i_idx] to new_val.
            Returns (accepted, old_val, delta_score)
            """
            old_val = self.rows[r_idx][i_idx]
            if old_val == new_val:
                return False, old_val, 0
            
            # Check bounds
            if i_idx > 0 and new_val <= self.rows[r_idx][i_idx-1]:
                return False, old_val, 0
            if i_idx < self.row_len - 1 and new_val >= self.rows[r_idx][i_idx+1]:
                return False, old_val, 0
            
            # Remove old differences
            old_row = self.rows[r_idx]
            old_diffs = []
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(old_val - old_row[j])
                    old_diffs.append(d)
            
            old_collisions = self.collisions
            old_max = self.max_val
            old_score = self.get_score()
            
            for d in old_diffs:
                self.diff_count[d] -= 1
                if self.diff_count[d] == 1:
                    self.collisions -= 1
                if self.diff_count[d] == 0:
                    del self.diff_count[d]
            
            # Apply change
            self.rows[r_idx][i_idx] = new_val
            
            # Add new differences
            new_row = self.rows[r_idx]
            new_diffs = []
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(new_val - new_row[j])
                    new_diffs.append(d)
            
            for d in new_diffs:
                if d in self.diff_count:
                    self.diff_count[d] += 1
                    if self.diff_count[d] == 2:
                        self.collisions += 1
                else:
                    self.diff_count[d] = 1
            
            # Update max_val
            if new_val > self.max_val:
                self.max_val = new_val
            elif old_val == old_max and new_val < old_max:
                # Need to recompute max
                self.max_val = max(max(r) for r in self.rows)
            
            new_score = self.get_score()
            delta_score = new_score - old_score
            
            return True, old_val, delta_score

        def undo_move(self, r_idx, i_idx, old_val):
            """Undo a move by restoring old_val."""
            new_val = self.rows[r_idx][i_idx]
            if new_val == old_val:
                return
            
            # Remove new differences
            new_row = self.rows[r_idx]
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(new_val - new_row[j])
                    self.diff_count[d] -= 1
                    if self.diff_count[d] == 1:
                        self.collisions -= 1
                    if self.diff_count[d] == 0:
                        del self.diff_count[d]
            
            # Restore old value
            self.rows[r_idx][i_idx] = old_val
            
            # Add old differences
            old_row = self.rows[r_idx]
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(old_val - old_row[j])
                    if d in self.diff_count:
                        self.diff_count[d] += 1
                        if self.diff_count[d] == 2:
                            self.collisions += 1
                    else:
                        self.diff_count[d] = 1
            
            # Update max_val
            if old_val > self.max_val:
                self.max_val = old_val
            elif new_val == self.max_val and old_val < new_val:
                self.max_val = max(max(r) for r in self.rows)

    def simulate_annealing(initial_rows, time_limit):
        state = SAState(initial_rows)
        current_score = state.get_score()
        best_score = current_score
        best_rows = [r[:] for r in state.rows]
        
        T = 50.0
        cooling = 0.999
        start = time.time()
        
        while time.time() - start < time_limit:
            # Choose a random row and position
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)  # Don't change the 0
            
            # Determine range for new value
            lower = state.rows[r_idx][i_idx - 1] + 1
            if i_idx < row_len - 1:
                upper = state.rows[r_idx][i_idx + 1] - 1
            else:
                # Allow some expansion for the last element
                upper = state.max_val + 10 
                
            if lower > upper:
                continue
            
            # Try a few random values
            for _ in range(2):
                new_val = random.randint(lower, upper)
                accepted, old_val, delta = state.try_move(r_idx, i_idx, new_val)
                
                if not accepted:
                    continue
                
                # Metropolis criterion
                if delta <= 0:
                    # Accept if score improves or stays same
                    # If score improves, update best
                    if state.get_score() < best_score:
                        best_score = state.get_score()
                        best_rows = [r[:] for r in state.rows]
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < math.exp(-delta / T):
                        # Accepted, but doesn't improve best necessarily
                        pass
                    else:
                        # Reject move
                        state.undo_move(r_idx, i_idx, old_val)
                        continue
                
                # If we accepted, we might want to break out of the inner loop
                # to keep the pace, or continue trying more moves on this position
                # Let's just break to keep it simple and fast
                break
                
            T *= cooling
            if T < 0.01:
                # Restart temperature to allow for larger jumps if stuck
                T = 5.0
                
    # Initial SA run
    rows = simulate_annealing(baseline_rows, deadline - time.time())
    
    # Check if we have time for a second phase or restart
    # If the first run didn't find a valid solution (collisions > 0), 
    # we might need to try harder.
    # However, the greedy start is usually valid or close.
    # Let's check the best_rows found
    # We need to return the best valid rows.
    # If best_rows has collisions, we should probably return the greedy one if it was valid,
    # or the one with least collisions? No, the task requires VALIDITY.
    # "Valid when all positive differences ... are distinct".
    # If we return invalid, score is 0.
    
    # So we must ensure we return a VALID set.
    # The SA tracks best_score which includes collisions.
    # If best_score has collisions > 0, we need a fallback.
    
    # Let's re-evaluate the best_rows validity
    # Actually, the greedy construct IS valid.
    # So we can always return the greedy construct if SA fails to find a valid one.
    # But SA usually finds valid ones quickly.
    
    # To be safe, let's check the best_rows from SA.
    # We need to verify validity of best_rows explicitly before returning.
    
    def is_valid(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    # The SA best_rows might have collisions if the process didn't converge to 0 collisions.
    # We should compare the SA result with the greedy result.
    # Greedy is always valid.
    # SA result is valid only if collisions == 0.
    
    sa_valid = is_valid(best_rows)
    greedy_valid = is_valid(baseline_rows)
    
    if sa_valid:
        # Compare scopes
        sa_scope = max(max(r) for r in best_rows)
        greedy_scope = max(max(r) for r in baseline_rows)
        if sa_scope <= greedy_scope:
            return best_rows
        else:
            return baseline_rows
    else:
        # SA failed to find a valid solution, return greedy
        return baseline_rows

    # Wait, the simulate_annealing function returns `rows` which is the current state at the end,
    # but inside it tracks `best_rows`. I need to return `best_rows` from SA.
    
    # Let's fix the SA function to return the best valid rows found.
    
    # Revised plan:
    # 1. Greedy init (Valid)
    # 2. SA optimizes (collisions, max_val)
    # 3. Return best valid found (either from SA or Greedy)

# Re-implementing with the fix

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    row_len = k + 1
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
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
        
        def get_score(self):
            return self.collisions * 10000 + self.max_val

        def try_move(self, r_idx, i_idx, new_val):
            old_val = self.rows[r_idx][i_idx]
            if old_val == new_val:
                return False, old_val, 0
            
            if i_idx > 0 and new_val <= self.rows[r_idx][i_idx-1]:
                return False, old_val, 0
            if i_idx < self.row_len - 1 and new_val >= self.rows[r_idx][i_idx+1]:
                return False, old_val, 0
            
            old_row = self.rows[r_idx]
            old_diffs = []
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(old_val - old_row[j])
                    old_diffs.append(d)
            
            old_score = self.get_score()
            
            for d in old_diffs:
                self.diff_count[d] -= 1
                if self.diff_count[d] == 1:
                    self.collisions -= 1
                if self.diff_count[d] == 0:
                    del self.diff_count[d]
            
            self.rows[r_idx][i_idx] = new_val
            
            new_row = self.rows[r_idx]
            new_diffs = []
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(new_val - new_row[j])
                    new_diffs.append(d)
            
            for d in new_diffs:
                if d in self.diff_count:
                    self.diff_count[d] += 1
                    if self.diff_count[d] == 2:
                        self.collisions += 1
                else:
                    self.diff_count[d] = 1
            
            if new_val > self.max_val:
                self.max_val = new_val
            elif old_val == self.max_val and new_val < old_val:
                self.max_val = max(max(r) for r in self.rows)
            
            new_score = self.get_score()
            delta_score = new_score - old_score
            
            return True, old_val, delta_score

        def undo_move(self, r_idx, i_idx, old_val):
            new_val = self.rows[r_idx][i_idx]
            if new_val == old_val:
                return
            
            new_row = self.rows[r_idx]
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(new_val - new_row[j])
                    self.diff_count[d] -= 1
                    if self.diff_count[d] == 1:
                        self.collisions -= 1
                    if self.diff_count[d] == 0:
                        del self.diff_count[d]
            
            self.rows[r_idx][i_idx] = old_val
            
            old_row = self.rows[r_idx]
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(old_val - old_row[j])
                    if d in self.diff_count:
                        self.diff_count[d] += 1
                        if self.diff_count[d] == 2:
                            self.collisions += 1
                    else:
                        self.diff_count[d] = 1
            
            if old_val > self.max_val:
                self.max_val = old_val
            elif new_val == self.max_val and old_val < new_val:
                self.max_val = max(max(r) for r in self.rows)

    def simulate_annealing(initial_rows, time_limit):
        state = SAState(initial_rows)
        current_score = state.get_score()
        best_score = current_score
        best_rows = [r[:] for r in state.rows]
        
        T = 50.0
        cooling = 0.999
        start = time.time()
        
        while time.time() - start < time_limit:
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, row_len - 1)
            
            lower = state.rows[r_idx][i_idx - 1] + 1
            if i_idx < row_len - 1:
                upper = state.rows[r_idx][i_idx + 1] - 1
            else:
                upper = state.max_val + 10 
                
            if lower > upper:
                continue
            
            for _ in range(2):
                new_val = random.randint(lower, upper)
                accepted, old_val, delta = state.try_move(r_idx, i_idx, new_val)
                
                if not accepted:
                    continue
                
                if delta <= 0:
                    if state.get_score() < best_score:
                        best_score = state.get_score()
                        best_rows = [r[:] for r in state.rows]
                else:
                    if random.random() < math.exp(-delta / T):
                        pass
                    else:
                        state.undo_move(r_idx, i_idx, old_val)
                        continue
                break
                
            T *= cooling
            if T < 0.01:
                T = 5.0
                
        return best_rows, best_score

    # Run SA
    time_left = deadline - time.time()
    if time_left > 1.0:
        sa_rows, sa_score = simulate_annealing(baseline_rows, time_left)
    else:
        sa_rows = baseline_rows
        sa_score = max(max(r) for r in baseline_rows) # Approx

    # Check validity of SA result
    def is_valid(rows):
        used = set()
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    sa_valid = is_valid(sa_rows)
    greedy_valid = is_valid(baseline_rows)
    
    if sa_valid:
        sa_scope = max(max(r) for r in sa_rows)
        greedy_scope = max(max(r) for r in baseline_rows)
        if sa_scope <= greedy_scope:
            return sa_rows
        else:
            return baseline_rows
    else:
        return baseline_rows
