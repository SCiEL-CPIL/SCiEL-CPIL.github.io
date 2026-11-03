import time
import random
import math
import sys
from collections import defaultdict

def construct(n_rows, k):
    row_len = k + 1
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Number of differences per row
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    
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
    best_rows = [r[:] for r in baseline_rows]
    best_max = max(max(r) for r in best_rows)
    
    # Function to compute cost: (collisions, max_val)
    # We want to minimize collisions first, then max_val
    def compute_cost(rows):
        used = defaultdict(int)
        max_val = 0
        for r in rows:
            if r[-1] > max_val:
                max_val = r[-1]
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    used[d] += 1
        
        collisions = 0
        for cnt in used.values():
            if cnt > 1:
                collisions += cnt - 1
        
        return collisions, max_val
    
    # Fast cost update for single element change
    # We maintain the difference counts incrementally
    class SAState:
        def __init__(self, rows):
            self.rows = [r[:] for r in rows]
            self.n_rows = len(rows)
            self.row_len = len(rows[0])
            self.diff_count = defaultdict(int)
            self.collisions = 0
            self.max_val = 0
            self._rebuild()
        
        def _rebuild(self):
            self.diff_count = defaultdict(int)
            self.collisions = 0
            self.max_val = 0
            for r in self.rows:
                if r[-1] > self.max_val:
                    self.max_val = r[-1]
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d = r[j] - r[i]
                        self.diff_count[d] += 1
                        if self.diff_count[d] == 2:
                            self.collisions += 1
        
        def get_cost(self):
            return self.collisions, self.max_val
        
        def is_valid(self):
            return self.collisions == 0
        
        def try_move(self, r_idx, i_idx, new_val):
            """Try to change rows[r_idx][i_idx] to new_val. Return True if accepted."""
            old_val = self.rows[r_idx][i_idx]
            if old_val == new_val:
                return False
            
            # Check bounds
            if i_idx > 0 and new_val <= self.rows[r_idx][i_idx-1]:
                return False
            if i_idx < self.row_len - 1 and new_val >= self.rows[r_idx][i_idx+1]:
                return False
            
            # Remove old differences
            old_row = self.rows[r_idx]
            old_diffs = []
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(old_val - old_row[j])
                    old_diffs.append(d)
            
            for d in old_diffs:
                self.diff_count[d] -= 1
                if self.diff_count[d] == 1:
                    self.collisions -= 1
                if self.diff_count[d] == 0:
                    del self.diff_count[d]
            
            # Apply change
            old_max = self.max_val
            self.rows[r_idx][i_idx] = new_val
            
            # Add new differences
            new_row = self.rows[r_idx]
            new_diffs = []
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(new_val - new_row[j])
                    new_diffs.append(d)
            
            for d in new_diffs:
                self.diff_count[d] += 1
                if self.diff_count[d] == 2:
                    self.collisions += 1
            
            # Update max_val
            if new_val > self.max_val:
                self.max_val = new_val
            elif old_val == old_max and new_val < old_max:
                # Need to recompute max
                self.max_val = max(max(r) for r in self.rows)
            
            return True
        
        def undo_move(self, r_idx, i_idx, old_val):
            """Undo a move by restoring old_val."""
            new_val = self.rows[r_idx][i_idx]
            if new_val == old_val:
                return
            
            # Remove new differences
            new_row = self.rows[r_idx]
            new_diffs = []
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(new_val - new_row[j])
                    new_diffs.append(d)
            
            for d in new_diffs:
                self.diff_count[d] -= 1
                if self.diff_count[d] == 1:
                    self.collisions -= 1
                if self.diff_count[d] == 0:
                    del self.diff_count[d]
            
            # Restore old value
            self.rows[r_idx][i_idx] = old_val
            
            # Add old differences
            old_row = self.rows[r_idx]
            old_diffs = []
            for j in range(self.row_len):
                if j != i_idx:
                    d = abs(old_val - old_row[j])
                    old_diffs.append(d)
            
            for d in old_diffs:
                self.diff_count[d] += 1
                if self.diff_count[d] == 2:
                    self.collisions += 1
            
            # Update max_val
            if old_val > self.max_val:
                self.max_val = old_val
            elif new_val == self.max_val and old_val < new_val:
                self.max_val = max(max(r) for r in self.rows)
    
    def local_polish(rows, time_limit):
        """Local search to reduce max value while maintaining validity."""
        state = SAState(rows)
        current_collisions, current_max = state.get_cost()
        best_collisions = current_collisions
        best_max = current_max
        best_rows = [r[:] for r in state.rows]
        
        start = time.time()
        
        while time.time() - start < time_limit:
            if state.collisions > 0:
                # Need to fix collisions first
                r_idx = random.randint(0, n_rows - 1)
                i_idx = random.randint(1, row_len - 1)
                lower = state.rows[r_idx][i_idx - 1] + 1
                if i_idx < row_len - 1:
                    upper = state.rows[r_idx][i_idx + 1] - 1
                else:
                    upper = best_max + 5
                
                if lower > upper:
                    continue
                
                new_val = random.randint(lower, upper)
                old_val = state.rows[r_idx][i_idx]
                if new_val == old_val:
                    continue
                
                if not state.try_move(r_idx, i_idx, new_val):
                    continue
                
                if state.collisions < best_collisions or (state.collisions == best_collisions and state.max_val < best_max):
                    best_collisions = state.collisions
                    best_max = state.max_val
                    best_rows = [r[:] for r in state.rows]
                else:
                    state.undo_move(r_idx, i_idx, old_val)
            else:
                # Valid solution - try to reduce max
                # Find the row with the maximum value
                max_row_idx = max(range(n_rows), key=lambda i: state.rows[i][-1])
                max_val_in_row = state.rows[max_row_idx][-1]
                
                if max_val_in_row <= best_max:
                    # Already at or below best, try other rows
                    r_idx = random.randint(0, n_rows - 1)
                else:
                    r_idx = max_row_idx
                
                i_idx = random.randint(1, row_len - 1)
                lower = state.rows[r_idx][i_idx - 1] + 1
                if i_idx < row_len - 1:
                    upper = state.rows[r_idx][i_idx + 1] - 1
                else:
                    upper = state.rows[r_idx][i_idx] - 1  # Try to reduce
                
                if lower > upper:
                    continue
                
                # Try values that might reduce the max
                old_val = state.rows[r_idx][i_idx]
                for _ in range(5):
                    if i_idx == row_len - 1:
                        # Try to reduce the last element
                        new_val = random.randint(lower, min(upper, old_val - 1))
                    else:
                        new_val = random.randint(lower, upper)
                    
                    if new_val == old_val:
                        continue
                    
                    if not state.try_move(r_idx, i_idx, new_val):
                        continue
                    
                    if state.collisions == 0 and state.max_val < best_max:
                        best_max = state.max_val
                        best_rows = [r[:] for r in state.rows]
                    else:
                        state.undo_move(r_idx, i_idx, old_val)
        
        return best_rows, best_max, best_collisions
    
    def simulate_annealing(initial_rows, time_limit):
        state = SAState(initial_rows)
        current_collisions, current_max = state.get_cost()
        best_collisions = current_collisions
        best_max = current_max
        best_rows = [r[:] for r in state.rows]
        
        T = 50.0
        cooling = 0.9995
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
                upper = best_max + 10  # Allow some expansion
            
            if lower > upper:
                continue
            
            # Try a few random values
            for _ in range(3):
                new_val = random.randint(lower, upper)
                old_val = state.rows[r_idx][i_idx]
                if new_val == old_val:
                    continue
                
                if not state.try_move(r_idx, i_idx, new_val):
                    continue
                
                new_collisions, new_max = state.get_cost()
                
                # Acceptance criterion
                if new_collisions < current_collisions:
                    accept = True
                elif new_collisions == current_collisions:
                    if new_max < current_max:
                        accept = True
                    else:
                        # Accept with some probability to escape local minima
                        delta = new_max - current_max
                        if delta < 0:
                            accept = True
                        else:
                            p = math.exp(-delta / max(T, 0.1))
                            accept = random.random() < p
                else:
                    # More collisions - rarely accept
                    delta = (new_collisions - current_collisions) * 100 + (new_max - current_max)
                    if delta < 0:
                        accept = True
                    else:
                        p = math.exp(-delta / max(T, 0.1))
                        accept = random.random() < p
                
                if accept:
                    current_collisions = new_collisions
                    current_max = new_max
                    if current_collisions < best_collisions or (current_collisions == best_collisions and current_max < best_max):
                        best_collisions = current_collisions
                        best_max = current_max
                        best_rows = [r[:] for r in state.rows]
                        if best_collisions == 0:
                            # Found a valid solution
                            # Try to reduce max further
                            pass
                else:
                    state.undo_move(r_idx, i_idx, old_val)
            
            T *= cooling
            if T < 0.5:
                T = 0.5
        
        return best_rows, best_max, best_collisions
    
    # Main loop
    while time.time() < deadline:
        # Create initial state
        if random.random() < 0.3:
            # Use best rows as starting point
            init_rows = [r[:] for r in best_rows]
        else:
            # Random initialization within a bound
            bound = best_max + 5
            init_rows = []
            for _ in range(n_rows):
                pts = random.sample(range(1, bound + 1), k)
                pts.sort()
                init_rows.append([0] + pts)
        
        # Run SA for a fixed time
        sa_time = min(6.0, deadline - time.time())
        if sa_time < 1.0:
            break
        
        local_rows, local_max, local_collisions = simulate_annealing(init_rows, sa_time)
        
        if local_collisions < compute_cost(best_rows)[0] or \
           (local_collisions == 0 and local_max < best_max):
            best_rows = local_rows
            best_max = local_max
        
        # If we have a valid solution, try to polish it
        if best_rows is not None and compute_cost(best_rows)[0] == 0 and time.time() < deadline - 5.0:
            polish_time = min(3.0, deadline - time.time())
            polished_rows, polished_max, _ = local_polish(best_rows, polish_time)
            if polished_max < best_max:
                best_rows = polished_rows
                best_max = polished_max
    
    return best_rows
