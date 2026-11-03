import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n_diffs_per_row = k * (k + 1) // 2
    
    # --- Efficient State ---
    class SAState:
        __slots__ = ['rows', 'n_rows', 'row_len', 'diff_count', 'collisions', 
                     'max_val', 'row_max']
        
        def __init__(self, rows):
            self.rows = [r[:] for r in rows]
            self.n_rows = len(rows)
            self.row_len = len(rows[0])
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            self.row_max = [0] * self.n_rows
            self._rebuild()
        
        def _rebuild(self):
            self.diff_count = {}
            self.collisions = 0
            self.max_val = 0
            self.row_max = [0] * self.n_rows
            for i, r in enumerate(self.rows):
                self.row_max[i] = r[-1]
                if r[-1] > self.max_val:
                    self.max_val = r[-1]
                for a in range(self.row_len):
                    for b in range(a+1, self.row_len):
                        d = r[b] - r[a]
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
            old_val = self.rows[r_idx][i_idx]
            if old_val == new_val:
                return None
            
            row = self.rows[r_idx]
            if i_idx > 0 and new_val <= row[i_idx-1]:
                return None
            if i_idx < self.row_len - 1 and new_val >= row[i_idx+1]:
                return None
            
            # Compute affected differences
            affected = []
            for j in range(self.row_len):
                if j != i_idx:
                    d_old = abs(old_val - row[j])
                    d_new = abs(new_val - row[j])
                    if d_old != d_new:
                        affected.append((d_old, d_new))
            
            # Compute collision delta
            # Group by difference value
            changes = {}
            for d_old, d_new in affected:
                changes[d_old] = changes.get(d_old, 0) - 1
                changes[d_new] = changes.get(d_new, 0) + 1
            
            delta_collisions = 0
            for d, delta in changes.items():
                cur = self.diff_count.get(d, 0)
                new_c = cur + delta
                if new_c < 0:
                    return None
                old_contrib = max(0, cur - 1)
                new_contrib = max(0, new_c - 1)
                delta_collisions += (new_contrib - old_contrib)
            
            old_max = self.max_val
            
            # Apply move
            self.rows[r_idx][i_idx] = new_val
            
            for d_old, d_new in affected:
                # Remove old
                c = self.diff_count.get(d_old, 0)
                if c == 1:
                    del self.diff_count[d_old]
                else:
                    self.diff_count[d_old] = c - 1
                # Add new
                self.diff_count[d_new] = self.diff_count.get(d_new, 0) + 1
            
            self.collisions += delta_collisions
            
            # Update max
            if i_idx == self.row_len - 1:
                self.row_max[r_idx] = new_val
            # Check if max changed
            if new_val > self.max_val:
                self.max_val = new_val
            elif old_val == self.max_val:
                # Find new max
                if i_idx == self.row_len - 1:
                    # row_max already updated
                    new_max = max(self.row_max)
                else:
                    # old_val was max but not last element
                    # max of this row is still row_max[r_idx]
                    new_max = max(self.row_max)
                self.max_val = new_max
            
            delta_max = self.max_val - old_max
            return old_val, delta_collisions, delta_max
        
        def undo_move(self, r_idx, i_idx, old_val, delta_collisions, delta_max):
            new_val = self.rows[r_idx][i_idx]
            row = self.rows[r_idx]
            
            for j in range(self.row_len):
                if j != i_idx:
                    d_new = abs(new_val - row[j])
                    d_old = abs(old_val - row[j])
                    if d_new != d_old:
                        # Remove new
                        c = self.diff_count.get(d_new, 0)
                        if c == 1:
                            del self.diff_count[d_new]
                        else:
                            self.diff_count[d_new] = c - 1
                        # Add old
                        self.diff_count[d_old] = self.diff_count.get(d_old, 0) + 1
            
            self.rows[r_idx][i_idx] = old_val
            self.collisions -= delta_collisions
            
            if i_idx == self.row_len - 1:
                self.row_max[r_idx] = old_val
            if old_val > self.max_val:
                self.max_val = old_val
            else:
                self.max_val = max(self.row_max)
    
    # --- Initialization: multiple random greedy tries ---
    def greedy_construct(seed):
        rng = random.Random(seed)
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
    
    def random_greedy(seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                # Randomize: sometimes skip valid values
                if rng.random() < 0.1:
                    continue
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
    
    # Try several initializations
    best_rows = None
    best_cost = (float('inf'), float('inf'))
    
    # Deterministic greedy
    rows = greedy_construct(42)
    state = SAState(rows)
    cost = state.get_cost()
    if cost < best_cost:
        best_cost = cost
        best_rows = [r[:] for r in rows]
    
    # Random greedy variants
    for seed in range(20):
        rows = random_greedy(1000 + seed)
        state = SAState(rows)
        cost = state.get_cost()
        if cost < best_cost:
            best_cost = cost
            best_rows = [r[:] for r in rows]
    
    best_collisions, best_max = best_cost
    best_state = SAState(best_rows)
    
    # --- SA Optimization ---
    current_state = SAState(best_rows)
    current_collisions, current_max = current_state.get_cost()
    
    T = 50.0
    T_min = 0.5
    cooling = 0.9995
    last_improvement = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    while elapsed() < deadline - 5.0:
        # Check stagnation
        if time.time() - last_improvement > 15.0:
            # Perturb and reset temperature
            for _ in range(n_rows * row_len * 2):
                r_idx = random.randint(0, n_rows - 1)
                # Bias towards last position (affects max)
                if random.random() < 0.4:
                    i_idx = row_len - 1
                else:
                    i_idx = random.randint(1, row_len - 1)
                
                row = current_state.rows[r_idx]
                lower = row[i_idx - 1] + 1
                if i_idx < row_len - 1:
                    upper = row[i_idx + 1] - 1
                else:
                    upper = current_max + 20
                
                if lower < upper:
                    new_val = random.randint(lower, upper)
                    res = current_state.try_move(r_idx, i_idx, new_val)
                    if res:
                        _, d_col, _ = res
                        if d_col <= 0 or random.random() < math.exp(-d_col * 5 / T):
                            pass
                        else:
                            current_state.undo_move(r_idx, i_idx, res[0], res[1], res[2])
            
            last_improvement = time.time()
            T = max(T, 30.0)
        
        # Choose move - bias towards last element and high-collision rows
        r_idx = random.randint(0, n_rows - 1)
        # Bias: 50% chance to pick last element
        if random.random() < 0.5:
            i_idx = row_len - 1
        else:
            i_idx = random.randint(1, row_len - 1)
        
        row = current_state.rows[r_idx]
        lower = row[i_idx - 1] + 1
        if i_idx < row_len - 1:
            upper = row[i_idx + 1] - 1
        else:
            upper = current_max + 10
        
        if lower > upper:
            T *= cooling
            if T < T_min:
                T = T_min
            continue
        
        # Try multiple candidate values
        move_made = False
        n_attempts = 8 if current_collisions > 0 else 12
        for _ in range(n_attempts):
            # For last element, try values near current or slightly lower
            if i_idx == row_len - 1 and current_collisions == 0:
                # Try to reduce max
                new_val = random.randint(lower, min(upper, current_state.rows[r_idx][i_idx]))
            else:
                new_val = random.randint(lower, upper)
            
            if new_val == current_state.rows[r_idx][i_idx]:
                continue
            
            res = current_state.try_move(r_idx, i_idx, new_val)
            if res is None:
                continue
            
            old_val, d_col, d_max = res
            new_collisions = current_collisions + d_col
            new_max = current_max + d_max
            
            # Acceptance
            accepted = False
            if new_collisions < current_collisions:
                accepted = True
            elif new_collisions == current_collisions:
                if new_max < current_max:
                    accepted = True
                elif new_max == current_max:
                    accepted = random.random() < 0.3
                else:
                    delta = new_max - current_max
                    if current_collisions == 0:
                        if delta <= 5:
                            accepted = random.random() < math.exp(-delta / (T * 0.5))
                    else:
                        if delta <= 30:
                            accepted = random.random() < math.exp(-delta / T)
            else:
                delta = new_collisions - current_collisions
                penalty = delta * 50
                if penalty < 500:
                    accepted = random.random() < math.exp(-penalty / T)
            
            if accepted:
                current_collisions = new_collisions
                current_max = new_max
                move_made = True
                if (current_collisions, current_max) < (best_collisions, best_max):
                    best_collisions = current_collisions
                    best_max = current_max
                    best_rows = [r[:] for r in current_state.rows]
                    last_improvement = time.time()
                break
            else:
                current_state.undo_move(r_idx, i_idx, old_val, d_col, d_max)
        
        # Cooling
        T *= cooling
        if T < T_min:
            T = T_min
    
    # --- Final polish: if valid, try to reduce max further ---
    if best_collisions == 0:
        polish_state = SAState(best_rows)
        polish_collisions, polish_max = polish_state.get_cost()
        
        # Aggressive local search to minimize max
        T_polish = 5.0
        polish_deadline = deadline - 1.0
        no_improve = 0
        
        while time.time() < polish_deadline:
            if no_improve > 5000:
                break
            
            r_idx = random.randint(0, n_rows - 1)
            # Focus on last element
            if random.random() < 0.7:
                i_idx = row_len - 1
            else:
                i_idx = random.randint(1, row_len - 1)
            
            row = polish_state.rows[r_idx]
            lower = row[i_idx - 1] + 1
            if i_idx < row_len - 1:
                upper = row[i_idx + 1] - 1
            else:
                upper = row[i_idx]  # Don't increase max
            
            if lower > upper:
                continue
            
            # Try to find a valid position with lower value
            best_new_val = None
            best_new_max = polish_max
            
            # Sample some values
            for _ in range(15):
                new_val = random.randint(lower, upper)
                if new_val >= row[i_idx]:
                    continue
                
                res = polish_state.try_move(r_idx, i_idx, new_val)
                if res is None:
                    continue
                
                old_val, d_col, d_max = res
                if d_col == 0 and polish_max + d_max < best_new_max:
                    best_new_val = new_val
                    best_new_max = polish_max + d_max
                    # Undo to try more
                    polish_state.undo_move(r_idx, i_idx, old_val, d_col, d_max)
                elif res:
                    polish_state.undo_move(r_idx, i_idx, old_val, d_col, d_max)
            
            if best_new_val is not None:
                res = polish_state.try_move(r_idx, i_idx, best_new_val)
                if res:
                    polish_collisions += res[1]
                    polish_max += res[2]
                    if polish_max < best_max:
                        best_max = polish_max
                        best_rows = [r[:] for r in polish_state.rows]
                    no_improve = 0
                T_polish *= 0.99
            else:
                no_improve += 1
                T_polish *= 0.999
    
    return best_rows

# Quick self-test
if __name__ == "__main__":
    rows = construct(7, 5)
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
    print(f"Rows: {rows}")
