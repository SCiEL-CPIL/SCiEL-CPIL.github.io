import time
import random
import copy
import sys

def construct(n_rows, k):
    row_len = k + 1
    n_diffs_per_row = row_len * (row_len - 1) // 2  # C(k+1, 2)
    total_diffs = n_rows * n_diffs_per_row
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    def get_diffs(row):
        """Compute all positive differences in a row."""
        diffs = []
        for i in range(row_len):
            for j in range(i+1, row_len):
                diffs.append(row[j] - row[i])
        return diffs
    
    def greedy_init():
        """Greedy construction: build rows one by one, picking smallest valid values."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        break
                    x += 1
                row.append(x)
                used.update(new_diffs)
            rows.append(row)
        return rows
    
    def compute_used_set(rows):
        """Compute the set of all differences across all rows."""
        used = set()
        for row in rows:
            for i in range(row_len):
                for j in range(i+1, row_len):
                    used.add(row[j] - row[i])
        return used
    
    def count_conflicts(rows, used_set):
        """Count total conflicts (duplicates) in the used set."""
        # used_set is the set of all diffs; conflicts = total_diffs - len(used_set)
        return total_diffs - len(used_set)
    
    def get_scope(rows):
        """Get the maximum entry across all rows."""
        return max(max(r) for r in rows)
    
    # Phase 1: Greedy initialization
    rows = greedy_init()
    best_rows = copy.deepcopy(rows)
    best_scope = get_scope(best_rows)
    
    # Phase 2: Simulated Annealing with row-level operations
    current_rows = copy.deepcopy(rows)
    current_used = compute_used_set(current_rows)
    current_scope = get_scope(current_rows)
    current_conflicts = count_conflicts(current_rows, current_used)
    
    # Precompute row maxes for fast scope calculation
    row_maxes = [max(r) for r in current_rows]
    
    T = 20.0
    cooling = 0.9995
    min_T = 0.01
    
    # Track best
    best_scope = current_scope
    best_rows = copy.deepcopy(current_rows)
    
    while time.time() < deadline and T > min_T:
        # Choose a random row and a move type
        i = random.randint(0, n_rows - 1)
        move_type = random.random()
        
        old_row = current_rows[i]
        
        if move_type < 0.6:
            # Point move: change one element (not the first, which is 0)
            j = random.randint(1, row_len - 1)
            # Choose a delta
            if random.random() < 0.3:
                delta = random.randint(-10, 10)
            else:
                delta = random.choice([-2, -1, 1, 2])
            
            new_val = old_row[j] + delta
            
            # Bounds check
            if new_val <= old_row[j-1]:
                continue
            if j < row_len - 1 and new_val >= old_row[j+1]:
                continue
            
            # Remove old diffs from used set
            for a in range(row_len):
                for b in range(a+1, row_len):
                    current_used.discard(old_row[b] - old_row[a])
            
            # Create new row
            new_row = old_row.copy()
            new_row[j] = new_val
            
            # Compute new diffs
            new_diffs = get_diffs(new_row)
            
            # Count conflicts
            c_inter = sum(1 for d in new_diffs if d in current_used)
            c_intra = len(new_diffs) - len(set(new_diffs))
            new_conflicts = current_conflicts + c_inter + c_intra - (n_diffs_per_row - len(set(old_diffs := get_diffs(old_row))))
            # Actually, let's recompute more carefully
            # Old conflicts contribution from this row: internal duplicates + overlaps with other rows
            # This is getting complex. Let's use a simpler cost.
            
            new_max_row = max(new_row)
            # Compute new scope
            other_max = max(row_maxes[idx] for idx in range(n_rows) if idx != i) if n_rows > 1 else 0
            new_scope = max(new_max_row, other_max)
            
            # Cost: scope + penalty for conflicts
            penalty = 100
            old_cost = current_scope + penalty * current_conflicts
            new_cost = new_scope + penalty * new_conflicts
            delta_cost = new_cost - old_cost
            
            if delta_cost < 0 or random.random() < pow(2.718281828, -delta_cost / T):
                # Accept
                current_rows[i] = new_row
                row_maxes[i] = new_max_row
                current_used.update(new_diffs)
                current_scope = new_scope
                current_conflicts = new_conflicts
                
                if new_conflicts == 0 and new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = copy.deepcopy(current_rows)
            else:
                # Reject: restore old diffs
                for a in range(row_len):
                    for b in range(a+1, row_len):
                        current_used.add(old_row[b] - old_row[a])
                        
        elif move_type < 0.85:
            # Shift move: shift all elements except the first
            delta = random.choice([-1, 1])
            if old_row[1] + delta <= 0:
                continue
            
            # Remove old diffs
            for a in range(row_len):
                for b in range(a+1, row_len):
                    current_used.discard(old_row[b] - old_row[a])
            
            new_row = [0] + [x + delta for x in old_row[1:]]
            new_diffs = get_diffs(new_row)
            
            c_inter = sum(1 for d in new_diffs if d in current_used)
            c_intra = len(new_diffs) - len(set(new_diffs))
            # Internal conflicts don't change with a uniform shift
            new_conflicts = current_conflicts + c_inter + c_intra - (n_diffs_per_row - len(set(get_diffs(old_row))))
            
            new_max_row = max(new_row)
            other_max = max(row_maxes[idx] for idx in range(n_rows) if idx != i) if n_rows > 1 else 0
            new_scope = max(new_max_row, other_max)
            
            penalty = 100
            old_cost = current_scope + penalty * current_conflicts
            new_cost = new_scope + penalty * new_conflicts
            delta_cost = new_cost - old_cost
            
            if delta_cost < 0 or random.random() < pow(2.718281828, -delta_cost / T):
                current_rows[i] = new_row
                row_maxes[i] = new_max_row
                current_used.update(new_diffs)
                current_scope = new_scope
                current_conflicts = new_conflicts
                
                if new_conflicts == 0 and new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = copy.deepcopy(current_rows)
            else:
                for a in range(row_len):
                    for b in range(a+1, row_len):
                        current_used.add(old_row[b] - old_row[a])
        else:
            # Swap move: swap two elements in a row (maintaining order)
            # Pick two positions and try to swap their values, but keep sorted
            j1 = random.randint(1, row_len - 2)
            j2 = random.randint(j1 + 1, row_len - 1)
            
            # Try swapping values at j1 and j2
            new_row = old_row.copy()
            new_row[j1], new_row[j2] = new_row[j2], new_row[j1]
            
            # Check if still sorted
            if any(new_row[t] >= new_row[t+1] for t in range(row_len - 1)):
                continue
            
            # Remove old diffs
            for a in range(row_len):
                for b in range(a+1, row_len):
                    current_used.discard(old_row[b] - old_row[a])
            
            new_diffs = get_diffs(new_row)
            
            c_inter = sum(1 for d in new_diffs if d in current_used)
            c_intra = len(new_diffs) - len(set(new_diffs))
            new_conflicts = current_conflicts + c_inter + c_intra - (n_diffs_per_row - len(set(get_diffs(old_row))))
            
            new_max_row = max(new_row)
            other_max = max(row_maxes[idx] for idx in range(n_rows) if idx != i) if n_rows > 1 else 0
            new_scope = max(new_max_row, other_max)
            
            penalty = 100
            old_cost = current_scope + penalty * current_conflicts
            new_cost = new_scope + penalty * new_conflicts
            delta_cost = new_cost - old_cost
            
            if delta_cost < 0 or random.random() < pow(2.718281828, -delta_cost / T):
                current_rows[i] = new_row
                row_maxes[i] = new_max_row
                current_used.update(new_diffs)
                current_scope = new_scope
                current_conflicts = new_conflicts
                
                if new_conflicts == 0 and new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = copy.deepcopy(current_rows)
            else:
                for a in range(row_len):
                    for b in range(a+1, row_len):
                        current_used.add(old_row[b] - old_row[a])
        
        T *= cooling
    
    return best_rows
