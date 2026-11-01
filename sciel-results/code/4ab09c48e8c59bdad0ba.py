import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = 115.0  # Keep a margin from the 150s limit
    
    # Total differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def compute_row_diffs(row):
        """Compute all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def check_valid(rows):
        """Check if all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True
    
    # First, get a valid greedy solution as fallback
    used_diffs_greedy = set()
    greedy_rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            x = current_max + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs_greedy:
                    break
                x += 1
            row.append(x)
            current_max = x
            used_diffs_greedy |= new_diffs
        greedy_rows.append(row)
    
    greedy_scope = get_scope(greedy_rows)
    best_rows = [r[:] for r in greedy_rows]
    best_scope = greedy_scope
    
    def backtrack_row(row, used_diffs, depth, max_val_bound):
        """
        Try to complete a row starting from given state.
        row is a list, used_diffs is the set of all used differences.
        Returns (completed_row, new_used_diffs) or None if failed.
        """
        if time.time() > deadline - 5.0:
            return None
        
        if len(row) == k + 1:
            return (row[:], used_diffs)
        
        prev = row[-1]
        remaining = k + 1 - len(row)
        
        # Try candidate values with MRV-like heuristic: prefer smaller values
        # but also consider spread
        candidates = []
        upper = min(prev + remaining * 20 + 20, max_val_bound)
        
        for x in range(prev + 1, upper + 1):
            if time.time() > deadline - 5.0:
                return None
            
            # Check if adding x creates any conflict with used_diffs
            new_diffs = {x - a for a in row}
            if new_diffs & used_diffs:
                continue
            
            # Also check internal consistency (differences within new row)
            # Since row is increasing, all new diffs are positive and unique within row
            # if x - a values are distinct, which they are since a values are distinct
            
            new_row = row + [x]
            new_used = used_diffs | new_diffs
            
            # If we're at the last position, we're done
            if len(new_row) == k + 1:
                return (new_row, new_used)
            
            # Recurse
            result = backtrack_row(new_row, new_used, depth + 1, max_val_bound)
            if result is not None:
                return result
        
        return None
    
    def fast_sa_optimize(rows, num_iterations=20000):
        """
        Fast simulated annealing with precomputed difference sets.
        Maintains per-row difference sets and a global used set for O(1) conflict checks.
        """
        current_rows = [r[:] for r in rows]
        
        # Precompute difference sets for each row
        row_diffs = [compute_row_diffs(r) for r in current_rows]
        
        # Global used diffs (union of all row diffs)
        global_used = set()
        for rd in row_diffs:
            global_used |= rd
        
        current_scope = get_scope(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        
        T = 5.0  # Initial temperature
        cooling = 0.9995
        
        for it in range(num_iterations):
            if time.time() > deadline - 3.0:
                break
            
            # Pick a random row to modify
            row_idx = random.randint(0, n_rows - 1)
            row = current_rows[row_idx]
            
            # Pick a random position to change (not the first element 0)
            pos = random.randint(1, len(row) - 1)
            
            old_val = row[pos]
            
            # Generate a new candidate value
            # Bias towards reducing the maximum
            is_max_pos = (pos == len(row) - 1)
            if is_max_pos and current_scope == max(max(r) for r in current_rows):
                # Try to reduce the binding element
                delta = random.randint(-8, 1)
            else:
                delta = random.randint(-4, 4)
            new_val = old_val + delta
            
            # Ensure strictly increasing
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
            
            # Compute new differences involving the changed position
            new_diffs = set()
            for j in range(len(row)):
                if j != pos:
                    if j < pos:
                        new_diffs.add(new_val - row[j])
                    else:
                        new_diffs.add(row[j] - new_val)
            
            # Old differences involving the changed position
            old_diffs = set()
            for j in range(len(row)):
                if j != pos:
                    if j < pos:
                        old_diffs.add(old_val - row[j])
                    else:
                        old_diffs.add(row[j] - old_val)
            
            # Check conflicts: new_diffs must not be in (global_used - old_diffs)
            # i.e., new_diffs can only use diffs that were previously in old_diffs
            forbidden = global_used - old_diffs
            if new_diffs & forbidden:
                continue
            
            # Also check that new_diffs don't have internal duplicates
            # (they shouldn't since row is strictly increasing)
            
            # Compute scope change
            new_row = row[:]
            new_row[pos] = new_val
            new_row_max = max(new_row)
            
            # Compute new scope
            other_max = 0
            for i in range(n_rows):
                if i != row_idx:
                    other_max = max(other_max, max(current_rows[i]))
            new_scope = max(other_max, new_row_max)
            
            delta_scope = new_scope - current_scope
            
            # Accept if better, or with probability based on temperature
            if delta_scope <= 0 or (T > 0.01 and random.random() < math.exp(-delta_scope / T)):
                # Accept the move
                # Update row_diffs[row_idx]
                # Remove old_diffs, add new_diffs
                row_diffs[row_idx] = (row_diffs[row_idx] - old_diffs) | new_diffs
                
                # Update global_used
                global_used = (global_used - old_diffs) | new_diffs
                
                current_rows[row_idx] = new_row
                current_scope = new_scope
                
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
            
            T *= cooling
        
        return best_rows, best_scope
    
    # Try multiple strategies
    best_found = best_scope
    
    # Strategy 1: Greedy + SA
    if time.time() < deadline - 20.0:
        opt_rows, opt_scope = fast_sa_optimize(greedy_rows, num_iterations=15000)
        if opt_scope < best_found:
            best_found = opt_scope
            best_rows = [r[:] for r in opt_rows]
    
    # Strategy 2: Backtracking-based construction with tighter bounds
    for restart in range(10):
        if time.time() > deadline - 15.0:
            break
        
        used_diffs = set()
        current_rows = []
        success = True
        
        for row_idx in range(n_rows):
            if time.time() > deadline - 5.0:
                success = False
                break
            
            row = [0]
            # Use backtracking with a tighter bound
            bound = best_found + 5
            result = backtrack_row(row, used_diffs, 0, bound)
            
            if result is None:
                # Fallback to greedy for this row
                row = [0]
                current_max = 0
                while len(row) < k + 1:
                    x = current_max + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used_diffs:
                            break
                        x += 1
                    row.append(x)
                    current_max = x
                    used_diffs |= new_diffs
            else:
                completed_row, used_diffs = result
                row = completed_row
            
            current_rows.append(row)
        
        if not success:
            continue
        
        # Verify validity
        if not check_valid(current_rows):
            continue
        
        cur_scope = get_scope(current_rows)
        if cur_scope < best_found:
            # Polish with SA
            opt_rows, opt_scope = fast_sa_optimize(current_rows, num_iterations=8000)
            if opt_scope < best_found:
                best_found = opt_scope
                best_rows = [r[:] for r in opt_rows]
            elif cur_scope < best_found:
                best_found = cur_scope
                best_rows = [r[:] for r in current_rows]
    
    # Final polish if time remains
    if time.time() < deadline - 5.0 and best_found < greedy_scope:
        opt_rows, opt_scope = fast_sa_optimize(best_rows, num_iterations=10000)
        if opt_scope < best_found:
            best_found = opt_scope
            best_rows = [r[:] for r in opt_rows]
    
    return best_rows
