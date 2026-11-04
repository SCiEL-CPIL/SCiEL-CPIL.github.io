import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Total differences needed
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # Fast validity check using set of differences
    def get_row_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid_solution(rows):
        """Check if all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            row_diffs = get_row_diffs(row)
            if all_diffs & row_diffs:
                return False
            all_diffs |= row_diffs
        return True
    
    def greedy_construct():
        """Build a valid solution greedily."""
        rows = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                new_diffs = set()
                for y in row:
                    d = x - y
                    if d <= 0 or d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    row.append(x)
                    used_diffs |= new_diffs
            rows.append(row)
        return rows
    
    # Start with greedy solution
    best_solution = greedy_construct()
    best_scope = max(max(r) for r in best_solution)
    
    # Simulated annealing to improve
    current_solution = [r[:] for r in best_solution]
    current_scope = best_scope
    temperature = 100.0
    cooling_rate = 0.995
    min_temperature = 0.01
    
    def compute_cost(rows):
        """Cost = scope + penalty for invalid differences."""
        scope = max(max(r) for r in rows)
        all_diffs = set()
        penalty = 0
        for row in rows:
            row_diffs = get_row_diffs(row)
            overlap = all_diffs & row_diffs
            penalty += len(overlap) * 1000  # Heavy penalty for duplicates
            all_diffs |= row_diffs
        return scope + penalty
    
    def try_move(solution, max_scope):
        """Try a random move: change one entry in one row."""
        new_solution = [r[:] for r in solution]
        
        # Pick a random row and a random position (not the first)
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        # Try to change this position to a smaller value
        old_val = new_solution[row_idx][pos]
        
        # Try values from old_val-1 down to new_solution[row_idx][pos-1]+1
        for new_val in range(old_val - 1, new_solution[row_idx][pos - 1], -1):
            if new_val <= new_solution[row_idx][pos - 1]:
                break
            new_solution[row_idx][pos] = new_val
            
            # Check if this row is still strictly increasing
            row = new_solution[row_idx]
            valid_row = True
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    valid_row = False
                    break
            
            if not valid_row:
                new_solution[row_idx][pos] = old_val
                continue
            
            # Check if this creates duplicate differences within this row
            row_diffs = get_row_diffs(row)
            if len(row_diffs) != (k + 1) * k // 2:
                new_solution[row_idx][pos] = old_val
                continue
            
            # Check if this creates duplicates with other rows
            all_diffs = set()
            for i, r in enumerate(new_solution):
                if i == row_idx:
                    continue
                r_diffs = get_row_diffs(r)
                if all_diffs & r_diffs:
                    new_solution[row_idx][pos] = old_val
                    valid_row = False
                    break
                all_diffs |= r_diffs
            
            if not valid_row:
                continue
            
            # Check against the modified row
            if all_diffs & row_diffs:
                new_solution[row_idx][pos] = old_val
                continue
            
            # Move is valid, return it
            return new_solution
        
        # No valid smaller value found, try a different strategy:
        # Pick a random position and try a random value in a range
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        old_val = new_solution[row_idx][pos]
        
        # Try a random value between min and max
        min_val = new_solution[row_idx][pos - 1] + 1
        max_val = old_val + 5  # Allow some increase
        if min_val >= max_val:
            return None
        
        new_val = random.randint(min_val, max_val)
        new_solution[row_idx][pos] = new_val
        
        # Validate
        row = new_solution[row_idx]
        valid_row = True
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                valid_row = False
                break
        
        if not valid_row:
            return None
        
        row_diffs = get_row_diffs(row)
        if len(row_diffs) != (k + 1) * k // 2:
            return None
        
        all_diffs = set()
        for i, r in enumerate(new_solution):
            if i == row_idx:
                continue
            r_diffs = get_row_diffs(r)
            if all_diffs & r_diffs:
                return None
            all_diffs |= r_diffs
        
        if all_diffs & row_diffs:
            return None
        
        return new_solution
    
    # Run simulated annealing
    while time.time() < deadline - 5.0:
        if temperature < min_temperature:
            break
        
        # Try a move
        new_solution = try_move(current_solution, current_scope)
        
        if new_solution is None:
            temperature *= cooling_rate
            continue
        
        new_scope = max(max(r) for r in new_solution)
        new_cost = compute_cost(new_solution)
        old_cost = compute_cost(current_solution)
        
        # Accept if better or with probability based on temperature
        if new_cost < old_cost:
            current_solution = new_solution
            if new_scope < best_scope:
                best_scope = new_scope
                best_solution = [r[:] for r in new_solution]
        else:
            # Metropolis criterion
            delta = new_cost - old_cost
            if delta < 0 or random.random() < pow(2.718281828, -delta / temperature):
                current_solution = new_solution
        
        temperature *= cooling_rate
    
    # Final check: if current solution is better and valid, use it
    if is_valid_solution(current_solution):
        curr_scope = max(max(r) for r in current_solution)
        if curr_scope < best_scope:
            best_scope = curr_scope
            best_solution = [r[:] for r in current_solution]
    
    return best_solution
