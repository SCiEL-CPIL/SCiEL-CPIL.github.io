"""Difference triangle set (7,5) solver using simulated annealing with targeted row optimization."""
import time
import random
import copy

def construct(n_rows, k):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Parameters
    k_plus_1 = k + 1  # number of entries per row
    num_diffs_per_row = k * (k + 1) // 2  # differences per row
    total_diffs = n_rows * num_diffs_per_row
    
    # Helper: compute all positive differences in a row
    def row_diffs(row):
        return set(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))
    
    # Helper: check if a set of differences is valid (all unique globally)
    def all_diffs_valid(rows):
        seen = set()
        for row in rows:
            for d in row_diffs(row):
                if d in seen:
                    return False
                seen.add(d)
        return True
    
    # Helper: get the largest entry across all rows
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    # Helper: validate a complete solution
    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        for row in rows:
            if len(row) != k_plus_1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
        return all_diffs_valid(rows)
    
    # Initial solution: greedy construction
    def greedy_initial():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k_plus_1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Local improvement: try to reduce the max entry of a specific row
    def improve_row(rows, row_idx, deadline):
        """Try to reduce the largest entry in row_idx while maintaining validity."""
        row = rows[row_idx]
        current_max = max(row)
        best_row = list(row)
        best_score = current_max
        
        # Get all differences used by other rows
        other_diffs = set()
        for i, r in enumerate(rows):
            if i != row_idx:
                other_diffs |= row_diffs(r)
        
        # Try to find a better row with smaller max
        # Strategy: start from current row and try to reduce the last element
        for _ in range(200):  # limited iterations
            if time.time() > deadline:
                break
            
            # Try reducing the last element
            new_row = list(best_row)
            last = new_row[-1]
            # Try values below last
            found = False
            for candidate in range(last - 1, 0, -1):
                new_row[-1] = candidate
                if candidate <= new_row[-2]:
                    continue
                # Check if this row is valid internally and doesn't conflict
                r_diffs = row_diffs(new_row)
                if len(r_diffs) != num_diffs_per_row:
                    continue
                if r_diffs & other_diffs:
                    continue
                # Check internal uniqueness
                if len(r_diffs) != num_diffs_per_row:
                    continue
                found = True
                break
            
            if found:
                best_row = new_row
                best_score = max(best_row)
                # Update other_diffs since this row's diffs changed
                other_diffs = set()
                for i, r in enumerate(rows):
                    if i != row_idx:
                        other_diffs |= row_diffs(r)
            else:
                break
        
        return best_row, best_score
    
    # Simulated annealing on the global structure
    def simulated_annealing(rows, deadline):
        """SA that modifies rows to reduce max entry."""
        current = [list(r) for r in rows]
        current_max = max_entry(current)
        best = [list(r) for r in current]
        best_max = current_max
        
        temperature = 50.0
        cooling = 0.995
        
        # Get initial differences
        current_diffs = set()
        for row in current:
            current_diffs |= row_diffs(row)
        
        iterations = 0
        while time.time() < deadline and temperature > 0.5:
            iterations += 1
            
            # Choose which row to modify
            # 70% chance to target the row with the global max
            row_maxes = [max(r) for r in current]
            global_max_idx = row_maxes.index(max(row_maxes))
            
            if random.random() < 0.7:
                target_row = global_max_idx
            else:
                target_row = random.randint(0, n_rows - 1)
            
            # Generate a neighbor: modify one element in the target row
            row = current[target_row]
            pos = random.randint(1, k)  # positions 1..k (not 0)
            
            # Try small perturbations
            delta = random.choice([-2, -1, 1, 2])
            new_val = row[pos] + delta
            
            # Check bounds
            if pos == 0:
                continue
            if new_val <= row[pos - 1] or (pos < k and new_val >= row[pos + 1]):
                continue
            if new_val < 0:
                continue
            
            # Compute new differences for this row
            old_row_diffs = row_diffs(row)
            
            # Tentatively change
            row[pos] = new_val
            new_row_diffs = row_diffs(row)
            
            # Check validity: new diffs must not conflict with other rows' diffs
            other_diffs = current_diffs - old_row_diffs
            conflict = new_row_diffs & other_diffs
            
            if len(conflict) == 0 and len(new_row_diffs) == num_diffs_per_row:
                # Valid move
                new_max = max_entry(current)
                delta_obj = new_max - current_max
                
                if delta_obj < 0 or random.random() < pow(2.718281828, -delta_obj / temperature):
                    # Accept
                    current_diffs = other_diffs | new_row_diffs
                    current_max = new_max
                    
                    if new_max < best_max:
                        best_max = new_max
                        best = [list(r) for r in current]
                else:
                    # Reject
                    row[pos] -= delta
            else:
                # Invalid, revert
                row[pos] -= delta
            
            temperature *= cooling
        
        return best, best_max
    
    # Main algorithm
    rows = greedy_initial()
    if not is_valid(rows):
        # Fallback: return greedy result even if not perfectly valid (shouldn't happen)
        return rows
    
    best_rows = [list(r) for r in rows]
    best_max = max_entry(best_rows)
    
    # Phase 1: Targeted row improvement for the worst row
    if time.time() < deadline:
        # Find row with max entry and try to improve it
        row_maxes = [max(r) for r in best_rows]
        worst_row = row_maxes.index(max(row_maxes))
        
        improved_row, new_score = improve_row(best_rows, worst_row, deadline)
        if new_score < max(improved_row):
            # Actually, new_score is the max of improved_row
            pass
        best_rows[worst_row] = improved_row
        best_max = max_entry(best_rows)
    
    # Phase 2: Simulated annealing
    if time.time() < deadline:
        sa_result, sa_max = simulated_annealing(best_rows, deadline)
        if sa_max < best_max:
            best_rows = sa_result
            best_max = sa_max
    
    # Phase 3: Final polish - try targeted improvements on the worst row again
    if time.time() < deadline - 5.0:
        row_maxes = [max(r) for r in best_rows]
        worst_row = row_maxes.index(max(row_maxes))
        
        improved_row, new_score = improve_row(best_rows, worst_row, deadline - 5.0)
        if max(improved_row) < max(best_rows[worst_row]):
            best_rows[worst_row] = improved_row
            best_max = max_entry(best_rows)
    
    # Final validation
    if not is_valid(best_rows):
        # Return the greedy solution as fallback
        return rows
    
    return best_rows
