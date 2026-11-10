"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 integers.
    
    Args:
        n_rows: number of rows
        k: each row has k+1 entries (including 0)
    
    Returns:
        list of n_rows rows, each a list of k+1 increasing integers starting at 0
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    def get_differences(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        all_diffs = []
        for row in rows:
            all_diffs.extend(get_differences(row))
        return len(all_diffs) == len(set(all_diffs))
    
    def max_entry(rows):
        """Get the largest entry across all rows."""
        return max(max(row) for row in rows)
    
    def greedy_construct():
        """Greedy construction: pick smallest possible next value."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Start with greedy solution
    best_rows = greedy_construct()
    best_score = max_entry(best_rows)
    
    if time.time() > deadline:
        return best_rows
    
    # Local search with simulated annealing
    current_rows = [row[:] for row in best_rows]
    current_score = best_score
    
    # Temperature for simulated annealing
    temperature = 50.0
    cooling_rate = 0.999
    
    def perturb(rows):
        """Apply a random perturbation to rows."""
        new_rows = [row[:] for row in rows]
        
        # Choose a random row to modify
        row_idx = random.randint(0, n_rows - 1)
        
        # Choose perturbation type
        pert_type = random.choice(['regenerate', 'shift', 'swap'])
        
        if pert_type == 'regenerate':
            # Regenerate the row using greedy but with some randomness
            used_by_others = set()
            for i, row in enumerate(new_rows):
                if i != row_idx:
                    used_by_others |= get_differences(row)
            
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used_by_others:
                    # With some probability, skip this valid value to explore
                    if random.random() < 0.1 and x < current_score:
                        continue
                    row.append(x)
                    used_by_others |= new_diffs
            new_rows[row_idx] = row
            
        elif pert_type == 'shift':
            # Shift a random entry in a random row
            row_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)  # Don't shift the first entry (0)
            delta = random.randint(-5, 5)
            new_val = new_rows[row_idx][pos] + delta
            
            # Ensure strictly increasing
            if pos > 0 and new_val <= new_rows[row_idx][pos - 1]:
                new_val = new_rows[row_idx][pos - 1] + 1
            if pos < k and new_val >= new_rows[row_idx][pos + 1]:
                new_val = new_rows[row_idx][pos + 1] - 1
            
            if new_val > 0:
                new_rows[row_idx][pos] = new_val
        
        elif pert_type == 'swap':
            # Swap entries between two rows (carefully)
            if n_rows >= 2:
                i, j = random.sample(range(n_rows), 2)
                pos_i = random.randint(1, k)
                pos_j = random.randint(1, k)
                
                val_i = new_rows[i][pos_i]
                val_j = new_rows[j][pos_j]
                
                # Try to swap and fix ordering
                new_rows[i][pos_i] = val_j
                new_rows[j][pos_j] = val_i
                
                # Fix ordering in row i
                for p in range(1, len(new_rows[i])):
                    if new_rows[i][p] <= new_rows[i][p-1]:
                        new_rows[i][p] = new_rows[i][p-1] + 1
                for p in range(len(new_rows[i])-2, -1, -1):
                    if new_rows[i][p] >= new_rows[i][p+1]:
                        new_rows[i][p] = new_rows[i][p+1] - 1
                
                # Fix ordering in row j
                for p in range(1, len(new_rows[j])):
                    if new_rows[j][p] <= new_rows[j][p-1]:
                        new_rows[j][p] = new_rows[j][p-1] + 1
                for p in range(len(new_rows[j])-2, -1, -1):
                    if new_rows[j][p] >= new_rows[j][p+1]:
                        new_rows[j][p] = new_rows[j][p+1] - 1
        
        return new_rows
    
    def repair(rows):
        """Try to repair an invalid solution by adjusting entries."""
        new_rows = [row[:] for row in rows]
        
        for _ in range(100):  # Limited repair attempts
            if is_valid(new_rows):
                break
            
            # Find a conflict
            all_diffs = []
            diff_to_row = {}
            for i, row in enumerate(new_rows):
                for d in get_differences(row):
                    if d in diff_to_row:
                        # Conflict found
                        # Try to fix by adjusting one of the rows
                        conflict_row = diff_to_row[d]
                        # Regenerate that row
                        used_by_others = set()
                        for j, r in enumerate(new_rows):
                            if j != conflict_row:
                                used_by_others |= get_differences(r)
                        
                        row = [0]
                        x = 0
                        while len(row) < k + 1:
                            x += 1
                            new_diffs = {x - a for a in row}
                            if not new_diffs & used_by_others:
                                row.append(x)
                                used_by_others |= new_diffs
                        new_rows[conflict_row] = row
                        break
                    else:
                        diff_to_row[d] = i
                        all_diffs.append(d)
            else:
                break
        
        return new_rows
    
    # Main search loop
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        # Perturb
        candidate = perturb(current_rows)
        
        # Try to repair if invalid
        if not is_valid(candidate):
            candidate = repair(candidate)
        
        if not is_valid(candidate):
            temperature *= cooling_rate
            continue
        
        candidate_score = max_entry(candidate)
        
        # Accept if better or with probability based on temperature
        if candidate_score < current_score:
            current_rows = candidate
            current_score = candidate_score
            no_improve_count = 0
            
            if current_score < best_score:
                best_rows = [row[:] for row in current_rows]
                best_score = current_score
        else:
            # Simulated annealing acceptance
            delta = candidate_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.1)):
                current_rows = candidate
                current_score = candidate_score
                no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Cool down
        temperature *= cooling_rate
        if temperature < 0.1:
            temperature = 0.1
        
        # Reset if stuck
        if no_improve_count > max_no_improve:
            current_rows = [row[:] for row in best_rows]
            current_score = best_score
            no_improve_count = 0
            temperature = 20.0
    
    return best_rows
# EVOLVE-BLOCK-END
