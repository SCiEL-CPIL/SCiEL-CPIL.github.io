import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 118.0
    row_len = k + 1
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    rng = random.Random(42)
    
    # Pre-compute pairs for a row to speed up difference calculation
    row_pairs = [(i, j) for i in range(row_len) for j in range(i+1, row_len)]
    
    def get_row_diffs(row):
        """Calculate all positive differences for a single row."""
        return [row[j] - row[i] for i, j in row_pairs]
    
    def get_scope(rows):
        """Calculate the maximum entry in all rows."""
        return max(row[-1] for row in rows)
    
    def count_conflicts(counter):
        """Count the number of differences that appear more than once."""
        return sum(c - 1 for c in counter.values() if c > 1)
    
    def greedy_initial():
        """Generate a valid initial solution using greedy."""
        matrix = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < row_len:
                last += 1
                new_diffs = {last - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(last)
                    used_diffs |= new_diffs
            matrix.append(row)
        return matrix

    def build_row_avoiding(used_diffs, cap):
        """
        Build a new row of length row_len starting at 0,
        avoiding differences in used_diffs.
        cap is the maximum allowed value for the last element.
        Returns None if not possible within cap.
        """
        row = [0]
        current_max = 0
        attempts = 0
        max_attempts = 2000
        
        while len(row) < row_len:
            if attempts > max_attempts:
                return None
            attempts += 1
            
            next_val = current_max + 1
            found = False
            
            while next_val <= cap:
                new_diffs = {next_val - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(next_val)
                    current_max = next_val
                    used_diffs = used_diffs | new_diffs
                    found = True
                    break
                next_val += 1
            
            if not found:
                return None
                
        return row
    
    def check_additive_consistency(row):
        """
        Check if a row's differences are additively consistent.
        For a valid row [a0, a1, ..., ak], the differences must satisfy:
        a_j - a_i = (a_{i+1} - a_i) + (a_{i+2} - a_{i+1}) + ... + (a_j - a_{j-1})
        This is automatically true for any valid row, so this check is trivially true.
        However, we can use it to verify that the differences form a valid structure.
        """
        # For a strictly increasing sequence starting at 0, the differences
        # automatically satisfy additive consistency. This is a no-op check
        # but included for clarity of the CSP approach.
        return True
    
    def generate_row_from_diffs(diffs):
        """
        Given a set of differences that should form a valid row,
        reconstruct the row. Returns None if not possible.
        """
        # This is complex; for now, we'll use a simpler approach
        # where we try to build the row greedily given the constraint
        # that certain differences must appear.
        pass
    
    # 1. Initial Solution
    current_matrix = greedy_initial()
    best_matrix = [r[:] for r in current_matrix]
    best_scope = get_scope(best_matrix)
    
    # Initialize difference counter for current state
    diff_counter = Counter()
    for row in current_matrix:
        for d in get_row_diffs(row):
            diff_counter[d] += 1
            
    current_conflicts = count_conflicts(diff_counter)
    current_scope = get_scope(current_matrix)
    current_cost = current_scope + 1000 * current_conflicts
    
    # 2. Enhanced Simulated Annealing with multiple move types
    T = 30.0
    cooling = 0.999
    min_T = 0.05
    
    # Track best per row for potential row-swap moves
    row_scores = [row[-1] for row in current_matrix]
    
    while time.time() < DEADLINE:
        # Choose move type with probabilities
        # 50% Single Entry Shift, 30% Row Rebuild, 20% Pair Swap
        move_type = rng.random()
        r_idx = rng.randint(0, n_rows - 1)
        
        if move_type < 0.5:
            # --- Move Type 1: Shift a single entry ---
            i_idx = rng.randint(1, row_len - 1)
            
            old_val = current_matrix[r_idx][i_idx]
            
            lo = current_matrix[r_idx][i_idx-1] + 1
            if i_idx < row_len - 1:
                hi = current_matrix[r_idx][i_idx+1] - 1
            else:
                hi = current_scope + 15 
            
            if lo > hi:
                continue
            
            # Propose new value with bias towards reduction
            if rng.random() < 0.6:
                new_val = rng.randint(lo, min(old_val, hi))
            else:
                new_val = rng.randint(lo, hi)
                
            if new_val == old_val:
                continue
            
            if i_idx > 0 and new_val <= current_matrix[r_idx][i_idx-1]:
                continue
            if i_idx < row_len - 1 and new_val >= current_matrix[r_idx][i_idx+1]:
                continue
                
            old_row = current_matrix[r_idx]
            old_diffs = get_row_diffs(old_row)
            
            for d in old_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            
            new_row = old_row[:]
            new_row[i_idx] = new_val
            
            new_diffs = get_row_diffs(new_row)
            for d in new_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1
                
            new_conflicts = count_conflicts(diff_counter)
            new_scope = get_scope(current_matrix)
            new_cost = new_scope + 1000 * new_conflicts
            delta = new_cost - current_cost
            
            if delta < 0 or (T > min_T and rng.random() < math.exp(-delta / T)):
                current_matrix[r_idx] = new_row
                current_conflicts = new_conflicts
                current_scope = new_scope
                current_cost = new_cost
                row_scores[r_idx] = new_row[-1]
                
                if new_conflicts == 0 and new_scope < best_scope:
                    best_scope = new_scope
                    best_matrix = [r[:] for r in current_matrix]
            else:
                for d in new_diffs:
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                for d in old_diffs:
                    diff_counter[d] = diff_counter.get(d, 0) + 1
                current_matrix[r_idx] = old_row
                
        elif move_type < 0.8:
            # --- Move Type 2: Rebuild Row ---
            old_row = current_matrix[r_idx]
            old_diffs = get_row_diffs(old_row)
            
            for d in old_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            
            used_diffs = set(diff_counter.keys())
            
            cap = current_scope + rng.randint(0, 8)
            new_row = build_row_avoiding(used_diffs, cap)
            
            if new_row is None:
                for d in old_diffs:
                    diff_counter[d] = diff_counter.get(d, 0) + 1
                continue
            
            new_diffs = get_row_diffs(new_row)
            for d in new_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1
                
            new_conflicts = count_conflicts(diff_counter)
            new_scope = get_scope(current_matrix)
            new_cost = new_scope + 1000 * new_conflicts
            delta = new_cost - current_cost
            
            if delta < 0 or (T > min_T and rng.random() < math.exp(-delta / T)):
                current_matrix[r_idx] = new_row
                current_conflicts = new_conflicts
                current_scope = new_scope
                current_cost = new_cost
                row_scores[r_idx] = new_row[-1]
                
                if new_conflicts == 0 and new_scope < best_scope:
                    best_scope = new_scope
                    best_matrix = [r[:] for r in current_matrix]
            else:
                for d in new_diffs:
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                for d in old_diffs:
                    diff_counter[d] = diff_counter.get(d, 0) + 1
                current_matrix[r_idx] = old_row

        else:
            # --- Move Type 3: Swap entries between two rows ---
            if n_rows < 2:
                continue
                
            r2_idx = rng.randint(0, n_rows - 1)
            if r2_idx == r_idx:
                r2_idx = (r_idx + 1) % n_rows
                
            i1 = rng.randint(1, row_len - 1)
            i2 = rng.randint(1, row_len - 1)
            
            row1 = current_matrix[r_idx]
            row2 = current_matrix[r2_idx]
            
            v1 = row1[i1]
            v2 = row2[i2]
            
            # Check if swap maintains increasing order
            # For row1: need row1[i1-1] < v2 < row1[i1+1] (if applicable)
            valid = True
            if i1 > 0 and v2 <= row1[i1-1]:
                valid = False
            if i1 < row_len - 1 and v2 >= row1[i1+1]:
                valid = False
            if i2 > 0 and v1 <= row2[i2-1]:
                valid = False
            if i2 < row_len - 1 and v1 >= row2[i2+1]:
                valid = False
            
            if not valid:
                continue
            
            # Apply swap
            old_diffs1 = get_row_diffs(row1)
            old_diffs2 = get_row_diffs(row2)
            
            for d in old_diffs1:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            for d in old_diffs2:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            
            new_row1 = row1[:]
            new_row2 = row2[:]
            new_row1[i1] = v2
            new_row2[i2] = v1
            
            new_diffs1 = get_row_diffs(new_row1)
            new_diffs2 = get_row_diffs(new_row2)
            
            for d in new_diffs1:
                diff_counter[d] = diff_counter.get(d, 0) + 1
            for d in new_diffs2:
                diff_counter[d] = diff_counter.get(d, 0) + 1
                
            new_conflicts = count_conflicts(diff_counter)
            new_scope = get_scope(current_matrix)
            new_cost = new_scope + 1000 * new_conflicts
            delta = new_cost - current_cost
            
            if delta < 0 or (T > min_T and rng.random() < math.exp(-delta / T)):
                current_matrix[r_idx] = new_row1
                current_matrix[r2_idx] = new_row2
                current_conflicts = new_conflicts
                current_scope = new_scope
                current_cost = new_cost
                row_scores[r_idx] = new_row1[-1]
                row_scores[r2_idx] = new_row2[-1]
                
                if new_conflicts == 0 and new_scope < best_scope:
                    best_scope = new_scope
                    best_matrix = [r[:] for r in current_matrix]
            else:
                for d in new_diffs1:
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                for d in new_diffs2:
                    diff_counter[d] -= 1
                    if diff_counter[d] == 0:
                        del diff_counter[d]
                for d in old_diffs1:
                    diff_counter[d] = diff_counter.get(d, 0) + 1
                for d in old_diffs2:
                    diff_counter[d] = diff_counter.get(d, 0) + 1
                current_matrix[r_idx] = row1
                current_matrix[r2_idx] = row2

        # Cooling
        T *= cooling
        if T < min_T:
            T = min_T

    # Final polish: try to reduce scope further with aggressive local search
    if current_conflicts == 0:
        polish_deadline = min(time.time() + 5.0, DEADLINE)
        polish_matrix = [r[:] for r in best_matrix]
        polish_scope = best_scope
        
        while time.time() < polish_deadline:
            improved = False
            for r in range(n_rows):
                for i in range(row_len - 1, 0, -1):
                    if i == row_len - 1:
                        old_val = polish_matrix[r][i]
                        lo = polish_matrix[r][i-1] + 1
                        hi = old_val - 1
                        if lo <= hi:
                            new_val = lo  # Try to minimize
                            new_row = polish_matrix[r][:]
                            new_row[i] = new_val
                            
                            # Check validity
                            valid = True
                            new_diffs = set(get_row_diffs(new_row))
                            for other_r in range(n_rows):
                                if other_r == r:
                                    continue
                                other_diffs = set(get_row_diffs(polish_matrix[other_r]))
                                if new_diffs & other_diffs:
                                    valid = False
                                    break
                            
                            if valid:
                                polish_matrix[r] = new_row
                                polish_scope = min(polish_scope, new_val)
                                improved = True
                                break
                    else:
                        # Try to compress this entry
                        old_val = polish_matrix[r][i]
                        lo = polish_matrix[r][i-1] + 1
                        hi = polish_matrix[r][i+1] - 1
                        if lo <= hi and lo < old_val:
                            new_val = lo
                            new_row = polish_matrix[r][:]
                            new_row[i] = new_val
                            
                            valid = True
                            new_diffs = set(get_row_diffs(new_row))
                            for other_r in range(n_rows):
                                if other_r == r:
                                    continue
                                other_diffs = set(get_row_diffs(polish_matrix[other_r]))
                                if new_diffs & other_diffs:
                                    valid = False
                                    break
                            
                            if valid:
                                polish_matrix[r] = new_row
                                improved = True
                                break
                if improved:
                    break
            
            if not improved:
                break
        
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_matrix = polish_matrix

    return best_matrix
