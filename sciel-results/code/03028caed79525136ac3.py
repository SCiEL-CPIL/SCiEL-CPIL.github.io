import random
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Precompute diffs per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Helper: get max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    # Helper: check if all diffs are distinct
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    # Helper: compute all diffs for a row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Pure greedy construction
    def pure_greedy():
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
    
    # Generate random rows with max value constraint
    def generate_random_rows(max_val):
        rows = []
        for _ in range(n_rows):
            if max_val < k:
                row = list(range(k+1))
            else:
                vals = random.sample(range(1, max_val + 1), k)
                vals.sort()
                row = [0] + vals
            rows.append(row)
        return rows
    
    # Fast cost computation with incremental diff tracking
    def compute_all_diffs(rows):
        """Returns (diff_counts dict, max_entry)"""
        diff_counts = {}
        max_entry = 0
        for row in rows:
            if row:
                m = max(row)
                if m > max_entry:
                    max_entry = m
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        return diff_counts, max_entry
    
    def count_conflicts(diff_counts):
        """Count number of conflicts (excess occurrences)"""
        conflicts = 0
        for count in diff_counts.values():
            if count > 1:
                conflicts += count - 1
        return conflicts
    
    # Greedy improvement: try to reduce max by moving elements to smaller values
    def greedy_tighten(rows, max_val, time_limit):
        """Try to reduce max entry by greedily moving elements to smaller values"""
        diff_counts, current_max = compute_all_diffs(rows)
        conflicts = count_conflicts(diff_counts)
        
        # We want to reduce max, so focus on elements that are at or near the max
        # Sort elements by their value (descending) to prioritize reducing large values
        for _ in range(1000):  # Limit iterations per call
            if time.time() > time_limit:
                break
            if conflicts > 0:
                break
            
            # Find the largest element
            max_elem = -1
            max_row_idx = -1
            max_col_idx = -1
            for r_idx in range(n_rows):
                for c_idx in range(1, len(rows[r_idx])):
                    val = rows[r_idx][c_idx]
                    if val > max_elem:
                        max_elem = val
                        max_row_idx = r_idx
                        max_col_idx = c_idx
            
            if max_elem < 1:
                break
            
            # Try to move this element to a smaller value
            min_val = rows[max_row_idx][max_col_idx - 1] + 1
            max_val_allowed = rows[max_row_idx][max_col_idx + 1] - 1 if max_col_idx < k else max_val - 1
            
            if min_val > max_val_allowed:
                continue
            
            # Try all smaller values
            improved = False
            for new_val in range(min_val, max_elem):
                # Check if this value creates conflicts
                old_row = rows[max_row_idx]
                old_diffs = row_diffs(old_row)
                
                # Compute new diffs if we change this element
                temp_row = old_row[:]
                temp_row[max_col_idx] = new_val
                new_diffs = row_diffs(temp_row)
                
                # Check conflicts with other rows
                ok = True
                # Remove old diffs from consideration
                for d in old_diffs - new_diffs:
                    diff_counts[d] -= 1
                for d in new_diffs - old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                    if diff_counts[d] > 1:
                        ok = False
                        break
                
                if ok:
                    # Accept this move
                    rows[max_row_idx][max_col_idx] = new_val
                    # Update diff_counts properly
                    for d in old_diffs - new_diffs:
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
                    for d in new_diffs - old_diffs:
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                    
                    current_max = max(max(row) for row in rows)
                    improved = True
                    break
                
                else:
                    # Revert
                    for d in old_diffs - new_diffs:
                        diff_counts[d] += 1
                    for d in new_diffs - old_diffs:
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
            
            if not improved:
                break
        
        return rows, current_max, count_conflicts(diff_counts)
    
    # Simulated Annealing
    def simulated_annealing(init_rows, target_max, T_start, T_end, cooling_rate, time_limit):
        rows = [row[:] for row in init_rows]
        diff_counts, current_max = compute_all_diffs(rows)
        conflicts = count_conflicts(diff_counts)
        
        # Cost function
        def cost(rows, diff_counts, current_max, conflicts):
            if conflicts > 0:
                return current_max + 1000 * conflicts
            return current_max
        
        current_cost = cost(rows, diff_counts, current_max, conflicts)
        
        # Check if already good
        if conflicts == 0 and current_max < target_max:
            return rows, current_max
        
        T = T_start
        while T > T_end and time.time() < time_limit:
            # Pick a random element to move
            r_idx = random.randint(0, n_rows - 1)
            c_idx = random.randint(1, k)
            
            min_val = rows[r_idx][c_idx - 1] + 1
            max_val_allowed = min(target_max, rows[r_idx][c_idx + 1] - 1 if c_idx < k else target_max)
            
            if min_val > max_val_allowed:
                T *= cooling_rate
                continue
            
            old_val = rows[r_idx][c_idx]
            new_val = random.randint(min_val, max_val_allowed)
            
            if new_val == old_val:
                T *= cooling_rate
                continue
            
            # Compute new diffs
            old_row = rows[r_idx]
            old_diffs = row_diffs(old_row)
            
            temp_row = old_row[:]
            temp_row[c_idx] = new_val
            new_diffs = row_diffs(temp_row)
            
            # Update diff_counts
            removed = old_diffs - new_diffs
            added = new_diffs - old_diffs
            
            for d in removed:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            for d in added:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            new_conflicts = count_conflicts(diff_counts)
            new_max = max(max(row) for row in rows)
            new_cost = cost(rows, diff_counts, new_max, new_conflicts)
            delta = new_cost - current_cost
            
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                rows[r_idx][c_idx] = new_val
                current_cost = new_cost
                current_max = new_max
                conflicts = new_conflicts
                
                # Check if we found a valid solution
                if conflicts == 0 and current_max < target_max:
                    return rows, current_max
            else:
                # Revert
                rows[r_idx][c_idx] = old_val
                for d in removed:
                    diff_counts[d] += 1
                for d in added:
                    diff_counts[d] -= 1
                    if diff_counts[d] == 0:
                        del diff_counts[d]
            
            T *= cooling_rate
        
        return rows, current_max
    
    # Main algorithm
    # Start with greedy solution
    best_rows = pure_greedy()
    best_max = get_max_entry(best_rows)
    
    # Try to improve using multiple strategies
    for strategy in range(20):
        if time.time() > deadline:
            break
        
        # Strategy 1: Greedy tightening
        time_limit = deadline - 10.0
        rows, max_val, conflicts = greedy_tighten([row[:] for row in best_rows], best_max, time_limit)
        if conflicts == 0 and max_val < best_max:
            best_rows = rows
            best_max = max_val
        
        # Strategy 2: Simulated Annealing from random start
        if time.time() > deadline - 20.0:
            break
        
        init_max = best_max + 5
        init_rows = generate_random_rows(init_max)
        
        # Check if random start is already good
        if is_valid(init_rows):
            init_max_entry = get_max_entry(init_rows)
            if init_max_entry < best_max:
                best_rows = [row[:] for row in init_rows]
                best_max = init_max_entry
        
        # Run SA
        time_limit = deadline - 5.0
        T_start = 50.0
        T_end = 1.0
        cooling_rate = 0.9995
        
        rows, max_val = simulated_annealing(init_rows, best_max - 1, T_start, T_end, cooling_rate, time_limit)
        
        if is_valid(rows):
            rows_max = get_max_entry(rows)
            if rows_max < best_max:
                best_rows = [row[:] for row in rows]
                best_max = rows_max
    
    # Final cleanup: try greedy tighten once more
    if time.time() < deadline - 2.0:
        rows, max_val, conflicts = greedy_tighten([row[:] for row in best_rows], best_max, deadline - 1.0)
        if conflicts == 0 and max_val < best_max:
            best_rows = rows
            best_max = max_val
    
    return best_rows
# EVOLVE-BLOCK-END
