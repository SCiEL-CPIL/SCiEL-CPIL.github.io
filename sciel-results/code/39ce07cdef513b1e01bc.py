"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, but task says 120s limit usually.
    # We use 115s to be safe against any overhead.
    deadline = start_time + 115.0
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    def get_max_entry(rows):
        return max(max(row) for row in rows) if rows else 0

    def check_validity(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def compute_conflicts(rows):
        """Returns list of (diff, count) for diffs with count > 1"""
        counts = {}
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    counts[d] = counts.get(d, 0) + 1
        conflicts = []
        for d, c in counts.items():
            if c > 1:
                conflicts.append((d, c))
        return conflicts

    def greedy_construct(max_val):
        """
        Greedily constructs rows one by one.
        For each row, picks the smallest possible next integer that doesn't
        create a conflict with previously used differences.
        Returns None if it gets stuck (no valid next integer <= max_val).
        """
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            for _pos in range(k):
                # Try to find the next smallest integer x > current_max
                # such that all (x - prev) for prev in row are not in used_diffs
                # and (x - prev) <= max_val
                found = False
                # We search from current_max + 1 upwards
                # Optimization: We can limit the search range if we are tight on max_val
                # But for greedy, we just want the smallest valid one.
                
                # To speed up, we can skip values that are likely to conflict.
                # But simple linear scan is O(k) per step, which is fine for small k.
                
                next_val = current_max + 1
                while next_val <= max_val:
                    # Check if adding next_val creates any conflict
                    valid = True
                    for prev in row:
                        diff = next_val - prev
                        if diff in used_diffs:
                            valid = False
                            break
                        # Also ensure diff is positive (guaranteed since next_val > prev)
                    if valid:
                        row.append(next_val)
                        current_max = next_val
                        # Update used_diffs
                        for prev in row[:-1]:
                            used_diffs.add(next_val - prev)
                        found = True
                        break
                    next_val += 1
                
                if not found:
                    return None # Stuck, cannot complete this row within max_val
            
            rows.append(row)
        
        return rows

    def repair_rows(rows, max_val, time_limit):
        """
        Local search to resolve conflicts.
        We identify conflicting differences and try to shift elements in the rows
        involved to break the conflict, while keeping the row valid (increasing)
        and within bounds.
        """
        rows = [row[:] for row in rows]
        T = 50.0
        cooling = 0.999
        
        # Initial conflict count
        conflicts = compute_conflicts(rows)
        current_cost = len(conflicts) * 1000 + get_max_entry(rows)
        
        while time.time() < time_limit:
            if not conflicts:
                # If no conflicts, try to reduce max_val? 
                # For now, just return if valid.
                # Actually, we want to minimize max entry.
                # If valid, we can try to "tighten" by shifting rows down?
                # But shifting is hard. Let's just return valid rows.
                # The caller will check if this is better than best.
                return rows, True
            
            # Pick a random conflict
            diff, count = random.choice(conflicts)
            
            # Find rows involved in this conflict
            # We need to find which rows contain this difference
            involved_rows = []
            for r_idx, row in enumerate(rows):
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        if row[j] - row[i] == diff:
                            involved_rows.append(r_idx)
                            break
                    else:
                        continue
                    break
            
            if not involved_rows:
                # Should not happen if conflict list is accurate, but safety
                continue
                
            # Pick a row from involved to modify
            r_idx = random.choice(involved_rows)
            row = rows[r_idx]
            
            # Pick a random position in the row to move (1..k, index 0 is fixed at 0)
            c_idx = random.randint(1, k)
            
            # Determine valid range for new value
            min_val = row[c_idx - 1] + 1
            max_val_limit = max_val
            if c_idx < k:
                max_val_limit = min(max_val_limit, row[c_idx + 1] - 1)
            
            if min_val > max_val_limit:
                continue
                
            # Try a random new value in the range
            # Bias towards smaller values to reduce max entry?
            # Or just random.
            new_val = random.randint(min_val, max_val_limit)
            
            if new_val == row[c_idx]:
                continue
                
            old_val = row[c_idx]
            row[c_idx] = new_val
            
            # Check if this move reduced conflicts
            new_conflicts = compute_conflicts(rows)
            new_cost = len(new_conflicts) * 1000 + get_max_entry(rows)
            
            delta = new_cost - current_cost
            
            # Accept if better, or with probability exp(-delta/T)
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                current_cost = new_cost
                conflicts = new_conflicts
                # If we found a valid solution, we can return early if we want,
                # but let's continue to try to reduce max entry.
                if not conflicts:
                    # Valid! Try to optimize max entry for a bit more?
                    # For simplicity, return valid rows.
                    # The outer loop can restart with lower max_val.
                    return rows, True
            else:
                # Revert
                row[c_idx] = old_val
            
            T *= cooling
            if T < 1.0:
                break
        
        return rows, False

    best_rows = None
    best_max = float('inf')
    
    # Phase 1: Find any valid solution using greedy with increasing max_val
    # Lower bound is total_diffs.
    # Greedy usually works for reasonable max_val.
    # Start from lower bound and increase.
    
    lb = total_diffs
    # Estimate a reasonable upper bound.
    # Greedy tends to use about n_rows * k * (k+1) / 2 * some_factor?
    # Let's start with lb and double until we find a solution.
    
    current_max = lb
    found_initial = False
    
    while time.time() < deadline - 5.0:
        rows = greedy_construct(current_max)
        if rows:
            best_rows = rows
            best_max = get_max_entry(rows)
            found_initial = True
            break
        # Increase max_val
        # Exponential growth
        current_max = int(current_max * 1.2) + 10
        if current_max > 10000:
            break
            
    if not found_initial:
        # Fallback: return a valid but potentially large set if we can't find one?
        # Greedy should always find one if max_val is large enough.
        # If we run out of time, return best effort (which might be None).
        # But task requires valid output.
        # Let's try a larger max_val quickly.
        rows = greedy_construct(10000)
        if rows:
            return rows
        # Last resort: return a trivially valid set (might be huge)
        # Just construct rows with large gaps
        rows = []
        base = 1
        for _ in range(n_rows):
            row = [0]
            for i in range(1, k+1):
                row.append(base * (i+1))
            rows.append(row)
            base *= (k+2) # Ensure no overlaps
        return rows

    # Phase 2: Improve the solution
    # We have a valid solution with best_max.
    # Strategy:
    # 1. Try to repair and reduce conflicts if we allow slightly larger max_val? No, we want smaller.
    # 2. Use local search to try to find a valid solution with max_val < best_max.
    #    We can do this by running repair on a "relaxed" problem where we allow conflicts
    #    but penalize them heavily, and constrain max_val to best_max - 1.
    
    # Actually, a better approach for "Difference Triangle Sets" is often:
    # 1. Generate a valid set.
    # 2. Try to shift rows to reduce the maximum.
    
    # Let's use a multi-start local search aiming for max_val < best_max.
    
    target_max = best_max - 1
    
    for _ in range(50): # Max 50 improvement attempts
        if time.time() > deadline - 5.0:
            break
            
        # Initialize with random rows within target_max
        # Or start from best_rows and perturb?
        # Starting from scratch within target_max is risky (many conflicts).
        # Starting from best_rows and forcing max_val down is better.
        
        current_rows = [row[:] for row in best_rows]
        
        # Force max_val constraint: if any element > target_max, it's a "violation"
        # We can treat max_val violation as a conflict type.
        
        # Let's run a specific local search:
        # Cost = num_conflicts * 10000 + max(0, max_entry - target_max) * 100
        # We want to reach cost = 0.
        
        T = 20.0
        cooling = 0.9995
        current_cost = float('inf')
        
        # Calculate initial cost
        def calc_cost(rows):
            confs = compute_conflicts(rows)
            num_conf = len(confs)
            max_ent = get_max_entry(rows)
            limit_pen = 0
            if max_ent > target_max:
                limit_pen = (max_ent - target_max) * 10
            return num_conf * 1000 + limit_pen
        
        current_cost = calc_cost(current_rows)
        
        # If already valid and within limit, update best
        if current_cost == 0:
             if check_validity(current_rows):
                 new_max = get_max_entry(current_rows)
                 if new_max < best_max:
                     best_max = new_max
                     best_rows = current_rows
                     target_max = best_max - 1
                     continue
                     
        while T > 1.0 and time.time() < deadline - 5.0:
            # Pick a row and position
            r_idx = random.randint(0, n_rows - 1)
            c_idx = random.randint(1, k)
            
            row = current_rows[r_idx]
            
            min_val = row[c_idx - 1] + 1
            # Allow going up to best_max (not target_max) to allow exploration
            # But if we go above target_max, we pay penalty.
            max_val_limit = best_max
            
            if c_idx < k:
                max_val_limit = min(max_val_limit, row[c_idx + 1] - 1)
                
            if min_val > max_val_limit:
                T *= cooling
                continue
                
            new_val = random.randint(min_val, max_val_limit)
            
            if new_val == row[c_idx]:
                T *= cooling
                continue
                
            old_val = row[c_idx]
            row[c_idx] = new_val
            
            new_cost = calc_cost(current_rows)
            delta = new_cost - current_cost
            
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                current_cost = new_cost
                if new_cost == 0:
                    # Valid and within limit!
                    if check_validity(current_rows):
                        new_max = get_max_entry(current_rows)
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [r[:] for r in current_rows]
                            target_max = best_max - 1
                            # Restart SA with new target? Or just break and restart loop
                            break
            else:
                row[c_idx] = old_val
                
            T *= cooling
            
        # End inner SA
        
    return best_rows

# EVOLVE-BLOCK-END
