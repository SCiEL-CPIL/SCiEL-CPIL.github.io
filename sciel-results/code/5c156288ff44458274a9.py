import time
import random
import sys
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    L = k + 1  # row length
    diffs_per_row = L * (k) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    def get_row_diffs(row):
        """Return a set of differences for a row."""
        diffs = set()
        for i in range(L):
            ri = row[i]
            for j in range(i + 1, L):
                diffs.add(row[j] - ri)
        return diffs
    
    def row_diffs_list(row):
        """Return a list of differences for a row (for counting)."""
        diffs = []
        for i in range(L):
            ri = row[i]
            for j in range(i + 1, L):
                diffs.append(row[j] - ri)
        return diffs
    
    def is_row_valid(row, used_diffs):
        """Check if row's diffs are distinct and don't conflict with used_diffs."""
        row_diffs = get_row_diffs(row)
        if len(row_diffs) != diffs_per_row:
            return False, set()
        if row_diffs & used_diffs:
            return False, set()
        return True, row_diffs
    
    def compute_cost(rows):
        """
        Compute cost for a set of rows.
        Cost = number of conflicts (duplicate diffs across all rows) + max_entry * weight
        Lower cost is better. A valid solution has 0 conflicts.
        """
        all_diffs = []
        max_entry = 0
        for row in rows:
            rd = row_diffs_list(row)
            all_diffs.extend(rd)
            max_entry = max(max_entry, row[-1])
        
        # Count conflicts: differences that appear more than once
        from collections import Counter
        diff_counts = Counter(all_diffs)
        conflicts = sum(c - 1 for c in diff_counts.values() if c > 1)
        
        # Also count internal conflicts (within a single row)
        for row in rows:
            rd = get_row_diffs(row)
            if len(rd) != diffs_per_row:
                conflicts += diffs_per_row - len(rd)
        
        # Weight the max entry to encourage smaller scope
        weight = 1.0
        cost = conflicts * 1000 + max_entry * weight
        return cost, max_entry, conflicts
    
    def is_fully_valid(rows):
        """Check if all rows have distinct internal diffs and no cross-row conflicts."""
        all_diffs = set()
        for row in rows:
            rd = get_row_diffs(row)
            if len(rd) != diffs_per_row:
                return False
            if rd & all_diffs:
                return False
            all_diffs |= rd
        return True
    
    def greedy_construct(n_rows, k, max_val=None):
        """Greedy construction: pick smallest valid x for each position."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < L:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                else:
                    x += 1
                if max_val is not None and x > max_val:
                    break
            rows.append(row)
        return rows
    
    def simulated_annealing(n_rows, k, max_val, time_limit=30.0):
        """
        Use simulated annealing to find a valid configuration with max entry <= max_val.
        Returns list of rows if found, else None.
        """
        t_start = time.time()
        
        # Initialize: use greedy to get a starting point, then cap at max_val
        # Actually, let's start with random valid rows
        rows = []
        used_diffs = set()
        
        for i in range(n_rows):
            # Generate a random row [0, x1, ..., xk] with 1 <= x1 < ... < xk <= max_val
            for attempt in range(1000):
                candidates = random.sample(range(1, max_val + 1), k)
                candidates.sort()
                row = [0] + candidates
                rd = get_row_diffs(row)
                if len(rd) == diffs_per_row and not (rd & used_diffs):
                    rows.append(row)
                    used_diffs |= rd
                    break
            else:
                # Fallback: use greedy for this row
                row = [0]
                x = 1
                while len(row) < L:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                    else:
                        x += 1
                rows.append(row)
        
        current_cost, current_max_val, current_conflicts = compute_cost(rows)
        best_rows = [r[:] for r in rows]
        best_cost = current_cost
        best_max_val = current_max_val
        best_conflicts = current_conflicts
        
        # If already valid, return
        if best_conflicts == 0 and best_max_val <= max_val:
            return best_rows
        
        T = 50.0  # Initial temperature
        T_min = 0.1
        cooling_rate = 0.995
        
        while time.time() - t_start < time_limit:
            # Pick a random row to modify
            r = random.randint(0, n_rows - 1)
            old_row = rows[r][:]
            old_diffs = get_row_diffs(old_row)
            
            # Generate a new candidate for row r
            # Strategy: slightly perturb the existing row
            new_row = old_row[:]
            
            # Randomly modify one or two elements
            num_mods = random.randint(1, 2)
            for _ in range(num_mods):
                pos = random.randint(1, L - 1)  # Don't modify position 0 (always 0)
                # Try to move the element to a nearby position
                delta = random.randint(-3, 3)
                new_val = new_row[pos] + delta
                # Ensure it stays in valid range and maintains strict increase
                lower_bound_val = new_row[pos - 1] + 1
                upper_bound_val = new_row[pos + 1] - 1 if pos < L - 1 else max_val
                if new_val < lower_bound_val:
                    new_val = lower_bound_val
                if new_val > upper_bound_val:
                    new_val = upper_bound_val
                new_row[pos] = new_val
            
            # Ensure strict increase
            valid = True
            for i in range(1, L):
                if new_row[i] <= new_row[i - 1]:
                    valid = False
                    break
            if not valid:
                continue
            
            # Compute new cost
            # Only need to check conflicts involving row r
            # For simplicity, recompute full cost (small enough)
            rows[r] = new_row
            new_cost, new_max_val, new_conflicts = compute_cost(rows)
            
            # Check if better
            if new_conflicts == 0 and new_max_val <= max_val:
                # Found a valid solution!
                return rows
            
            delta_cost = new_cost - current_cost
            if delta_cost < 0:
                # Accept: better
                current_cost = new_cost
                current_max_val = new_max_val
                current_conflicts = new_conflicts
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_max_val = new_max_val
                    best_conflicts = new_conflicts
                    best_rows = [r2[:] for r2 in rows]
            else:
                # Accept with probability exp(-delta/T)
                if random.random() < math.exp(-delta_cost / T):
                    current_cost = new_cost
                    current_max_val = new_max_val
                    current_conflicts = new_conflicts
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_max_val = new_max_val
                        best_conflicts = new_conflicts
                        best_rows = [r2[:] for r2 in rows]
                else:
                    # Reject: restore
                    rows[r] = old_row
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
            
            # Periodic restart: if stuck, do a random restart
            if current_conflicts > 0 and random.random() < 0.001:
                # Try a fresh random initialization
                rows = []
                used_diffs = set()
                for i in range(n_rows):
                    for attempt in range(200):
                        candidates = random.sample(range(1, max_val + 1), k)
                        candidates.sort()
                        row = [0] + candidates
                        rd = get_row_diffs(row)
                        if len(rd) == diffs_per_row and not (rd & used_diffs):
                            rows.append(row)
                            used_diffs |= rd
                            break
                    else:
                        row = [0]
                        x = 1
                        while len(row) < L:
                            new_diffs = {x - a for a in row}
                            if not new_diffs & used_diffs:
                                row.append(x)
                                used_diffs |= new_diffs
                            else:
                                x += 1
                        rows.append(row)
                current_cost, current_max_val, current_conflicts = compute_cost(rows)
                if current_conflicts == 0 and current_max_val <= max_val:
                    return rows
        
        # Return best found if valid
        if best_conflicts == 0 and best_max_val <= max_val:
            return best_rows
        
        return None
    
    # Strategy:
    # 1. Try to find a solution with max_val < 112 using SA.
    # 2. If not found, try max_val = 112.
    # 3. If still not found, use greedy fallback.
    
    best_rows = None
    best_max_val = float('inf')
    
    # First, get a greedy baseline
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    if is_fully_valid(greedy_rows):
        best_rows = [r[:] for r in greedy_rows]
        best_max_val = greedy_max
    
    # Try to improve with SA, targeting smaller max_val
    # Start from lower bound and go up
    start_M = lower_bound
    # But we know 112 is achievable, so we don't need to try below that if greedy is worse
    # Let's try to find something better than the current best
    target_max = min(best_max_val, 112)
    
    # Try SA for max_val values from lower_bound to target_max
    for M in range(start_M, target_max + 1):
        if time.time() > deadline:
            break
        
        # Give each M a time slice
        time_slice = min(15.0, deadline - time.time() - 5.0)
        if time_slice < 2.0:
            break
        
        result = simulated_annealing(n_rows, k, M, time_limit=time_slice)
        if result is not None:
            # Verify
            if is_fully_valid(result):
                result_max = max(max(r) for r in result)
                if result_max <= M:
                    best_rows = [r[:] for r in result]
                    best_max_val = result_max
                    break  # Found a valid solution with this M, stop
    
    # If we didn't find a better solution, try M = 112 specifically
    if best_max_val > 112:
        if time.time() < deadline - 5.0:
            time_slice = min(30.0, deadline - time.time() - 5.0)
            result = simulated_annealing(n_rows, k, 112, time_limit=time_slice)
            if result is not None and is_fully_valid(result):
                result_max = max(max(r) for r in result)
                if result_max <= 112:
                    best_rows = [r[:] for r in result]
                    best_max_val = result_max
    
    # If still no valid solution, use greedy (which should always produce a valid one)
    if best_rows is None or not is_fully_valid(best_rows):
        best_rows = greedy_rows
        best_max_val = greedy_max
    
    # Final verification
    if not is_fully_valid(best_rows):
        # This should never happen, but just in case
        best_rows = greedy_rows
    
    return [list(r) for r in best_rows]
