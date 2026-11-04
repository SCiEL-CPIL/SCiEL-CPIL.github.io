"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import math
from collections import deque

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # Helper to get max entry
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    # Validate a set of rows
    def validate(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i + 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True
    
    # Get the set of differences for a single row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Greedy baseline
    def greedy_solution(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            if randomize:
                if len(row) > 2 and random.random() < 0.3:
                    i1, i2 = random.sample(range(1, len(row)), 2)
                    row[i1], row[i2] = row[i2], row[i1]
                    row = [0] + sorted(row[1:])
            rows.append(row)
        return rows
    
    # Get a random valid row given the set of already-used differences
    def random_row(used_diffs, max_val=None):
        if max_val is None:
            max_val = 200
        for _ in range(50):  # Try up to 50 times
            row = [0]
            for pos in range(k):
                # Try to find a value that doesn't create conflicts
                candidates = list(range(row[-1] + 1, max_val + 1))
                random.shuffle(candidates)
                found = False
                for v in candidates:
                    new_diffs = {v - a for a in row}
                    if len(new_diffs) != len(row) + 1:  # Internal conflict
                        continue
                    if new_diffs & used_diffs:  # Global conflict
                        continue
                    row.append(v)
                    found = True
                    break
                if not found:
                    break
            if len(row) == k + 1:
                return row
        return None
    
    # Local search: try to improve by replacing rows
    def local_search(rows, deadline_t, max_iter=200):
        rows = [list(r) for r in rows]
        used_diffs = set()
        for row in rows:
            used_diffs |= row_diffs(row)
        
        best_rows = [list(r) for r in rows]
        best_max = get_max_entry(best_rows)
        
        for iteration in range(max_iter):
            if time.time() > deadline_t:
                break
            
            # Pick a random row to try to improve
            row_idx = random.randint(0, n_rows - 1)
            old_row = list(rows[row_idx])
            old_diffs = row_diffs(old_row)
            
            # Remove this row's diffs temporarily
            used_diffs -= old_diffs
            
            # Try to find a better row
            current_max = old_row[-1]
            improved = False
            
            # Try to find a row with smaller max
            for target_max in range(max(2, current_max - 1), 0, -1):
                new_row = random_row(used_diffs, max_val=target_max)
                if new_row is not None:
                    # Check if this improves the overall max
                    rows[row_idx] = new_row
                    new_max = get_max_entry(rows)
                    if new_max < best_max:
                        best_max = new_max
                        best_rows = [list(r) for r in rows]
                        used_diffs |= row_diffs(new_row)
                        improved = True
                        break
                    else:
                        # Revert
                        rows[row_idx] = old_row
                        used_diffs |= old_diffs
                        break
            
            if not improved:
                # Restore the original diffs
                used_diffs |= old_diffs
        
        return best_rows, best_max
    
    # Simulated annealing: global optimization
    def simulated_annealing(initial_rows, deadline_t, max_iter=5000):
        rows = [list(r) for r in initial_rows]
        used_diffs = set()
        for row in rows:
            used_diffs |= row_diffs(row)
        
        best_rows = [list(r) for r in rows]
        best_max = get_max_entry(best_rows)
        
        current_temp = 10.0
        cool_rate = 0.999
        
        for iteration in range(max_iter):
            if time.time() > deadline_t:
                break
            
            # Pick a random row to modify
            row_idx = random.randint(0, n_rows - 1)
            old_row = list(rows[row_idx])
            old_diffs = row_diffs(old_row)
            
            # Remove this row's diffs
            used_diffs -= old_diffs
            
            # Generate a new row
            new_row = random_row(used_diffs, max_val=200)
            
            if new_row is None:
                # Revert
                used_diffs |= old_diffs
                continue
            
            new_diffs = row_diffs(new_row)
            
            # Check for conflicts
            if new_diffs & used_diffs:
                # Revert
                used_diffs |= old_diffs
                continue
            
            # Calculate energy change
            old_max = get_max_entry(rows)
            rows[row_idx] = new_row
            new_max = get_max_entry(rows)
            
            delta = new_max - old_max
            
            if delta < 0 or random.random() < math.exp(-delta / current_temp):
                # Accept the move
                used_diffs |= new_diffs
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [list(r) for r in rows]
            else:
                # Revert
                rows[row_idx] = old_row
                used_diffs |= old_diffs
            
            current_temp *= cool_rate
        
        return best_rows, best_max
    
    # Main search
    best_rows = greedy_solution()
    best_max = get_max_entry(best_rows)
    
    # Phase 1: Local search
    if time.time() < deadline - 20:
        improved_rows, improved_max = local_search(best_rows, deadline - 20, max_iter=100)
        if improved_max < best_max:
            best_max = improved_max
            best_rows = improved_rows
    
    # Phase 2: Simulated annealing
    if time.time() < deadline - 10:
        sa_rows, sa_max = simulated_annealing(best_rows, deadline - 10, max_iter=5000)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
    
    # Phase 3: More local search
    if time.time() < deadline - 5:
        improved_rows, improved_max = local_search(best_rows, deadline - 5, max_iter=50)
        if improved_max < best_max:
            best_max = improved_max
            best_rows = improved_rows
    
    # Final validation
    if not validate(best_rows):
        # Fallback to pure greedy
        best_rows = greedy_solution()
        best_max = get_max_entry(best_rows)
    
    return best_rows

# EVOLVE-BLOCK-END
