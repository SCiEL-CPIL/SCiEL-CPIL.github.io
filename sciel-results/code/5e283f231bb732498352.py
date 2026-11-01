<<<<<<< SEARCH
"""Difference triangle set (4,7) - local search with multi-start."""
import random
import itertools

def _row_differences(row):
    """Get all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i+1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid_row(row, used_diffs):
    """Check if row's differences don't conflict with used_diffs and are internally distinct."""
    diffs = _row_differences(row)
    return len(diffs) == len(row) * (len(row)-1) // 2 and not (diffs & used_diffs)

def _construct_row_greedy(k, used_diffs, start=0, end=200):
    """Greedy construction of a row with k+1 elements starting at 0."""
    row = [0]
    x = start
    while len(row) < k + 1:
        if x > end:
            return None
        new_diffs = {x - a for a in row}
        if not (new_diffs & used_diffs) and len(new_diffs) == len(row):
            row.append(x)
            used_diffs = used_diffs | new_diffs
        x += 1
    return row

def _construct_row_random(k, used_diffs, max_val=150):
    """Randomized construction of a valid row."""
    for _ in range(1000):
        row = [0]
        used = set(used_diffs)
        x = random.randint(1, max_val // 4)
        success = True
        while len(row) < k + 1:
            # Try to find next element
            found = False
            for _attempt in range(50):
                if x > max_val:
                    break
                new_diffs = {x - a for a in row}
                if len(new_diffs) == len(row) and not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    found = True
                    break
                x += 1
            if not found:
                success = False
                break
            x += random.randint(0, 5)
        if success:
            return row
    return None

def _max_entry(rows):
    return max(max(row) for row in rows)

def _total_differences(rows):
    """Count total distinct differences; should be n_rows * k*(k+1)//2 if valid."""
    all_diffs = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                all_diffs.add(row[j] - row[i])
    return len(all_diffs)

def _is_valid(rows, k):
    """Check full validity: all differences distinct across all rows."""
    all_diffs = set()
    expected = len(rows) * k * (k + 1) // 2
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return len(all_diffs) == expected

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all pairwise differences are distinct across all rows.
    Minimize the largest entry.
    """
    best_rows = None
    best_score = float('inf')
    
    # Multi-start local search
    for restart in range(30):
        # Phase 1: Generate initial solution
        used_diffs = set()
        rows = []
        success = True
        for r in range(n_rows):
            row = _construct_row_greedy(k, used_diffs, start=0, end=300)
            if row is None:
                success = False
                break
            rows.append(row)
            used_diffs |= _row_differences(row)
        
        if not success:
            continue
        
        # Phase 2: Local search to improve
        current_rows = rows
        current_score = _max_entry(current_rows)
        
        for iteration in range(5000):
            # Pick a random row to replace
            idx = random.randint(0, n_rows - 1)
            
            # Compute used diffs from other rows
            other_diffs = set()
            for i, row in enumerate(current_rows):
                if i != idx:
                    other_diffs |= _row_differences(row)
            
            # Try to construct a better row for position idx
            # We want a row with smaller max value
            current_row = current_rows[idx]
            current_max = max(current_row)
            
            # Try greedy with tighter bound
            new_row = _construct_row_greedy(k, other_diffs, start=0, end=current_max - 1)
            if new_row is not None:
                new_score = _max_entry(current_rows[:idx] + [new_row] + current_rows[idx+1:])
                if new_score < current_score:
                    current_rows[idx] = new_row
                    current_score = new_score
            
            # Try random construction
            new_row = _construct_row_random(k, other_diffs, max_val=current_max)
            if new_row is not None and max(new_row) < current_max:
                new_score = _max_entry(current_rows[:idx] + [new_row] + current_rows[idx+1:])
                if new_score < current_score:
                    current_rows[idx] = new_row
                    current_score = new_score
        
        if current_score < best_score:
            best_score = current_score
            best_rows = current_rows
    
    # Fallback: return greedy if no better found
    if best_rows is None:
        used_diffs = set()
        best_rows = []
        for _ in range(n_rows):
            row = _construct_row_greedy(k, used_diffs, start=0, end=500)
            if row is None:
                row = [0]
                x = 0
                while len(row) < k + 1:
                    x += 1
                    new = {x - a for a in row}
                    if not (new & used_diffs):
                        row.append(x)
                        used_diffs |= new
            best_rows.append(row)
            used_diffs |= _row_differences(row)
    
    return best_rows
=======
"""Difference triangle set (4,7) - Simulated Annealing with conflict tracking."""
import random
import math
import time

def _row_diffs(row):
    """Calculate set of differences for a single row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _total_conflicts(rows):
    """Count how many differences are duplicated across all rows."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) - len(set(all_diffs))

def _max_entry(rows):
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k):
    """Generate a valid initial solution using greedy method."""
    used_global = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            found = False
            # Try small numbers first to keep max entry low
            for val in range(1, 300):
                if val <= row[-1]:
                    continue
                new_row = row + [val]
                new_diffs = _row_diffs(new_row)
                # Check if new diffs are internally unique (guaranteed by set size if row is strict inc)
                # and don't conflict with global
                if len(new_diffs) == len(new_row) * (len(new_row) - 1) // 2:
                    if not (new_diffs & used_global):
                        row = new_row
                        used_global |= new_diffs
                        found = True
                        break
            if not found:
                # Fallback: pick a large number to ensure completion
                for val in range(row[-1] + 1, row[-1] + 1000):
                    new_row = row + [val]
                    new_diffs = _row_diffs(new_row)
                    if len(new_diffs) == len(new_row) * (len(new_row) - 1) // 2:
                        if not (new_diffs & used_global):
                            row = new_row
                            used_global |= new_diffs
                            found = True
                            break
                if not found:
                    # Last resort: random large
                    val = row[-1] + 100
                    row.append(val)
                    used_global |= _row_diffs(row)
        rows.append(row)
    return rows

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all pairwise differences are distinct across all rows.
    Minimize the largest entry.
    """
    # Parameters
    start_temp = 50.0
    end_temp = 0.1
    cooling_rate = 0.9995
    max_iterations = 100000
    time_limit = 110.0
    
    # Initialize
    current_rows = _greedy_init(n_rows, k)
    current_max = _max_entry(current_rows)
    current_conflicts = _total_conflicts(current_rows)
    current_score = current_max + current_conflicts * 1000
    
    best_valid_rows = [r[:] for r in current_rows] if current_conflicts == 0 else None
    best_valid_score = current_max if current_conflicts == 0 else float('inf')
    
    # Precompute row diffs
    row_diffs_list = [_row_diffs(r) for r in current_rows]
    
    start_time = time.time()
    temp = start_temp
    iter_count = 0
    
    while temp > end_temp and iter_count < max_iterations and (time.time() - start_time) < time_limit:
        iter_count += 1
        r_idx = random.randint(0, n_rows - 1)
        # Position 0 is fixed at 0. Pick from 1 to k
        pos_idx = random.randint(1, k)
        
        old_val = current_rows[r_idx][pos_idx]
        
        # Determine valid range for new value
        min_val = current_rows[r_idx][pos_idx - 1] + 1
        max_val = current_rows[r_idx][pos_idx + 1] - 1 if pos_idx < k else current_max + 100
        
        if min_val > max_val:
            temp *= cooling_rate
            continue
            
        # Bias towards smaller values
        if random.random() < 0.6 and old_val > min_val:
            new_val = random.randint(min_val, old_val - 1)
        else:
            upper_bound = min(max_val, old_val + 10)
            if old_val < upper_bound:
                new_val = random.randint(old_val + 1, upper_bound)
            else:
                temp *= cooling_rate
                continue
        
        if new_val == old_val:
            temp *= cooling_rate
            continue
            
        old_diffs = row_diffs_list[r_idx]
        
        new_row = current_rows[r_idx][:]
        new_row[pos_idx] = new_val
        new_diffs = _row_diffs(new_row)
        
        # Update state
        current_rows[r_idx] = new_row
        row_diffs_list[r_idx] = new_diffs
        
        new_max = _max_entry(current_rows)
        new_conflicts = _total_conflicts(current_rows)
        
        new_score = new_max + new_conflicts * 1000
        old_score = current_max + current_conflicts * 1000
        
        delta = new_score - old_score
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_max = new_max
            current_conflicts = new_conflicts
            current_score = new_score
            
            if current_conflicts == 0:
                if current_max < best_valid_score:
                    best_valid_score = current_max
                    best_valid_rows = [r[:] for r in current_rows]
        else:
            # Reject: Revert
            current_rows[r_idx][pos_idx] = old_val
            row_diffs_list[r_idx] = old_diffs
            
        temp *= cooling_rate

    if best_valid_rows is not None:
        return best_valid_rows
    else:
        # Fallback: Return a valid greedy solution
        return _greedy_init(n_rows, k)
>>>>>>> REPLACE
