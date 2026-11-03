import time
import random
import sys

def _row_diffs_set(row):
    """Compute all positive differences in a row as a set."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i):
            diffs.add(row[i] - row[j])
    return diffs

def _check_valid(rows, k):
    """Verify that all differences across all rows are distinct."""
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i-1]:
                return False
        for d in _row_diffs_set(row):
            if d in all_diffs:
                return False
            all_diffs.add(d)
    return True

def _compute_scope(rows):
    """Compute the scope (largest entry) of a solution."""
    return max(max(row) for row in rows)

def _greedy_initial(n_rows, k, rng=None):
    """
    Greedy construction to get a valid initial solution.
    Picks the smallest valid x at each step.
    """
    if rng is None:
        rng = random.Random()
    
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    row.append(x)
                    used |= new_diffs
                    break
                x += 1
        rows.append(row)
    
    return rows

def _sa_solve(n_rows, k, deadline, rng=None):
    """
    Simulated annealing to minimize the scope (largest entry).
    Uses two types of moves:
    1. Element move: change one element in one row
    2. Row regeneration: rebuild an entire row using internal SA
    """
    if rng is None:
        rng = random.Random()
    
    # Start with greedy solution
    rows = _greedy_initial(n_rows, k, rng)
    best_rows = [r[:] for r in rows]
    best_scope = _compute_scope(best_rows)
    
    # Temperature schedule
    T_start = best_scope * 0.5
    T_end = 1.0
    cooling_rate = 0.9995
    
    T = T_start
    step = 0
    
    while time.time() < deadline:
        T *= cooling_rate
        if T < T_end:
            T = T_end
        
        step += 1
        
        # Choose move type
        # 70% element move, 30% row regeneration
        if rng.random() < 0.7:
            # Element move: pick a random row and a random non-first element
            row_idx = rng.randint(0, n_rows - 1)
            if len(rows[row_idx]) <= 1:
                continue
            elem_idx = rng.randint(1, len(rows[row_idx]) - 1)
            
            old_val = rows[row_idx][elem_idx]
            # Try to decrease or slightly increase
            delta = rng.randint(-5, 5)
            new_val = old_val + delta
            
            # Ensure increasing order
            if elem_idx > 0 and new_val <= rows[row_idx][elem_idx - 1]:
                continue
            if elem_idx < len(rows[row_idx]) - 1 and new_val >= rows[row_idx][elem_idx + 1]:
                continue
            if new_val <= 0:
                continue
            
            # Check validity of new row
            new_row = rows[row_idx][:]
            new_row[elem_idx] = new_val
            
            # Check if new row is strictly increasing
            valid_row = True
            for i in range(len(new_row) - 1):
                if new_row[i] >= new_row[i + 1]:
                    valid_row = False
                    break
            if not valid_row:
                continue
            
            # Check global validity
            new_diffs = _row_diffs_set(new_row)
            conflict = False
            for d in new_diffs:
                # Check if d is used by other rows
                for other_idx in range(n_rows):
                    if other_idx == row_idx:
                        continue
                    if d in _row_diffs_set(rows[other_idx]):
                        conflict = True
                        break
                if conflict:
                    break
            
            if conflict:
                continue
            
            # Compute new scope
            new_scope = max(max(r) for r in rows) if all(r == rows[r] for r in range(n_rows)) else 0
            # Actually need to recompute properly
            candidate_rows = [r[:] for r in rows]
            candidate_rows[row_idx] = new_row
            candidate_scope = _compute_scope(candidate_rows)
            
            # Acceptance criterion
            delta_scope = candidate_scope - best_scope
            if delta_scope < 0 or rng.random() < pow(2.718281828, -delta_scope / T):
                rows = candidate_rows
                if candidate_scope < best_scope:
                    best_rows = [r[:] for r in rows]
                    best_scope = candidate_scope
        else:
            # Row regeneration: rebuild one row entirely
            row_idx = rng.randint(0, n_rows - 1)
            
            # Collect diffs from other rows
            other_diffs = set()
            for other_idx in range(n_rows):
                if other_idx != row_idx:
                    other_diffs |= _row_diffs_set(rows[other_idx])
            
            # Try to build a new row that avoids other_diffs
            # Use a mini-SA or greedy with randomization
            new_row = [0]
            row_diffs = set()
            success = False
            
            for attempt in range(10):
                new_row = [0]
                row_diffs = set()
                success = True
                for pos in range(1, k + 1):
                    last_val = new_row[-1]
                    # Try candidates from last_val+1 upward, with some randomization
                    found = False
                    start_x = last_val + 1
                    end_x = last_val + rng.randint(1, 20)
                    
                    candidates = list(range(start_x, end_x + 1))
                    rng.shuffle(candidates)
                    
                    for x in candidates:
                        valid = True
                        new_d = set()
                        for a in new_row:
                            d = x - a
                            if d in other_diffs or d in row_diffs:
                                valid = False
                                break
                            new_d.add(d)
                        if valid:
                            new_row.append(x)
                            row_diffs |= new_d
                            found = True
                            break
                    
                    if not found:
                        success = False
                        break
                
                if success:
                    break
            
            if not success:
                continue
            
            # Check if new row is better
            candidate_rows = [r[:] for r in rows]
            candidate_rows[row_idx] = new_row
            candidate_scope = _compute_scope(candidate_rows)
            
            # Also check full validity
            if not _check_valid(candidate_rows, k):
                continue
            
            delta_scope = candidate_scope - best_scope
            if delta_scope < 0 or rng.random() < pow(2.718281828, -delta_scope / T):
                rows = candidate_rows
                if candidate_scope < best_scope:
                    best_rows = [r[:] for r in rows]
                    best_scope = candidate_scope
    
    return best_rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    rng = random.Random(42)
    
    # Get initial greedy solution
    initial_rows = _greedy_initial(n_rows, k, rng)
    initial_scope = _compute_scope(initial_rows)
    
    # Run SA
    sa_rows = _sa_solve(n_rows, k, deadline, rng)
    
    # Verify validity
    if _check_valid(sa_rows, k):
        return sa_rows
    
    # Fallback to initial
    return initial_rows
