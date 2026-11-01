"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _get_diffs(row):
    """Get all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i+1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    all_diffs = set()
    for row in rows:
        diffs = _get_diffs(row)
        if all_diffs & diffs:
            return False
        all_diffs |= diffs
    return True

def _max_entry(rows):
    return max(max(row) for row in rows)

def _greedy_row(used_diffs, k, max_limit=200):
    """Greedily construct a row avoiding used_diffs."""
    row = [0]
    x = 1
    while len(row) < k + 1:
        new_diffs = {x - a for a in row}
        if not new_diffs & used_diffs:
            row.append(x)
            used_diffs |= new_diffs
        x += 1
        if x > max_limit:
            return None
    return row

def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0
    with all pairwise differences distinct across all rows."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial greedy solution
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used_diffs, k)
        if row is None:
            break
        rows.append(row)
    
    if len(rows) < n_rows:
        # Fallback: return what we have (shouldn't happen for reasonable inputs)
        return rows
    
    best_rows = [row[:] for row in rows]
    best_score = _max_entry(best_rows)
    
    # Local search: try to reduce the maximum entry
    while time.time() < deadline:
        current = [row[:] for row in best_rows]
        current_score = _max_entry(current)
        
        # Find the row with the maximum entry
        max_row_idx = max(range(n_rows), key=lambda i: max(current[i]))
        max_val = max(current[max_row_idx])
        
        # Try to reduce elements in the max row
        improved = False
        for pos in range(k, 0, -1):  # Try from the end
            old_val = current[max_row_idx][pos]
            # Try decreasing
            for new_val in range(old_val - 1, current[max_row_idx][pos-1], -1):
                test_row = current[max_row_idx][:]
                test_row[pos] = new_val
                test_rows = current[:max_row_idx] + [test_row] + current[max_row_idx+1:]
                if _is_valid(test_rows):
                    current[max_row_idx] = test_row
                    new_score = _max_entry(current)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            if current_score < best_score:
                best_score = current_score
                best_rows = [row[:] for row in current]
        else:
            # Try random moves
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            old_val = current[r_idx][pos]
            # Try a range of new values
            candidates = []
            for delta in range(-20, 21):
                new_val = old_val + delta
                if new_val > current[r_idx][pos-1]:
                    if pos < k and new_val < current[r_idx][pos+1]:
                        candidates.append(new_val)
                    elif pos == k:
                        candidates.append(new_val)
            
            random.shuffle(candidates)
            for new_val in candidates[:5]:
                test_row = current[r_idx][:]
                test_row[pos] = new_val
                test_rows = current[:r_idx] + [test_row] + current[r_idx+1:]
                if _is_valid(test_rows):
                    new_score = _max_entry(test_rows)
                    if new_score <= current_score:
                        current[r_idx] = test_row
                        if new_score < best_score:
                            best_score = new_score
                            best_rows = [row[:] for row in current]
                        break
    
    return best_rows
# EVOLVE-BLOCK-END
