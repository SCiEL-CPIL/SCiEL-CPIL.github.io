"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _build_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with no difference collision."""
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
        rows.append(row)
    return rows

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _row_diffs_list(row):
    out = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            out.append(row[j] - row[i])
    return out

def _check_row_valid(row, k):
    """Check if a single row is strictly increasing and starts at 0."""
    if len(row) != k + 1:
        return False
    if row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i - 1]:
            return False
    return True

def _try_move(cur, ri, pi, new_val, cur_used, cur_row_diffs, k, n_rows):
    """
    Try to move position pi in row ri to new_val.
    Returns (cand_row, new_diffs, new_max) if valid, else None.
    """
    # Check strict increase within row
    if new_val <= cur[ri][pi - 1]:
        return None
    if pi < k and new_val >= cur[ri][pi + 1]:
        return None
    
    # Build candidate row
    cand_row = cur[ri][:]
    cand_row[pi] = new_val
    
    # Check row validity
    if not _check_row_valid(cand_row, k):
        return None
    
    # Compute new diffs for this row
    new_diffs = _row_diffs(cand_row)
    
    # Check for conflicts with other rows' diffs
    other_diffs = cur_used - cur_row_diffs[ri]
    if new_diffs & other_diffs:
        return None
    
    # Check internal uniqueness in new_diffs (Sidon set property)
    diff_list = _row_diffs_list(cand_row)
    if len(diff_list) != len(set(diff_list)):
        return None
    
    # Compute new max
    cand_max = max(max(r) for r in cur if r is not cand_row) if n_rows > 1 else 0
    cand_max = max(cand_max, max(cand_row))
    
    return (cand_row, new_diffs, cand_max)

def _resolve_row_given_others(ri, rows, used_by_others, k):
    """
    Try to re-solve row ri given all other rows are fixed.
    Uses backtracking to find a valid row with minimal max entry.
    Returns the new row if found, else None.
    """
    # We want to find a row of k+1 increasing integers starting at 0
    # such that all pairwise differences are not in used_by_others
    # and all pairwise differences within the row are distinct (Sidon)
    
    # Greedy approach with limited backtracking
    best_row = None
    best_max = float('inf')
    
    # Try multiple random starting points for the search
    for attempt in range(50):
        row = [0]
        x = 0
        found = True
        while len(row) < k + 1:
            x += 1
            # Check if x can be added
            new_diffs = {x - a for a in row}
            if new_diffs & used_by_others:
                continue
            # Check Sidon property: no duplicate differences within row
            diff_list = []
            for a in row:
                diff_list.append(x - a)
            # Check all differences are distinct
            if len(diff_list) != len(set(diff_list)):
                continue
            row.append(x)
            if x >= best_max and best_row is not None:
                # Can't improve, break early
                found = False
                break
        if found and len(row) == k + 1:
            row_max = row[-1]
            if row_max < best_max:
                best_max = row_max
                best_row = row[:]
    
    return best_row

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 105  # safety margin before 120s limit
    random.seed(42)
    
    # Start from greedy
    rows = _build_greedy(n_rows, k)
    best = [r[:] for r in rows]
    best_val = _max_entry(best)
    best_used = set()
    for r in best:
        best_used |= _row_diffs(r)
    
    # Local search with incremental difference tracking
    cur = [r[:] for r in best]
    cur_val = best_val
    cur_used = set(best_used)
    
    # Track diffs per row for efficient updates
    cur_row_diffs = [_row_diffs(r) for r in cur]
    
    T = 30.0
    T_min = 0.01
    cool = 0.99995
    
    step = 0
    no_improve_steps = 0
    max_no_improve = 50000
    
    while time.time() < deadline and no_improve_steps < max_no_improve:
        step += 1
        
        # Pick a random row and a random position to perturb
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        old_val = cur[ri][pi]
        
        # Generate candidate moves
        candidates = []
        
        # Try decreasing values
        for d in range(1, 20):
            new_val = old_val - d
            if new_val <= cur[ri][pi - 1]:
                break
            candidates.append(new_val)
        
        # Try small increases
        for d in range(1, 10):
            new_val = old_val + d
            if pi < k and new_val >= cur[ri][pi + 1]:
                break
            candidates.append(new_val)
        
        # Occasionally try larger jumps
        if random.random() < 0.4:
            for d in range(20, 50):
                new_val = old_val - d
                if new_val > cur[ri][pi - 1]:
                    candidates.append(new_val)
                new_val = old_val + d
                if pi < k and new_val < cur[ri][pi + 1]:
                    candidates.append(new_val)
        
        # Occasionally try row re-solve
        if random.random() < 0.15:
            other_diffs = cur_used - cur_row_diffs[ri]
            new_row = _resolve_row_given_others(ri, cur, other_diffs, k)
            if new_row is not None:
                # Compute max for this candidate
                cand_max = max(max(r) for r in cur if r is not cur[ri]) if n_rows > 1 else 0
                cand_max = max(cand_max, max(new_row))
                new_diffs = _row_diffs(new_row)
                # Check no conflicts
                if not (new_diffs & other_diffs):
                    # Check Sidon
                    diff_list = _row_diffs_list(new_row)
                    if len(diff_list) == len(set(diff_list)):
                        # Accept if it improves or is same
                        delta_e = cand_max - cur_val
                        accept = False
                        if delta_e < 0:
                            accept = True
                        elif delta_e == 0:
                            if random.random() < 0.3:
                                accept = True
                        else:
                            if T > T_min:
                                prob = 1.0 / (1.0 + delta_e / max(T, 0.01))
                                if random.random() < prob:
                                    accept = True
                        
                        if accept:
                            cur_used -= cur_row_diffs[ri]
                            cur_used |= new_diffs
                            cur[ri] = new_row
                            cur_row_diffs[ri] = new_diffs
                            cur_val = cand_max
                            
                            if cur_val < best_val:
                                best_val = cur_val
                                best = [r[:] for r in cur]
                                best_used = set(cur_used)
                                no_improve_steps = 0
                            else:
                                no_improve_steps += 1
                            
                            T = max(T_min, T * cool)
                            continue
        
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            result = _try_move(cur, ri, pi, new_val, cur_used, cur_row_diffs, k, n_rows)
            if result is None:
                continue
            
            cand_row, new_diffs, cand_max = result
            
            # Acceptance
            delta_e = cand_max - cur_val
            accept = False
            if delta_e < 0:
                accept = True
            elif delta_e == 0:
                if random.random() < 0.3:
                    accept = True
            else:
                if T > T_min:
                    prob = 1.0 / (1.0 + delta_e / max(T, 0.01))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                cur_used -= cur_row_diffs[ri]
                cur_used |= new_diffs
                cur[ri] = cand_row
                cur_row_diffs[ri] = new_diffs
                cur_val = cand_max
                
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    best_used = set(cur_used)
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                
                moved = True
                break
        
        if not moved:
            no_improve_steps += 1
        
        # Cool down
        T = max(T_min, T * cool)
        
        # Periodic restart
        if no_improve_steps > 20000:
            cur = [r[:] for r in best]
            cur_val = best_val
            cur_used = set(best_used)
            cur_row_diffs = [_row_diffs(r) for r in cur]
            T = 30.0
            no_improve_steps = 0
    
    # Final polish: try to re-solve each row to minimize its max
    for _ in range(3):
        for ri in range(n_rows):
            other_diffs = cur_used - cur_row_diffs[ri]
            new_row = _resolve_row_given_others(ri, cur, other_diffs, k)
            if new_row is not None:
                new_diffs = _row_diffs(new_row)
                if not (new_diffs & other_diffs):
                    diff_list = _row_diffs_list(new_row)
                    if len(diff_list) == len(set(diff_list)):
                        cur_used -= cur_row_diffs[ri]
                        cur_used |= new_diffs
                        cur[ri] = new_row
                        cur_row_diffs[ri] = new_diffs
                        cur_val = _max_entry(cur)
                        if cur_val < best_val:
                            best_val = cur_val
                            best = [r[:] for r in cur]
                            best_used = set(cur_used)
    
    # Ensure best is valid
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    return best
# EVOLVE-BLOCK-END
