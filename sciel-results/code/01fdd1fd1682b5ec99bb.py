import time
import random

def construct(n_rows, k):
    deadline = time.time() + 120  # 120 second deadline
    
    # Greedy construction to get initial solution
    rows = _greedy_construct(n_rows, k)
    used = _compute_used_diffs(rows)
    best_scope = max(max(row) for row in rows)
    best_rows = [row[:] for row in rows]
    
    # Local search to improve
    rows = _local_search(rows, used, best_scope, deadline)
    final_scope = max(max(row) for row in rows)
    
    if final_scope < best_scope:
        best_rows = [row[:] for row in rows]
    
    return best_rows


def _greedy_construct(n_rows, k):
    """Construct rows greedily, minimizing each new element."""
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


def _compute_used_diffs(rows):
    """Compute all used positive differences."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    return used


def _local_search(rows, used, current_scope, deadline):
    """Local search to reduce the maximum scope."""
    rows = [row[:] for row in rows]
    best_scope = current_scope
    
    while time.time() < deadline - 5:
        improved = False
        
        # Try to reduce the maximum element
        max_row_idx = max(range(len(rows)), key=lambda i: max(rows[i]))
        max_val = max(rows[max_row_idx])
        
        # Try replacing the maximum element with a smaller one
        for new_val in range(max_val - 1, 0, -1):
            if new_val in rows[max_row_idx]:
                continue
            
            # Check if replacing works
            old_row = rows[max_row_idx]
            old_diffs = set()
            for i in range(len(old_row)):
                for j in range(i + 1, len(old_row)):
                    old_diffs.add(old_row[j] - old_row[i])
            
            new_row = old_row[:]
            new_row[-1] = new_val
            new_row.sort()
            
            new_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_diffs.add(new_row[j] - new_row[i])
            
            # Check for conflicts
            other_diffs = used - old_diffs
            if not (new_diffs & other_diffs):
                # Valid replacement
                used = (used - old_diffs) | new_diffs
                rows[max_row_idx] = new_row
                new_scope = max(max(row) for row in rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                improved = True
                break
        
        if not improved:
            # Try swapping elements between rows
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    for ai in range(1, len(rows[i])):
                        for aj in range(1, len(rows[j])):
                            old_ri = rows[i][ai]
                            old_rj = rows[j][aj]
                            
                            if old_ri == old_rj:
                                continue
                            
                            # Compute diffs before
                            diffs_i_before = set()
                            for x in range(len(rows[i])):
                                for y in range(x + 1, len(rows[i])):
                                    diffs_i_before.add(rows[i][y] - rows[i][x])
                            
                            diffs_j_before = set()
                            for x in range(len(rows[j])):
                                for y in range(x + 1, len(rows[j])):
                                    diffs_j_before.add(rows[j][y] - rows[j][x])
                            
                            # Try swap
                            rows[i][ai] = old_rj
                            rows[j][aj] = old_ri
                            rows[i].sort()
                            rows[j].sort()
                            
                            diffs_i_after = set()
                            for x in range(len(rows[i])):
                                for y in range(x + 1, len(rows[i])):
                                    diffs_i_after.add(rows[i][y] - rows[i][x])
                            
                            diffs_j_after = set()
                            for x in range(len(rows[j])):
                                for y in range(x + 1, len(rows[j])):
                                    diffs_j_after.add(rows[j][y] - rows[j][x])
                            
                            # Check validity
                            other_diffs = used - diffs_i_before - diffs_j_before
                            if not (diffs_i_after & other_diffs) and not (diffs_j_after & other_diffs) and not (diffs_i_after & diffs_j_after):
                                # Valid swap
                                used = (used - diffs_i_before - diffs_j_before) | diffs_i_after | diffs_j_after
                                new_scope = max(max(row) for row in rows)
                                if new_scope < best_scope:
                                    best_scope = new_scope
                                improved = True
                                break
                            
                            # Undo swap
                            rows[i][ai] = old_ri
                            rows[j][aj] = old_rj
                            rows[i].sort()
                            rows[j].sort()
                            
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            # No improvement possible, try random restarts with perturbation
            if time.time() < deadline - 10:
                # Perturb and try again
                perturbed = _perturb(rows, used)
                if perturbed is not None:
                    rows, used = perturbed
                    continue
            else:
                break
    
    return rows


def _perturb(rows, used):
    """Perturb the current solution to escape local optima."""
    # Try a random replacement
    for attempt in range(50):
        new_rows = [row[:] for row in rows]
        row_idx = random.randint(0, len(new_rows) - 1)
        pos = random.randint(1, len(new_rows[row_idx]) - 1)
        
        old_val = new_rows[row_idx][pos]
        new_val = old_val + random.randint(-10, 10)
        
        if new_val <= 0 or new_val == old_val:
            continue
        
        # Check if new_val is already in the row
        if new_val in new_rows[row_idx]:
            continue
        
        new_rows[row_idx][pos] = new_val
        new_rows[row_idx].sort()
        
        # Check validity
        old_diffs = set()
        for i in range(len(rows[row_idx])):
            for j in range(i + 1, len(rows[row_idx])):
                old_diffs.add(rows[row_idx][j] - rows[row_idx][i])
        
        new_diffs = set()
        for i in range(len(new_rows[row_idx])):
            for j in range(i + 1, len(new_rows[row_idx])):
                new_diffs.add(new_rows[row_idx][j] - new_rows[row_idx][i])
        
        other_diffs = used - old_diffs
        if not (new_diffs & other_diffs):
            new_used = (used - old_diffs) | new_diffs
            return new_rows, new_used
    
    return None
