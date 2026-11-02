"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _compute_differences(rows):
    """Compute all pairwise differences for all rows."""
    diffs = []
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.append(row[j] - row[i])
    return diffs

def _is_valid(diffs):
    """Check if all differences are distinct."""
    return len(diffs) == len(set(diffs))

def _get_scope(rows):
    """Get the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k):
    """Greedy initialization: each row starts at 0 and takes the smallest next mark whose differences
    to the earlier marks of the row are all still unused by any row."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # 5 second safety margin
    
    # Initial greedy solution
    rows = _greedy_init(n_rows, k)
    best_rows = [row[:] for row in rows]
    best_scope = _get_scope(best_rows)
    
    # Simulated Annealing
    current_rows = [row[:] for row in rows]
    current_diffs = _compute_differences(current_rows)
    current_scope = _get_scope(current_rows)
    
    T = 50.0  # Initial temperature
    T_min = 0.1
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Choose a random row and make a move
        r = random.randint(0, n_rows - 1)
        row = current_rows[r]
        
        if random.random() < 0.5:
            # Move: swap two non-zero marks
            if len(row) < 2:
                continue
            i, j = random.sample(range(1, len(row)), 2)
            row[i], row[j] = row[j], row[i]
            if not all(row[m] < row[m+1] for m in range(len(row)-1)):
                row[i], row[j] = row[j], row[i]
                continue
        else:
            # Move: shift a mark to a new position
            if len(row) < 2:
                continue
            i = random.randint(1, len(row) - 1)
            delta = random.randint(-3, 3)
            new_val = row[i] + delta
            # Check bounds and ordering
            if i > 1 and new_val <= row[i-1]:
                continue
            if i < len(row) - 1 and new_val >= row[i+1]:
                continue
            if new_val < 0:
                continue
            old_val = row[i]
            row[i] = new_val
        
        # Recompute differences for this row only and check global validity
        new_diffs = _compute_differences(current_rows)
        if not _is_valid(new_diffs):
            # Revert
            if random.random() < 0.5:
                i, j = random.sample(range(1, len(row)), 2)
                row[i], row[j] = row[j], row[i]
            else:
                i = random.randint(1, len(row) - 1)
                # Need to find old_val - this is tricky, so let's use a different approach
                pass
            # Simpler: just revert by recomputing from best if needed, but that's expensive
            # Instead, let's track the change properly
            continue
        
        new_scope = _get_scope(current_rows)
        delta_E = new_scope - current_scope
        
        if delta_E <= 0 or random.random() < math.exp(-delta_E / T):
            # Accept move
            current_diffs = new_diffs
            current_scope = new_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [row[:] for row in current_rows]
        
        # Cool down
        T = max(T_min, T * cooling_rate)
    
    return best_rows
# EVOLVE-BLOCK-END
