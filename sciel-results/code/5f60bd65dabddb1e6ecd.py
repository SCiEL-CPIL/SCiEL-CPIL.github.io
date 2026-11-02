"""Improved difference triangle set solver using simulated annealing with smart moves."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # For the specific case (4,7): k=7 means 8 entries per row, n_rows=4
    # Total differences per row: C(8,2) = 28, total = 112
    # Best known scope: 134, counting bound: 112
    
    # Initialize with a good greedy solution
    best_rows = _greedy_init(n_rows, k)
    best_scope = max(max(row) for row in best_rows)
    
    # Simulated annealing
    current_rows = [row[:] for row in best_rows]
    current_scope = best_scope
    
    # Precompute used differences for current solution
    current_used = _compute_all_diffs(current_rows)
    current_conflicts = len(current_used) - (n_rows * k * (k + 1) // 2)
    
    t = 0.1
    cooling = 0.99995
    min_temp = 0.0001
    
    while time.time() < deadline and t > min_temp:
        # Check time
        if time.time() > deadline - 5.0:
            break
            
        # Choose a move
        move_type = random.random()
        new_rows = [row[:] for row in current_rows]
        
        if move_type < 0.4:
            # Change one entry in one row
            r = random.randint(0, n_rows - 1)
            j = random.randint(1, k)  # not the first element (which is 0)
            old_val = new_rows[r][j]
            # Try to find a new value that reduces conflicts
            # Search in a range around current value
            search_range = max(10, current_scope // 10)
            candidates = []
            for delta in range(-search_range, search_range + 1):
                nv = old_val + delta
                if nv > 0 and nv < current_scope + 20:
                    # Check if it maintains strict increase
                    if (j == 0 or nv > new_rows[r][j-1]) and \
                       (j == k or nv < new_rows[r][j+1]):
                        candidates.append(nv)
            if candidates:
                new_val = random.choice(candidates)
                new_rows[r][j] = new_val
        elif move_type < 0.6:
            # Swap two entries in one row (maintain order)
            r = random.randint(0, n_rows - 1)
            i, j = random.sample(range(1, k + 1), 2)
            if i > j:
                i, j = j, i
            new_rows[r][i], new_rows[r][j] = new_rows[r][j], new_rows[r][i]
            # Ensure strict increase
            if not _is_strictly_increasing(new_rows[r]):
                # Revert
                new_rows[r][i], new_rows[r][j] = new_rows[r][j], new_rows[r][i]
                continue
        elif move_type < 0.75:
            # Rebuild one row greedily with current used set
            r = random.randint(0, n_rows - 1)
            other_used = _compute_all_diffs([row for i, row in enumerate(current_rows) if i != r])
            new_row = _greedy_row(k, other_used)
            if new_row:
                new_rows[r] = new_row
        else:
            # Random restart of one row
            r = random.randint(0, n_rows - 1)
            other_used = _compute_all_diffs([row for i, row in enumerate(current_rows) if i != r])
            new_row = _random_row(k, current_scope, other_used)
            if new_row:
                new_rows[r] = new_row
        
        # Compute new scope and conflicts
        new_scope = max(max(row) for row in new_rows)
        new_used = _compute_all_diffs(new_rows)
        new_conflicts = len(new_used) - (n_rows * k * (k + 1) // 2)
        
        # Acceptance criterion: prefer lower scope, then lower conflicts
        if new_conflicts == 0 and new_scope < current_scope:
            # Valid improvement in scope
            current_rows = new_rows
            current_scope = new_scope
            current_used = new_used
            current_conflicts = new_conflicts
            if current_scope < best_scope:
                best_rows = [row[:] for row in current_rows]
                best_scope = current_scope
        elif new_conflicts == 0 and new_scope == current_scope:
            # Same scope, no conflicts - accept
            current_rows = new_rows
            current_used = new_used
        elif new_conflicts < current_conflicts:
            # Fewer conflicts
            accept_prob = 1.0
        elif new_conflicts == current_conflicts:
            # Same conflicts, prefer lower scope
            if new_scope < current_scope:
                accept_prob = 1.0
            else:
                accept_prob = min(1.0, t * 0.5)
        else:
            # More conflicts - reject
            accept_prob = 0.0
        
        if accept_prob > 0 and random.random() < accept_prob:
            current_rows = new_rows
            current_scope = new_scope
            current_used = new_used
            current_conflicts = new_conflicts
            if current_conflicts == 0 and current_scope < best_scope:
                best_rows = [row[:] for row in current_rows]
                best_scope = current_scope
        
        t *= cooling
    
    # Final polish: try to reduce scope of best solution
    if time.time() < deadline - 10.0:
        best_rows = _polish_scope(best_rows, n_rows, k, deadline)
    
    return best_rows


def _greedy_init(n_rows, k):
    """Initialize with greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows


def _compute_all_diffs(rows):
    """Compute all positive differences across all rows."""
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    return used


def _is_strictly_increasing(row):
    """Check if row is strictly increasing starting from 0."""
    if not row or row[0] != 0:
        return False
    for i in range(1, len(row)):
        if row[i] <= row[i-1]:
            return False
    return True


def _greedy_row(k, used):
    """Build a single row greedily avoiding used differences."""
    row, x = [0], 0
    while len(row) < k + 1:
        x += 1
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used = used | new_diffs
    return row


def _random_row(k, scope, used):
    """Try to build a valid random row."""
    for _ in range(100):
        # Generate random increasing sequence
        row = [0]
        for i in range(k):
            max_val = scope
            if i < k - 1:
                max_val = scope - (k - 1 - i)
            if max_val <= row[-1]:
                break
            val = random.randint(row[-1] + 1, max_val)
            row.append(val)
        if len(row) == k + 1:
            # Check differences
            row_diffs = set()
            valid = True
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used or d in row_diffs:
                        valid = False
                        break
                    row_diffs.add(d)
                if not valid:
                    break
            if valid:
                return row
    return None


def _polish_scope(rows, n_rows, k, deadline):
    """Try to reduce the maximum scope."""
    best = [row[:] for row in rows]
    best_scope = max(max(row) for row in best)
    
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Try to reduce the largest entry
        max_row_idx = max(range(n_rows), key=lambda i: best[i][-1])
        row = best[max_row_idx]
        old_last = row[-1]
        
        # Try smaller values for the last entry
        found = False
        for new_last in range(old_last - 1, row[-2], -1):
            trial_row = row[:-1] + [new_last]
            if not _is_strictly_increasing(trial_row):
                continue
            
            # Check if this creates conflicts
            other_rows = [best[i] for i in range(n_rows) if i != max_row_idx]
            other_diffs = _compute_all_diffs(other_rows)
            
            row_diffs = set()
            valid = True
            for i in range(len(trial_row)):
                for j in range(i + 1, len(trial_row)):
                    d = trial_row[j] - trial_row[i]
                    if d in other_diffs or d in row_diffs:
                        valid = False
                        break
                    row_diffs.add(d)
                if not valid:
                    break
            
            if valid:
                best[max_row_idx] = trial_row
                best_scope = new_last
                found = True
                break
        
        if not found:
            # Try changing other entries
            r = random.randint(0, n_rows - 1)
            j = random.randint(1, k)
            old_val = best[r][j]
            other_rows = [best[i] for i in range(n_rows) if i != r]
            other_diffs = _compute_all_diffs(other_rows)
            
            # Try to find a better value
            for delta in range(-5, 6):
                nv = old_val + delta
                if nv <= 0:
                    continue
                trial_row = best[r][:]
                trial_row[j] = nv
                if not _is_strictly_increasing(trial_row):
                    continue
                
                row_diffs = set()
                valid = True
                for i in range(len(trial_row)):
                    for j2 in range(i + 1, len(trial_row)):
                        d = trial_row[j2] - trial_row[i]
                        if d in other_diffs or d in row_diffs:
                            valid = False
                            break
                        row_diffs.add(d)
                    if not valid:
                        break
                
                if valid:
                    new_scope = max(max(row) for row in best if best.index(row) != r)
                    if trial_row[-1] > new_scope:
                        new_scope = trial_row[-1]
                    if new_scope < best_scope:
                        best[r] = trial_row
                        best_scope = new_scope
                        break
    
    return best
# EVOLVE-BLOCK-END
