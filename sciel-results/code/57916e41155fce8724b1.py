"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    random.seed(42)
    
    num_entries = k + 1  # 7 entries per row
    diffs_per_row = num_entries * (num_entries - 1) // 2  # 21
    
    def get_diffs(row):
        """Get all pairwise differences for a row as a set."""
        return set(row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row)))
    
    def row_max(row):
        return row[-1]
    
    # Phase 1: Greedy initialization
    def greedy_construct():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < num_entries:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    rows = greedy_construct()
    best_rows = [list(r) for r in rows]
    best_max = max(max(r) for r in best_rows)
    
    # Phase 2: Local search to improve
    def try_improve(rows, used_diffs, current_max):
        """Try to improve by rebuilding one row at a time."""
        nonlocal best_rows, best_max, deadline
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Pick a random row to rebuild
            idx = random.randint(0, n_rows - 1)
            
            # Remove its differences from used set
            row_diffs = get_diffs(rows[idx])
            temp_used = used_diffs - row_diffs
            
            # Try to find a better row for position idx
            # Greedily build a row with small max
            best_new_row = None
            best_new_max = rows[idx][-1] + 1  # must be at least current max
            
            for max_val in range(num_entries - 1, best_new_max):
                if time.time() > deadline:
                    break
                
                # Try to build a row ending at max_val
                # Use greedy: pick smallest available values
                row = [0]
                for pos in range(1, num_entries - 1):
                    found = False
                    for x in range(1, max_val):
                        if x in row:
                            continue
                        candidate_row = row + [x]
                        # Check if we can complete to max_val
                        # Just check differences so far don't conflict
                        partial_diffs = set(candidate_row[j] - candidate_row[i] 
                                           for i in range(len(candidate_row)) 
                                           for j in range(i+1, len(candidate_row)))
                        if partial_diffs & temp_used:
                            continue
                        # Check if we can still add max_val
                        full_diffs = partial_diffs | {max_val - a for a in candidate_row}
                        if full_diffs & temp_used:
                            continue
                        if len(full_diffs) != len(partial_diffs) + (num_entries - 1):
                            # Some differences collide within the row
                            continue
                        row.append(x)
                        found = True
                        break
                    if not found:
                        break
                
                if len(row) == num_entries - 1:
                    row.append(max_val)
                    final_diffs = get_diffs(row)
                    if not (final_diffs & temp_used):
                        if max_val < best_new_max:
                            best_new_row = row
                            best_new_max = max_val
                        break
            
            if best_new_row is not None:
                old_row = rows[idx]
                old_diffs = get_diffs(old_row)
                new_diffs = get_diffs(best_new_row)
                
                new_used = temp_used | new_diffs
                new_max = max(row_max(r) for r in rows if r is not old_row)
                new_max = max(new_max, best_new_max)
                
                if new_max < best_max:
                    rows[idx] = best_new_row
                    used_diffs = new_used
                    best_max = new_max
                    best_rows = [list(r) for r in rows]
    
    # Compute initial used_diffs
    used_diffs = set()
    for row in rows:
        used_diffs |= get_diffs(row)
    
    try_improve(rows, used_diffs, best_max)
    
    return best_rows

# EVOLVE-BLOCK-END
