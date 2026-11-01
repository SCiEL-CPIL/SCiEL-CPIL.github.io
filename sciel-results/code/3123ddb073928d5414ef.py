import time
import random
import sys

# EVOLVE-BLOCK-START

def _row_diffs_set(row):
    """Return the set of all pairwise differences in a row."""
    n = len(row)
    s = set()
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _is_sidon(row):
    """Check if a row is a Sidon set (all pairwise differences distinct)."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    if len(rows) != n_rows:
        return False
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        if not _is_sidon(row):
            return False
        for d in _row_diffs_set(row):
            if d in used:
                return False
            used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

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
            if not (new & used):
                # Check Sidon property for the partial row
                partial_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        partial_diffs.add(row[j] - row[i])
                if not (new & partial_diffs):
                    row.append(x)
                    used |= new
        rows.append(row)
    return rows

def _local_search_optimized(rows, n_rows, k, deadline):
    """
    Optimized local search with incremental difference updates.
    """
    start_time = time.time()
    random.seed(42)
    
    # Deep copy initial rows
    cur = [r[:] for r in rows]
    best = [r[:] for r in cur]
    
    # Pre-compute difference sets for each row and global set
    cur_row_diffs = [_row_diffs_set(r) for r in cur]
    cur_used = set()
    for d_set in cur_row_diffs:
        cur_used |= d_set
    
    cur_val = _max_entry(cur)
    best_val = cur_val
    
    T = 50.0
    T_min = 0.1
    cool = 0.9999
    
    no_improve = 0
    max_no_improve = 50000
    
    # Helper to get diffs involving index i in row r
    def get_diffs_involving(row, i):
        diffs = set()
        for j in range(len(row)):
            if i != j:
                diffs.add(abs(row[i] - row[j]))
        return diffs

    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random row and position
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = cur[ri][pi]
        
        # Generate candidates: try to reduce the value, especially if it's large
        # Or try small perturbations
        candidates = []
        # Try values in a window around old_val
        # Bias towards smaller values if old_val is large
        if old_val > 10:
            candidates.extend(range(max(1, old_val - 10), old_val))
        candidates.extend(range(old_val + 1, old_val + 11))
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            # Check bounds
            if new_val <= cur[ri][pi - 1]:
                continue
            if pi < k and new_val >= cur[ri][pi + 1]:
                continue
            
            # Calculate old diffs involving pi
            old_diffs_i = get_diffs_involving(cur[ri], pi)
            
            # Calculate new diffs involving pi
            new_diffs_i = set()
            for j in range(k + 1):
                if j != pi:
                    new_diffs_i.add(abs(new_val - cur[ri][j]))
            
            # Check Sidon property for the new row
            # The diffs not involving pi remain the same.
            # We need to check if new_diffs_i intersects with (cur_row_diffs[ri] - old_diffs_i)
            other_diffs = cur_row_diffs[ri] - old_diffs_i
            if new_diffs_i & other_diffs:
                continue
            
            # Check global uniqueness
            # The diffs not involving pi in this row remain in cur_used.
            # The diffs involving pi in this row are removed from cur_used.
            # We need to check if new_diffs_i intersects with (cur_used - old_diffs_i)
            other_used = cur_used - old_diffs_i
            if new_diffs_i & other_used:
                continue
            
            # If valid, apply the move
            # Update cur_row_diffs[ri]
            cur_row_diffs[ri] = (cur_row_diffs[ri] - old_diffs_i) | new_diffs_i
            # Update cur_used
            cur_used = (cur_used - old_diffs_i) | new_diffs_i
            # Update row
            cur[ri][pi] = new_val
            
            # Calculate new max
            new_max = max(max(r) for r in cur)
            delta_e = new_max - cur_val
            
            accept = False
            if delta_e < 0:
                accept = True
            elif delta_e == 0:
                if random.random() < 0.5:
                    accept = True
            else:
                if T > T_min:
                    prob = 1.0 / (1.0 + delta_e / max(T, 0.01))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                cur_val = new_max
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
                    no_improve = 0
                else:
                    no_improve += 1
                moved = True
                break
            else:
                # Reject, revert changes
                cur_row_diffs[ri] = (cur_row_diffs[ri] - new_diffs_i) | old_diffs_i
                cur_used = (cur_used - new_diffs_i) | old_diffs_i
                cur[ri][pi] = old_val
        
        if not moved:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        if no_improve > 20000:
            # Restart from best
            cur = [r[:] for r in best]
            cur_val = best_val
            cur_row_diffs = [_row_diffs_set(r) for r in cur]
            cur_used = set()
            for d_set in cur_row_diffs:
                cur_used |= d_set
            T = 50.0
            no_improve = 0

    if not _valid(best, n_rows, k):
        return _build_greedy(n_rows, k)
    return best

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with greedy
    rows = _build_greedy(n_rows, k)
    
    # Run local search
    result = _local_search_optimized(rows, n_rows, k, deadline)
    
    # Verify validity
    if not _valid(result, n_rows, k):
        result = _build_greedy(n_rows, k)
    
    return result

# EVOLVE-BLOCK-END
