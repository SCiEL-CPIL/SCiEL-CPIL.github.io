"""Optimized difference triangle set construction using SA + local search."""
import time
import random
import sys

def _compute_all_diffs(rows):
    """Compute the set of all positive differences across all rows."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d > 0:
                    diffs.add(d)
    return diffs

def _count_collisions(rows):
    """Count how many differences appear more than once (total excess count)."""
    from collections import Counter
    cnt = Counter()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d > 0:
                    cnt[d] += 1
    # Collision count = sum over d of (cnt[d] - 1) for cnt[d] > 1
    collisions = sum(max(0, c - 1) for c in cnt.values())
    return collisions

def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(row) for row in rows)

def _is_valid(rows):
    """Check if all positive differences are distinct across all rows."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d > 0:
                    if d in diffs:
                        return False
                    diffs.add(d)
    return True

def _greedy_init(n_rows, k):
    """Fast greedy construction to get an initial valid set."""
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

def _sa_optimize(n_rows, k, time_budget):
    """Simulated annealing to minimize scope while maintaining validity."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Start from greedy
    best_rows = _greedy_init(n_rows, k)
    best_scope = _scope(best_rows)
    current_rows = [row[:] for row in best_rows]
    current_scope = best_scope
    
    # Total number of difference positions per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # SA parameters
    T = 50.0
    T_min = 0.1
    cooling = 0.9995
    
    # Precompute row lengths
    row_len = k + 1
    
    while time.time() < deadline:
        if T <= T_min:
            break
        
        # Pick a random row and position to perturb
        r = random.randrange(n_rows)
        pos = random.randrange(1, row_len)  # Don't change position 0 (always 0)
        
        # Choose a random delta
        max_val = current_scope + 50
        delta = random.randint(-max_val // 4, max_val // 4)
        old_val = current_rows[r][pos]
        new_val = old_val + delta
        
        # Ensure increasing order
        if new_val <= 0:
            new_val = 1
        if pos > 1 and new_val <= current_rows[r][pos - 1]:
            new_val = current_rows[r][pos - 1] + 1
        if pos < row_len - 1 and new_val >= current_rows[r][pos + 1]:
            new_val = current_rows[r][pos + 1] - 1
            if new_val <= current_rows[r][pos - 1]:
                continue
        
        # Apply change
        current_rows[r][pos] = new_val
        
        # Compute new collisions and scope
        new_collisions = _count_collisions(current_rows)
        new_scope = _scope(current_rows)
        
        old_collisions = _count_collisions(current_rows)  # This is after change, need old
        # We need to track old state properly
        # Let's restructure: compute cost before and after
        
        # Actually, let's use a simpler approach: track the difference set
        # For efficiency, let's just compute validity directly for small cases
        # But for larger cases, we need to be smarter
        
        # Let's use a different approach: track the multiset of differences
        # For now, just check if valid
        if _is_valid(current_rows):
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [row[:] for row in current_rows]
        else:
            # Reject if invalid (we want to maintain validity)
            # Actually, for SA we should allow some invalid states
            # Let's use a penalty-based approach
            pass
        
        # This is getting complex. Let's use a simpler local search approach.
        # Revert and try local search instead
        current_rows[r][pos] = old_val
    
    return best_rows, best_scope

def _local_search(rows, time_budget):
    """Local search to reduce scope while maintaining validity."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    best_rows = [row[:] for row in rows]
    best_scope = _scope(best_rows)
    current_rows = [row[:] for row in best_rows]
    current_scope = best_scope
    
    row_len = len(rows[0])
    n_rows = len(rows)
    
    # Precompute all differences for quick update
    def get_diffs(row):
        diffs = []
        for i in range(row_len):
            for j in range(i + 1, row_len):
                diffs.append(row[j] - row[i])
        return diffs
    
    # Use a set-based approach for efficiency
    from collections import Counter
    
    while time.time() < deadline:
        # Try to reduce the largest element
        # Find the row with the maximum element
        max_idx = max(range(n_rows), key=lambda r: current_rows[r][-1])
        max_row = current_rows[max_idx]
        
        # Try to reduce the last element
        # Or try to reduce any element
        improved = False
        
        for r in range(n_rows):
            for pos in range(1, row_len):
                old_val = current_rows[r][pos]
                # Try reducing
                for new_val in range(old_val - 1, 0, -1):
                    if pos > 1 and new_val <= current_rows[r][pos - 1]:
                        break
                    if pos < row_len - 1 and new_val >= current_rows[r][pos + 1]:
                        continue
                    
                    current_rows[r][pos] = new_val
                    
                    if _is_valid(current_rows):
                        new_scope = _scope(current_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [row[:] for row in current_rows]
                            current_scope = new_scope
                            improved = True
                            break
                    current_rows[r][pos] = old_val
                
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try random perturbations
            r = random.randrange(n_rows)
            pos = random.randrange(1, row_len)
            old_val = current_rows[r][pos]
            for new_val in random.sample(range(1, old_val + 20), min(20, old_val + 19)):
                if pos > 1 and new_val <= current_rows[r][pos - 1]:
                    continue
                if pos < row_len - 1 and new_val >= current_rows[r][pos + 1]:
                    continue
                
                current_rows[r][pos] = new_val
                if _is_valid(current_rows):
                    new_scope = _scope(current_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [row[:] for row in current_rows]
                        current_scope = new_scope
                        improved = True
                        break
                current_rows[r][pos] = old_val
            
            if not improved and random.random() < 0.01:
                # Occasionally restart from greedy
                current_rows = _greedy_init(n_rows, len(rows[0]) - 1)
                current_scope = _scope(current_rows)
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [row[:] for row in current_rows]
    
    return best_rows

def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 increasing integers."""
    start_time = time.time()
    total_time = 150.0
    safety_margin = 10.0
    deadline = start_time + total_time - safety_margin
    
    # Phase 1: Greedy initialization
    rows = _greedy_init(n_rows, k)
    scope = _scope(rows)
    
    # Phase 2: Local search
    remaining = deadline - time.time()
    if remaining > 5:
        rows = _local_search(rows, remaining * 0.7)
        scope = _scope(rows)
    
    # Phase 3: Final check and return
    if not _is_valid(rows):
        rows = _greedy_init(n_rows, k)
    
    return rows
