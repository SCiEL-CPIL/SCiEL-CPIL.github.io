"""Difference triangle set (7,5) - rewritten with constraint propagation and multi-start search."""
import time
import random
import sys

def _compute_diffs(row):
    """Compute all pairwise differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _validate(rows):
    """Check if all differences across all rows are distinct."""
    all_diffs = set()
    for row in rows:
        row_diffs = _compute_diffs(row)
        if len(row_diffs) != len(row):  # Quick check: should be k*(k+1)/2 = 15 for k=5
            # Actually for k+1 elements, there are k*(k+1)/2 differences
            pass
        if all_diffs & row_diffs:
            return False
        all_diffs |= row_diffs
    return True

def _scope(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)

def _greedy_row(k, used_diffs, start=0):
    """
    Greedily build a single row of k+1 increasing integers starting at 0,
    avoiding all differences in used_diffs.
    Returns the row or None if stuck.
    """
    row = [0]
    for _ in range(k):
        x = row[-1] + 1
        while True:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                row.append(x)
                break
            x += 1
            # Safety: if we go too far, give up
            if x > 500:
                return None
    return row

def _greedy_full(n_rows, k, seed=0):
    """
    Build all rows greedily, one at a time, with randomized row order.
    """
    random.seed(seed)
    rows = []
    used = set()
    order = list(range(n_rows))
    random.shuffle(order)
    
    for r in order:
        row = _greedy_row(k, used)
        if row is None:
            return None
        rows.append(row)
        used |= _compute_diffs(row)
    
    return rows

def _interleaved_greedy(n_rows, k, seed=0):
    """
    Interleaved greedy: build all rows simultaneously, one element at a time.
    """
    random.seed(seed)
    rows = [[0] for _ in range(n_rows)]
    used = set()
    
    for step in range(1, k + 1):
        order = list(range(n_rows))
        random.shuffle(order)
        for r in order:
            row = rows[r]
            x = row[-1] + 1
            while True:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                    break
            if x > 500:
                return None
    
    return rows

def _local_search_reduce(rows, n_rows, k, time_limit=30.0, seed=42):
    """
    Local search: try to reduce each element to its minimum valid value.
    Uses a difference count array for fast conflict checking.
    """
    random.seed(seed)
    start_time = time.time()
    current = [list(r) for r in rows]
    
    def get_scope():
        return max(max(r) for r in current)
    
    best = [list(r) for r in current]
    best_scope = get_scope()
    
    def build_diff_counts(rows):
        max_val = max(max(r) for r in rows)
        counts = {}
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts
    
    diff_counts = build_diff_counts(current)
    
    def try_reduce(r, pos, new_val):
        """Try to set current[r][pos] = new_val. Returns True if successful."""
        row = current[r]
        old_val = row[pos]
        if new_val >= old_val:
            return False
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return False
        
        # Collect old and new differences
        old_diffs = []
        new_diffs = []
        for i in range(len(row)):
            if i == pos:
                continue
            d_old = abs(old_val - row[i])
            d_new = abs(new_val - row[i])
            old_diffs.append(d_old)
            new_diffs.append(d_new)
        
        # Temporarily remove old diffs
        for d in old_diffs:
            diff_counts[d] = diff_counts.get(d, 0) - 1
            if diff_counts[d] <= 0:
                del diff_counts[d]
        
        # Check new diffs
        valid = True
        for d in new_diffs:
            if diff_counts.get(d, 0) > 0:
                valid = False
                break
        
        if not valid:
            # Restore
            for d in old_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            return False
        
        # Apply
        row[pos] = new_val
        for d in new_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        
        return True
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try reducing from the rightmost elements first
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current[r]
            for pos in range(len(row) - 1, 0, -1):
                if (time.time() - start_time) > time_limit:
                    break
                old_val = row[pos]
                lower = row[pos - 1] + 1
                # Try from lower up to old_val - 1
                for new_val in range(lower, old_val):
                    if try_reduce(r, pos, new_val):
                        improved = True
                        break
    
    sc = get_scope()
    if sc < best_scope:
        best_scope = sc
        best = [list(r) for r in current]
    
    return best

def _simulated_annealing(rows, n_rows, k, time_limit=30.0, seed=43):
    """
    Simulated annealing: allow worsening moves to escape local minima.
    """
    random.seed(seed)
    start_time = time.time()
    current = [list(r) for r in rows]
    
    def get_scope():
        return max(max(r) for r in current)
    
    best = [list(r) for r in current]
    best_scope = get_scope()
    cur_scope = best_scope
    
    def build_diff_counts(rows):
        counts = {}
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts
    
    diff_counts = build_diff_counts(current)
    
    T = 5.0
    T_min = 0.1
    alpha = 0.995
    
    while (time.time() - start_time) < time_limit and T > T_min:
        r = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        row = current[r]
        old_val = row[pos]
        
        # Generate candidate
        delta = random.randint(-8, 8)
        new_val = old_val + delta
        lower = row[pos - 1] + 1
        upper = row[pos + 1] - 1 if pos < len(row) - 1 else old_val + 15
        if new_val < lower:
            new_val = lower
        if new_val > upper:
            new_val = upper
        if new_val == old_val:
            continue
        
        old_diffs = []
        new_diffs = []
        for i in range(len(row)):
            if i == pos:
                continue
            old_diffs.append(abs(old_val - row[i]))
            new_diffs.append(abs(new_val - row[i]))
        
        # Remove old
        for d in old_diffs:
            diff_counts[d] = diff_counts.get(d, 0) - 1
            if diff_counts[d] <= 0:
                del diff_counts[d]
        
        valid = True
        for d in new_diffs:
            if diff_counts.get(d, 0) > 0:
                valid = False
                break
        
        if not valid:
            for d in old_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            T *= alpha
            continue
        
        # Apply
        row[pos] = new_val
        for d in new_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        
        new_scope = get_scope()
        if new_scope < cur_scope:
            cur_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best = [list(r) for r in current]
        else:
            de = new_scope - cur_scope
            if random.random() < pow(2.71828, -de / T):
                cur_scope = new_scope
            else:
                # Revert
                row[pos] = old_val
                for d in new_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) - 1
                    if diff_counts[d] <= 0:
                        del diff_counts[d]
                for d in old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        
        T *= alpha
    
    return best

def _final_polish(rows, n_rows, k, time_limit=5.0):
    """
    Deterministic final polish: repeatedly sweep all positions trying to reduce.
    """
    start_time = time.time()
    current = [list(r) for r in rows]
    
    def build_diff_counts(rows):
        counts = {}
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i + 1, n):
                    d = row[j] - row[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts
    
    diff_counts = build_diff_counts(current)
    
    def try_reduce(r, pos, new_val):
        row = current[r]
        old_val = row[pos]
        if new_val >= old_val:
            return False
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return False
        
        old_diffs = []
        new_diffs = []
        for i in range(len(row)):
            if i == pos:
                continue
            old_diffs.append(abs(old_val - row[i]))
            new_diffs.append(abs(new_val - row[i]))
        
        for d in old_diffs:
            diff_counts[d] = diff_counts.get(d, 0) - 1
            if diff_counts[d] <= 0:
                del diff_counts[d]
        
        valid = True
        for d in new_diffs:
            if diff_counts.get(d, 0) > 0:
                valid = False
                break
        
        if not valid:
            for d in old_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            return False
        
        row[pos] = new_val
        for d in new_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        return True
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current[r]
            for pos in range(len(row) - 1, 0, -1):
                if (time.time() - start_time) > time_limit:
                    break
                old_val = row[pos]
                lower = row[pos - 1] + 1
                for new_val in range(lower, old_val):
                    if try_reduce(r, pos, new_val):
                        improved = True
                        break
    
    return current

def construct(n_rows, k):
    """
    Construct difference triangle set with n_rows rows of k+1 increasing integers.
    Each row starts at 0 and increases strictly.
    All positive differences a_j - a_i (i < j) must be distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Generate diverse initial solutions
    # Try multiple strategies with different seeds
    initial_candidates = []
    
    # Strategy A: Sequential greedy with shuffled row order
    for seed in range(15):
        if time.time() > deadline - 80.0:
            break
        rows = _greedy_full(n_rows, k, seed=seed)
        if rows is not None:
            sc = _scope(rows)
            initial_candidates.append((sc, rows))
    
    # Strategy B: Interleaved greedy
    for seed in range(15):
        if time.time() > deadline - 80.0:
            break
        rows = _interleaved_greedy(n_rows, k, seed=seed)
        if rows is not None:
            sc = _scope(rows)
            initial_candidates.append((sc, rows))
    
    if not initial_candidates:
        # Fallback: simple sequential greedy
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Sort by scope and keep top candidates
    initial_candidates.sort(key=lambda x: x[0])
    top_candidates = initial_candidates[:min(5, len(initial_candidates))]
    
    best_scope = top_candidates[0][0]
    best_rows = [list(r) for r in top_candidates[0][1]]
    
    # Phase 2: Local search on each top candidate
    time_remaining = deadline - time.time()
    if time_remaining > 20.0:
        # Split time among top candidates
        n_opt = min(3, len(top_candidates))
        time_per_opt = (time_remaining - 20.0) / n_opt
        
        for i in range(n_opt):
            if time.time() > deadline - 20.0:
                break
            sc, rows = top_candidates[i]
            if sc > best_scope:
                break
            
            # Local search reduce
            t1 = min(time_per_opt * 0.4, 15.0)
            if t1 > 2.0:
                opt_rows = _local_search_reduce(rows, n_rows, k, time_limit=t1, seed=100 + i)
                opt_sc = _scope(opt_rows)
                if opt_sc < best_scope:
                    best_scope = opt_sc
                    best_rows = opt_rows
            
            # Simulated annealing
            t2 = min(time_per_opt * 0.4, 15.0)
            if t2 > 2.0 and (time.time() < deadline - 15.0):
                sa_rows = _simulated_annealing(best_rows, n_rows, k, time_limit=t2, seed=200 + i)
                sa_sc = _scope(sa_rows)
                if sa_sc < best_scope:
                    best_scope = sa_sc
                    best_rows = sa_rows
    
    # Phase 3: Final polish
    t_polish = deadline - time.time() - 5.0
    if t_polish > 1.0:
        polished = _final_polish(best_rows, n_rows, k, time_limit=t_polish)
        pol_sc = _scope(polished)
        if pol_sc < best_scope:
            best_scope = pol_sc
            best_rows = polished
    
    # Final validation
    if not _validate(best_rows):
        # If validation fails (shouldn't happen), fall back to a known-valid solution
        used = set()
        fallback = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            fallback.append(row)
        best_rows = fallback
    
    return best_rows
