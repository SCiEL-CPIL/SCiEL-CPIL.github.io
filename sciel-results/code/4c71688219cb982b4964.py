"""Difference triangle set (7,5) – optimized greedy + local search + simulated annealing."""
import time
import random
import math

_START = time.time()
_DEADLINE = _START + 142.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START


def _dead():
    return time.time() >= _DEADLINE


def _diffs_of_row(row):
    """All positive differences a_j - a_i for i<j in one row."""
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def _scope(rows):
    return max(max(r) for r in rows)


def _valid(rows, n_rows, k):
    if len(rows) != n_rows:
        return False
    used = set()
    for r in rows:
        if len(r) != k + 1 or r[0] != 0:
            return False
        for i in range(k + 1):
            for j in range(i + 1, k + 1):
                if r[j] <= r[i]:
                    return False
        for d in _diffs_of_row(r):
            if d in used:
                return False
            used.add(d)
    return True


def _build_diff_array(rows, max_val):
    """Build a difference count array for all rows."""
    size = max_val + 2
    diff_counts = [0] * (size + 1)
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d <= size:
                    diff_counts[d] += 1
    return diff_counts


def _interleaved_greedy(n_rows, k, seed=0):
    """Interleaved greedy: build all rows simultaneously, one element at a time."""
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
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    break
    
    return rows


def _greedy_row(used, k, max_mark, rnd):
    """Greedy place one row of length k+1 starting at 0."""
    row = [0]
    local = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        if x > max_mark:
            return None, local, False
        new = []
        bad = False
        for a in row:
            dv = x - a
            if dv in used or dv in local:
                bad = True
                break
            new.append(dv)
        if not bad:
            row.append(x)
            local.update(new)
    return row, local, True


def _local_search_reduce(rows, n_rows, k, time_limit=30.0):
    """
    Local search that tries to reduce each element to its minimum valid value.
    Uses efficient difference counting.
    """
    start_time = time.time()
    current_rows = [list(r) for r in rows]
    
    max_val = max(max(r) for r in current_rows)
    diff_size = max_val + 50
    diff_counts = [0] * (diff_size + 1)
    
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    def get_scope():
        return max(max(r) for r in current_rows)
    
    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current_rows[r]
            for pos in range(len(row) - 1, 0, -1):
                if (time.time() - start_time) > time_limit:
                    break
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                
                if lower_bound >= old_val:
                    continue
                
                # Precompute old differences
                old_diffs = [abs(old_val - row[i]) for i in range(len(row)) if i != pos]
                
                # Remove old diffs from counts
                for d in old_diffs:
                    if d < len(diff_counts):
                        diff_counts[d] -= 1
                
                # Try values from lower_bound upward until we find a valid one
                found = False
                for new_val in range(lower_bound, old_val):
                    valid = True
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        d = abs(new_val - row[i])
                        if d < len(diff_counts) and diff_counts[d] > 0:
                            valid = False
                            break
                    if valid:
                        row[pos] = new_val
                        new_diffs = [abs(new_val - row[i]) for i in range(len(row)) if i != pos]
                        for d in new_diffs:
                            if d < len(diff_counts):
                                diff_counts[d] += 1
                        improved = True
                        found = True
                        break
                
                if not found:
                    for d in old_diffs:
                        if d < len(diff_counts):
                            diff_counts[d] += 1
    
    current_scope = get_scope()
    if current_scope < best_scope:
        best_scope = current_scope
        best_rows = [list(r) for r in current_rows]
        
    return best_rows


def _sa_with_local_moves(rows, n_rows, k, time_budget, seed=42):
    """
    Simulated annealing with both row replacement and element-level moves.
    """
    rnd = random.Random(seed)
    best = [r[:] for r in rows]
    best_scope = _scope(best)
    current = [r[:] for r in best]
    current_scope = best_scope

    # Build diff count array for fast lookup
    max_val = max(max(r) for r in current) + 10
    diff_counts = [0] * (max_val + 1)
    for row in current:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d < len(diff_counts):
                    diff_counts[d] += 1

    def get_used_set():
        used = set()
        for d, c in enumerate(diff_counts):
            if c > 0:
                used.add(d)
        return used

    def remove_row_diffs(row):
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d < len(diff_counts):
                    diff_counts[d] -= 1

    def add_row_diffs(row):
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1

    def can_place_element(row, pos, val):
        """Check if placing val at position pos in row is valid (no diff conflicts)."""
        for i in range(len(row)):
            if i == pos:
                continue
            d = abs(val - row[i])
            if d < len(diff_counts) and diff_counts[d] > 0:
                return False
        return True

    def remove_pos_diffs(row, pos):
        val = row[pos]
        for i in range(len(row)):
            if i == pos:
                continue
            d = abs(val - row[i])
            if d < len(diff_counts):
                diff_counts[d] -= 1

    def add_pos_diffs(row, pos):
        val = row[pos]
        for i in range(len(row)):
            if i == pos:
                continue
            d = abs(val - row[i])
            if d >= len(diff_counts):
                diff_counts.extend([0] * (d - len(diff_counts) + 1))
            diff_counts[d] += 1

    deadline = time.time() + time_budget
    T = 15.0
    cooling = 0.997

    while time.time() < deadline and T > 0.05:
        if _dead():
            break

        move_type = rnd.random()
        if move_type < 0.4:
            # Move type 1: Replace one row
            ri = rnd.randint(0, n_rows - 1)
            old_row = current[ri]
            remove_row_diffs(old_row)

            max_mark = min(current_scope + 3, best_scope + 3)
            new_row = None
            for attempt in range(30):
                row = [0]
                local = set()
                x = 0
                success = True
                while len(row) < k + 1:
                    x += 1
                    if x > max_mark:
                        success = False
                        break
                    if rnd.random() < 0.03 and x < max_mark:
                        x += 1
                        if x > max_mark:
                            success = False
                            break
                    bad = False
                    for a in row:
                        dv = x - a
                        if dv in local:
                            bad = True
                            break
                        if dv < len(diff_counts) and diff_counts[dv] > 0:
                            bad = True
                            break
                    if not bad:
                        row.append(x)
                        local.update({x - a for a in row[:-1]})
                if success and len(row) == k + 1:
                    new_row = row
                    break

            if new_row is None:
                add_row_diffs(old_row)
                T *= cooling
                continue

            # Check if new_row is valid with current diff_counts (old_row already removed)
            valid = True
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    d = new_row[j] - new_row[i]
                    if d < len(diff_counts) and diff_counts[d] > 0:
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                add_row_diffs(old_row)
                T *= cooling
                continue

            add_row_diffs(new_row)
            other_max = max(max(r) for i, r in enumerate(current) if i != ri)
            new_scope = max(other_max, max(new_row))
            delta = new_scope - current_scope
            if delta <= 0 or rnd.random() < math.exp(-delta / T):
                current[ri] = new_row
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best = [r[:] for r in current]
            else:
                remove_row_diffs(new_row)
                add_row_diffs(old_row)

        elif move_type < 0.75:
            # Move type 2: Shift a single element down
            ri = rnd.randint(0, n_rows - 1)
            row = current[ri]
            pi = rnd.randint(1, len(row) - 1)
            old_val = row[pi]
            
            shift = rnd.choice([1, 1, 2])
            new_val = old_val - shift
            if new_val <= row[pi - 1]:
                T *= cooling
                continue
            if pi < len(row) - 1 and new_val >= row[pi + 1]:
                T *= cooling
                continue

            remove_pos_diffs(row, pi)
            if can_place_element(row, pi, new_val):
                row[pi] = new_val
                add_pos_diffs(row, pi)
                
                other_max = max(max(r) for i, r in enumerate(current) if i != ri)
                new_scope = max(other_max, max(row))
                delta = new_scope - current_scope
                if delta <= 0 or rnd.random() < math.exp(-delta / T):
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best = [r[:] for r in current]
                else:
                    remove_pos_diffs(row, pi)
                    row[pi] = old_val
                    add_pos_diffs(row, pi)
            else:
                row[pi] = old_val
                add_pos_diffs(row, pi)

        else:
            # Move type 3: Swap two elements in the same row (if they don't violate order)
            ri = rnd.randint(0, n_rows - 1)
            row = current[ri]
            if len(row) < 3:
                T *= cooling
                continue
            pi = rnd.randint(1, len(row) - 2)
            pj = pi + 1
            
            remove_pos_diffs(row, pi)
            remove_pos_diffs(row, pj)
            
            row[pi], row[pj] = row[pj], row[pi]
            
            # Check validity
            valid = True
            for i in range(len(row)):
                if i > 0 and row[i] <= row[i-1]:
                    valid = False
                    break
            if valid:
                # Check diff conflicts
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        d = row[j] - row[i]
                        if d < len(diff_counts) and diff_counts[d] > 0:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                add_pos_diffs(row, pi)
                add_pos_diffs(row, pj)
                
                other_max = max(max(r) for i, r in enumerate(current) if i != ri)
                new_scope = max(other_max, max(row))
                delta = new_scope - current_scope
                if delta <= 0 or rnd.random() < math.exp(-delta / T):
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best = [r[:] for r in current]
                else:
                    remove_pos_diffs(row, pi)
                    remove_pos_diffs(row, pj)
                    row[pi], row[pj] = row[pj], row[pi]
                    add_pos_diffs(row, pi)
                    add_pos_diffs(row, pj)
            else:
                row[pi], row[pj] = row[pj], row[pi]
                add_pos_diffs(row, pi)
                add_pos_diffs(row, pj)

        T *= cooling

    return best, best_scope


def _final_polish(rows, n_rows, k, time_limit=10.0):
    """
    Final high-precision polish: repeatedly try to reduce each element to its minimum valid value.
    """
    start_time = time.time()
    current_rows = [list(r) for r in rows]
    
    max_val = max(max(r) for r in current_rows)
    diff_counts = [0] * (max_val + 30)
    
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current_rows[r]
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                if lower_bound >= old_val:
                    continue
                
                old_diffs = [abs(old_val - row[i]) for i in range(len(row)) if i != pos]
                
                for d in old_diffs:
                    if d < len(diff_counts):
                        diff_counts[d] -= 1
                
                found = False
                for new_val in range(lower_bound, old_val):
                    valid = True
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        d = abs(new_val - row[i])
                        if d < len(diff_counts) and diff_counts[d] > 0:
                            valid = False
                            break
                    if valid:
                        row[pos] = new_val
                        new_diffs = [abs(new_val - row[i]) for i in range(len(row)) if i != pos]
                        for d in new_diffs:
                            if d < len(diff_counts):
                                diff_counts[d] += 1
                        improved = True
                        found = True
                        break
                
                if not found:
                    for d in old_diffs:
                        if d < len(diff_counts):
                            diff_counts[d] += 1
    
    return current_rows


def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 increasing integers."""
    start_time = time.time()
    deadline = start_time + 142.0
    
    best_rows = None
    best_scope = float('inf')
    
    # 1. Generate diverse initial solutions using interleaved greedy
    max_greedy_seeds = 12
    
    for seed in range(max_greedy_seeds):
        if time.time() > deadline - 110.0:
            break
            
        rows = _interleaved_greedy(n_rows, k, seed=seed)
        sc = max(max(r) for r in rows)
        
        if sc < best_scope:
            best_scope = sc
            best_rows = [list(r) for r in rows]
            
    if best_rows is None:
        # Fallback: simple sequential greedy
        rows = []
        used = set()
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

    # 2. Optimization Phase
    time_remaining = deadline - time.time()
    
    # Phase 2a: Local search to reduce scope
    if time_remaining > 25.0:
        t_ls = min(35.0, time_remaining * 0.3)
        optimized = _local_search_reduce(best_rows, n_rows, k, time_limit=t_ls)
        sc = max(max(r) for r in optimized)
        if sc < best_scope:
            best_scope = sc
            best_rows = optimized
    
    # Phase 2b: Simulated annealing with local moves
    time_remaining = deadline - time.time()
    if time_remaining > 20.0:
        t_sa = min(60.0, time_remaining * 0.6)
        sa_result, sa_scope = _sa_with_local_moves(best_rows, n_rows, k, time_budget=t_sa, seed=43)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_result
    
    # Phase 2c: Another round of local search
    time_remaining = deadline - time.time()
    if time_remaining > 15.0:
        t_ls2 = min(20.0, time_remaining * 0.4)
        optimized2 = _local_search_reduce(best_rows, n_rows, k, time_limit=t_ls2)
        sc = max(max(r) for r in optimized2)
        if sc < best_scope:
            best_scope = sc
            best_rows = optimized2
    
    # Phase 2d: Final polish
    time_remaining = deadline - time.time()
    if time_remaining > 8.0:
        t_polish = min(12.0, time_remaining * 0.8)
        polished = _final_polish(best_rows, n_rows, k, time_limit=t_polish)
        sc = max(max(r) for r in polished)
        if sc < best_scope:
            best_scope = sc
            best_rows = polished
    
    return best_rows
