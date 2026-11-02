"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    
    def row_diffs(r):
        s = set()
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                s.add(r[j] - r[i])
        return s
    
    def is_valid_row(row, used_other):
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
        if row_diffs(row) & used_other:
            return False
        return True
    
    def compute_used_except(row_idx, rows):
        u = set()
        for i in range(n_rows):
            if i == row_idx:
                continue
            r = rows[i]
            for a in range(len(r)):
                for b in range(a + 1, len(r)):
                    u.add(r[b] - r[a])
        return u
    
    def scope_of(rows):
        return max(max(r) for r in rows)
    
    def greedy_row_safe(used, row_len, max_val=None, skip_prob=0.0):
        row = [0]
        x = 0
        limit = max_val if max_val is not None else 300
        new_diffs = set()
        while len(row) < row_len:
            x += 1
            if x > limit:
                return None, None
            if skip_prob > 0 and random.random() < skip_prob:
                continue
            new = {x - a for a in row}
            if not new & used and not new & new_diffs:
                row.append(x)
                new_diffs |= new
        if len(row) < row_len:
            return None, None
        return row, new_diffs
    
    # Phase 1: Greedy initialization
    used = set()
    rows = []
    for i in range(n_rows):
        row, diffs = greedy_row_safe(used, k + 1)
        if row is None:
            row, diffs = greedy_row_safe(used, k + 1, max_val=500)
        if row is None:
            row = list(range(k + 1))
            diffs = row_diffs(row)
        rows.append(row)
        used |= diffs
    
    best_rows = [r[:] for r in rows]
    best_scope = scope_of(best_rows)
    
    # Phase 2: Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    
    random.seed(42)
    
    while time.time() < deadline:
        r = random.random()
        if r < 0.55:
            # Replace a row, biased toward rows with the max entry
            cur_scope = scope_of(rows)
            max_rows = [i for i in range(n_rows) if rows[i][-1] == cur_scope]
            if max_rows and random.random() < 0.7:
                row_idx = random.choice(max_rows)
            else:
                row_idx = random.randint(0, n_rows - 1)
            
            used_other = compute_used_except(row_idx, rows)
            found = False
            for attempt in range(30):
                sp = 0.0 if attempt == 0 else 0.03
                limit = best_scope + 30
                new_row, _ = greedy_row_safe(used_other, k + 1, max_val=limit, skip_prob=sp)
                if new_row is None:
                    continue
                if is_valid_row(new_row, used_other):
                    new_scope = max(scope_of(rows[:row_idx] + [new_row] + rows[row_idx+1:]), new_row[-1])
                    delta = new_scope - cur_scope
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        rows[row_idx] = new_row
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [rr[:] for rr in rows]
                        found = True
                        break
            # If not found, try a smaller limit to force compression
            if not found and random.random() < 0.5:
                for limit in range(cur_scope - 1, 0, -1):
                    new_row, _ = greedy_row_safe(used_other, k + 1, max_val=limit)
                    if new_row is not None and is_valid_row(new_row, used_other):
                        new_scope = max(scope_of(rows[:row_idx] + [new_row] + rows[row_idx+1:]), new_row[-1])
                        delta = new_scope - cur_scope
                        if delta < 0 or random.random() < math.exp(-delta / T):
                            rows[row_idx] = new_row
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best_rows = [rr[:] for rr in rows]
                        break
        elif r < 0.8:
            # Modify a single element
            row_idx = random.randint(0, n_rows - 1)
            rrow = rows[row_idx]
            if len(rrow) < 2:
                continue
            pos = random.randint(1, len(rrow) - 1)
            old_val = rrow[pos]
            used_other = compute_used_except(row_idx, rows)
            candidates = []
            for d in range(-8, 9):
                if d == 0:
                    continue
                nv = old_val + d
                if nv <= 0:
                    continue
                if pos > 0 and nv <= rrow[pos - 1]:
                    continue
                if pos < len(rrow) - 1 and nv >= rrow[pos + 1]:
                    continue
                candidates.append(nv)
            random.shuffle(candidates)
            for nv in candidates:
                new_r = rrow[:]
                new_r[pos] = nv
                if is_valid_row(new_r, used_other):
                    new_scope = max(scope_of(rows[:row_idx] + [new_r] + rows[row_idx+1:]), new_r[-1])
                    cur_scope = scope_of(rows)
                    delta = new_scope - cur_scope
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        rows[row_idx] = new_r
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [rr[:] for rr in rows]
                        break
        else:
            # Compress last element of a max row
            cur_scope = scope_of(rows)
            max_rows = [i for i in range(n_rows) if rows[i][-1] == cur_scope]
            if not max_rows:
                continue
            row_idx = random.choice(max_rows)
            rrow = rows[row_idx]
            prefix = rrow[:-1]
            used_other = compute_used_except(row_idx, rows)
            for new_last in range(rrow[-1] - 1, prefix[-1], -1):
                test_r = prefix + [new_last]
                if is_valid_row(test_r, used_other):
                    new_scope = max(scope_of(rows[:row_idx] + [test_r] + rows[row_idx+1:]), new_last)
                    delta = new_scope - cur_scope
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        rows[row_idx] = test_r
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [rr[:] for rr in rows]
                    break
        
        T = max(T_min, T * cooling)
    
    return best_rows
# EVOLVE-BLOCK-END
