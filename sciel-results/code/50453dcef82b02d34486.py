"""Difference triangle set (7,5) solver - combined: delta-based SA + aggressive polish + Hungarian matching."""
import time
import random
import math
from collections import Counter

_DEADLINE = 142.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _row_diffs(row):
    n = len(row)
    return [row[j] - row[i] for i in range(n) for j in range(i+1, n)]

def _row_diffs_set(row):
    n = len(row)
    return set(row[j] - row[i] for i in range(n) for j in range(i+1, n))

def _max_entry(rows):
    return max(max(r) for r in rows) if rows else 0

def _sum_entries(rows):
    return sum(sum(r) for r in rows)

def _is_valid(rows, k):
    if not rows:
        return False
    seen = set()
    for row in rows:
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(k+1):
            for j in range(i+1, k+1):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _build_row_given_used(used, k, rng, cap=None, jitter=0.0):
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if cap is not None and x > cap:
            return None
        if jitter > 0 and rng.random() < jitter:
            x += rng.randint(1, 3)
            if cap is not None and x > cap:
                return None
        new_diffs = {x - a for a in row}
        if not (new_diffs & used):
            row.append(x)
            used = used | new_diffs
    return row

def _initial_construction(n_rows, k, rng, diversity_base=0.04):
    best = None
    best_score = float('inf')
    for attempt in range(500):
        if _elapsed() > _DEADLINE - 25:
            break
        used = set()
        rows = []
        ok = True
        for r in range(n_rows):
            div = diversity_base * (0.5 + rng.random())
            row = _build_row_given_used(used, k, rng, jitter=div)
            if row is None:
                ok = False
                break
            rows.append(row)
            used = used | set(_row_diffs(row))
        if ok and len(rows) == n_rows:
            if _is_valid(rows, k):
                score = _max_entry(rows)
                if score < best_score:
                    best_score = score
                    best = [list(r) for r in rows]
    return best, best_score

def _hungarian_match(rows, k, rng, max_scope):
    """
    Try to reassign differences using a cost-based matching approach.
    For each row, we want to assign k+1 values (including 0) such that
    all pairwise differences are globally unique.
    
    This is a heuristic: we try to compress rows by trying to fit them
    into a smaller scope, using a cost function that penalizes large values.
    """
    n_rows = len(rows)
    n_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * n_diffs_per_row
    
    # Current used differences
    used_diffs = set()
    for row in rows:
        used_diffs |= _row_diffs_set(row)
    
    # Try to rebuild each row with a tighter cap
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(best_rows)
    
    if max_scope <= 0:
        max_scope = best_scope + 5
    
    for attempt in range(200):
        if _elapsed() > _DEADLINE - 10:
            break
        
        cap = max_scope - attempt // 50  # Gradually tighten
        if cap < best_scope:
            cap = best_scope
        
        used = set()
        new_rows = []
        ok = True
        
        # Shuffle row order for diversity
        order = list(range(n_rows))
        rng.shuffle(order)
        
        for ri in order:
            div = 0.03 * rng.random()
            row = _build_row_given_used(used, k, rng, cap=cap, jitter=div)
            if row is None:
                ok = False
                break
            new_rows.append(row)
            used |= _row_diffs_set(row)
        
        if ok and len(new_rows) == n_rows:
            if _is_valid(new_rows, k):
                sc = _max_entry(new_rows)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = [list(r) for r in new_rows]
    
    return best_rows, best_scope

def _sa_main(rows, k, rng, time_budget=100):
    """Simulated annealing with delta-based conflict tracking."""
    n_rows = len(rows)
    cur = [list(r) for r in rows]
    best = [list(r) for r in rows]
    
    # Initialize diff counts
    diff_counts = {}
    conflict_count = 0
    for row in cur:
        for d in _row_diffs(row):
            old_cnt = diff_counts.get(d, 0)
            if old_cnt == 1:
                conflict_count += 1
            elif old_cnt > 1:
                conflict_count += 1
            diff_counts[d] = old_cnt + 1
    
    cur_scope = _max_entry(cur)
    cur_sum = _sum_entries(cur)
    PENALTY = 500
    SUM_WEIGHT = 0.05
    cur_cost = cur_scope + PENALTY * conflict_count + SUM_WEIGHT * cur_sum
    
    best_scope = cur_scope
    best_conf = conflict_count
    best_sum = cur_sum
    best_cost = cur_cost
    best = [list(r) for r in cur]
    
    # Precompute row diffs
    row_diffs_list = [_row_diffs(row) for row in cur]
    
    t0 = time.time()
    temp = 15.0
    cool = 0.9995
    
    while time.time() - t0 < time_budget:
        if _elapsed() > _DEADLINE - 5:
            break
        
        move_type = rng.random()
        ri = rng.randrange(n_rows)
        
        if move_type < 0.4:
            # Rebuild row
            old_row = cur[ri]
            old_diffs = row_diffs_list[ri]
            
            # Remove old diffs
            for d in old_diffs:
                cnt = diff_counts[d]
                if cnt == 2:
                    conflict_count -= 1
                elif cnt > 2:
                    conflict_count -= 1
                if cnt == 1:
                    del diff_counts[d]
                else:
                    diff_counts[d] = cnt - 1
            
            used = set(diff_counts.keys())
            cap = cur_scope + rng.randint(0, 5)
            new_row = _build_row_given_used(used, k, rng, cap=cap, jitter=0.05)
            
            if new_row is None or len(new_row) < k + 1:
                # Restore
                for d in old_diffs:
                    cnt = diff_counts.get(d, 0)
                    if cnt == 1:
                        conflict_count += 1
                    elif cnt > 1:
                        conflict_count += 1
                    diff_counts[d] = cnt + 1
                continue
            
            new_diffs = _row_diffs(new_row)
            for d in new_diffs:
                cnt = diff_counts.get(d, 0)
                if cnt == 1:
                    conflict_count += 1
                elif cnt > 1:
                    conflict_count += 1
                diff_counts[d] = cnt + 1
            
            cur[ri] = new_row
            row_diffs_list[ri] = new_diffs
            cur_scope = _max_entry(cur)
            cur_sum = _sum_entries(cur)
            cur_cost = cur_scope + PENALTY * conflict_count + SUM_WEIGHT * cur_sum
            
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_scope = cur_scope
                best_conf = conflict_count
                best_sum = cur_sum
                best = [list(r) for r in cur]
        
        elif move_type < 0.85:
            # Shift one entry
            row = cur[ri]
            idx = rng.randint(1, len(row) - 1)
            old_val = row[idx]
            
            r = rng.random()
            if r < 0.55:
                new_val = old_val - rng.randint(1, 4)
            elif r < 0.8:
                new_val = old_val + rng.randint(-3, 3)
            else:
                new_val = old_val + rng.randint(-8, 8)
            
            if new_val <= 0 or new_val == old_val:
                temp = max(0.01, temp * cool)
                continue
            if idx > 1 and new_val <= row[idx-1]:
                temp = max(0.01, temp * cool)
                continue
            if idx < len(row) - 1 and new_val >= row[idx+1]:
                temp = max(0.01, temp * cool)
                continue
            
            old_diffs = row_diffs_list[ri]
            old_diff_set = set(old_diffs)
            
            row[idx] = new_val
            new_diffs = _row_diffs(row)
            new_diff_set = set(new_diffs)
            
            removed = old_diff_set - new_diff_set
            added = new_diff_set - old_diff_set
            
            delta_conf = 0
            for d in removed:
                cnt = diff_counts.get(d, 0)
                if cnt == 2:
                    delta_conf -= 1
                elif cnt > 2:
                    delta_conf -= 1
            for d in added:
                cnt = diff_counts.get(d, 0)
                if cnt == 1:
                    delta_conf += 1
                elif cnt > 1:
                    delta_conf += 1
            
            # Apply changes
            for d in removed:
                cnt = diff_counts[d]
                if cnt == 1:
                    del diff_counts[d]
                else:
                    diff_counts[d] = cnt - 1
            for d in added:
                cnt = diff_counts.get(d, 0)
                diff_counts[d] = cnt + 1
            
            conflict_count += delta_conf
            row_diffs_list[ri] = new_diffs
            cur_scope = _max_entry(cur)
            cur_sum = _sum_entries(cur)
            cur_cost = cur_scope + PENALTY * conflict_count + SUM_WEIGHT * cur_sum
            
            delta = cur_cost - (best_cost if cur is best else cur_cost)
            
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_scope = cur_scope
                best_conf = conflict_count
                best_sum = cur_sum
                best = [list(r) for r in cur]
            elif temp > 0.01:
                # Accept with probability
                # We need to compare to current cost before this move
                # For simplicity, always accept if it doesn't increase conflicts too much
                if conflict_count <= best_conf + 2:
                    pass  # Keep the move
                else:
                    # Revert
                    row[idx] = old_val
                    for d in added:
                        cnt = diff_counts[d]
                        if cnt == 1:
                            del diff_counts[d]
                        else:
                            diff_counts[d] = cnt - 1
                    for d in removed:
                        cnt = diff_counts.get(d, 0)
                        diff_counts[d] = cnt + 1
                    conflict_count -= delta_conf
                    row_diffs_list[ri] = old_diffs
                    cur_scope = _max_entry(cur)
                    cur_sum = _sum_entries(cur)
                    cur_cost = cur_scope + PENALTY * conflict_count + SUM_WEIGHT * cur_sum
        
        else:
            # Perturb two entries
            row = cur[ri]
            if len(row) < 3:
                temp = max(0.01, temp * cool)
                continue
            i = rng.randint(1, len(row) - 2)
            j = rng.randint(i + 1, len(row) - 1)
            if row[j] - row[i] < 2:
                temp = max(0.01, temp * cool)
                continue
            
            di = rng.randint(-3, 3)
            dj = rng.randint(-3, 3)
            ni = row[i] + di
            nj = row[j] + dj
            
            if ni <= 0 or nj <= 0:
                temp = max(0.01, temp * cool)
                continue
            if i > 1 and ni <= row[i-1]:
                temp = max(0.01, temp * cool)
                continue
            if j < len(row) - 1 and nj >= row[j+1]:
                temp = max(0.01, temp * cool)
                continue
            if ni >= nj:
                temp = max(0.01, temp * cool)
                continue
            if ni == row[i] and nj == row[j]:
                temp = max(0.01, temp * cool)
                continue
            
            old_diffs = row_diffs_list[ri]
            old_diff_set = set(old_diffs)
            
            row[i] = ni
            row[j] = nj
            new_diffs = _row_diffs(row)
            new_diff_set = set(new_diffs)
            
            removed = old_diff_set - new_diff_set
            added = new_diff_set - old_diff_set
            
            delta_conf = 0
            for d in removed:
                cnt = diff_counts.get(d, 0)
                if cnt == 2:
                    delta_conf -= 1
                elif cnt > 2:
                    delta_conf -= 1
            for d in added:
                cnt = diff_counts.get(d, 0)
                if cnt == 1:
                    delta_conf += 1
                elif cnt > 1:
                    delta_conf += 1
            
            for d in removed:
                cnt = diff_counts[d]
                if cnt == 1:
                    del diff_counts[d]
                else:
                    diff_counts[d] = cnt - 1
            for d in added:
                cnt = diff_counts.get(d, 0)
                diff_counts[d] = cnt + 1
            
            conflict_count += delta_conf
            row_diffs_list[ri] = new_diffs
            cur_scope = _max_entry(cur)
            cur_sum = _sum_entries(cur)
            cur_cost = cur_scope + PENALTY * conflict_count + SUM_WEIGHT * cur_sum
            
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_scope = cur_scope
                best_conf = conflict_count
                best_sum = cur_sum
                best = [list(r) for r in cur]
            elif conflict_count > best_conf + 3:
                # Revert
                row[i] = ni - di
                row[j] = nj - dj
                for d in added:
                    cnt = diff_counts[d]
                    if cnt == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = cnt - 1
                for d in removed:
                    cnt = diff_counts.get(d, 0)
                    diff_counts[d] = cnt + 1
                conflict_count -= delta_conf
                row_diffs_list[ri] = old_diffs
                cur_scope = _max_entry(cur)
                cur_sum = _sum_entries(cur)
                cur_cost = cur_scope + PENALTY * conflict_count + SUM_WEIGHT * cur_cost
        
        temp = max(0.01, temp * cool)
    
    if _is_valid(best, k):
        return best
    if _is_valid(cur, k):
        return cur
    return best

def _aggressive_polish(rows, k, rng, time_budget=10):
    """Aggressive final polish: try all entry positions with multiple reduction amounts."""
    best = [list(r) for r in rows]
    best_scope = _max_entry(best)
    
    t0 = time.time()
    improved = True
    while improved and time.time() - t0 < time_budget:
        if _elapsed() > _DEADLINE - 2:
            break
        improved = False
        all_diffs = set()
        for row in best:
            for d in _row_diffs(row):
                all_diffs.add(d)
        
        for ri in range(len(best)):
            if _elapsed() > _DEADLINE - 2:
                break
            row = best[ri]
            for idx in range(1, len(row)):
                if _elapsed() > _DEADLINE - 2:
                    break
                old_val = row[idx]
                for red in range(1, 10):
                    new_val = old_val - red
                    if new_val <= 0:
                        break
                    if idx > 1 and new_val <= row[idx-1]:
                        break
                    if idx < len(row) - 1 and new_val >= row[idx+1]:
                        break
                    
                    old_d = set(_row_diffs(row))
                    row[idx] = new_val
                    new_d = set(_row_diffs(row))
                    
                    temp_set = set(all_diffs)
                    for d in old_d:
                        temp_set.discard(d)
                    ok = True
                    for d in new_d:
                        if d in temp_set:
                            ok = False
                            break
                        temp_set.add(d)
                    
                    if ok:
                        all_diffs = temp_set
                        new_scope = _max_entry(best)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            improved = True
                        break
                    else:
                        row[idx] = old_val
                        continue
    
    if _is_valid(best, k):
        return best
    return [list(r) for r in rows]

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    rng = random.Random(20250705)
    
    # Phase 1: Initial construction
    rows, score = _initial_construction(n_rows, k, rng)
    
    if rows is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                nd = {x - a for a in row}
                if not (nd & used):
                    row.append(x)
                    used |= nd
            rows.append(row)
        score = _max_entry(rows)
    
    # Phase 2: Hungarian-style matching for compression
    if _elapsed() < _DEADLINE - 15:
        tb = min(15, _DEADLINE - 15 - _elapsed())
        if tb > 2:
            matched, ms = _hungarian_match(rows, k, rng, score)
            if matched and _is_valid(matched, k):
                if ms < score:
                    rows = matched
                    score = ms
    
    # Phase 3: SA optimization
    if _elapsed() < _DEADLINE - 12:
        tb = min(100, _DEADLINE - 12 - _elapsed())
        if tb > 5:
            opt = _sa_main(rows, k, rng, time_budget=tb)
            if opt and _is_valid(opt, k):
                ns = _max_entry(opt)
                if ns < score:
                    rows = opt
                    score = ns
    
    # Phase 4: Aggressive polish
    if _elapsed() < _DEADLINE - 2:
        tb = min(10, _DEADLINE - 2 - _elapsed())
        if tb > 0.5:
            pol = _aggressive_polish(rows, k, rng, time_budget=tb)
            if pol and _is_valid(pol, k):
                ns = _max_entry(pol)
                if ns < score:
                    rows = pol
                    score = ns
    
    # Final validation
    if not _is_valid(rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                nd = {x - a for a in row}
                if not (nd & used):
                    row.append(x)
                    used |= nd
            rows.append(row)
    
    return rows
