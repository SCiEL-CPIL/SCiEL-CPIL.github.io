"""Difference triangle set (7,5) solver with Simulated Annealing and efficient diff tracking."""
import time
import random
import sys

# Deadline guard: return best result before 145s (5s safety margin under 150s kill)
_DEADLINE = 145.0
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

def _sum_entries(rows):
    """Return the sum of all entries across all rows (secondary objective)."""
    return sum(sum(r) for r in rows)

def _row_diffs(row):
    """Return the list of all positive differences in a row."""
    diffs = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs

def _max_entry(rows):
    return max(max(r) for r in rows) if rows else 0

def _is_valid(rows, k):
    """Check that all positive differences across all rows are distinct."""
    seen = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _greedy_row(used, k, rng, max_val=None):
    """Build one row greedily, choosing the smallest available next mark.
    used: set of forbidden differences.
    max_val: optional upper bound for entries.
    """
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        if max_val is not None and x > max_val:
            break
        # Occasionally jump ahead to create diversity
        if rng.random() < 0.15:
            x += rng.randint(1, 5)
            if max_val is not None and x > max_val:
                break
        
        new_diffs = {x - a for a in row}
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row

def _greedy_full(n_rows, k, rng, max_val=None):
    """Build all rows greedily with randomization."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k, rng, max_val)
        if len(row) < k + 1:
            return None
        rows.append(row)
    return rows

def _row_diffs_set(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _sa_optimize(rows, k, rng, time_budget=60):
    """Simulated Annealing to minimize the scope (max entry).
    
    Uses delta-based conflict tracking for O(1) cost evaluation per move.
    """
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    current_rows = [list(r) for r in best_rows]
    
    # Initialize global diff counts and conflict count
    diff_counts = {}
    conflict_count = 0
    for row in current_rows:
        for d in _row_diffs(row):
            old_cnt = diff_counts.get(d, 0)
            if old_cnt == 1:
                conflict_count += 1
            elif old_cnt > 1:
                conflict_count += 1
            diff_counts[d] = old_cnt + 1
    
    current_scope = _max_entry(current_rows)
    current_sum = _sum_entries(current_rows)
    
    PENALTY = 500
    SUM_WEIGHT = 0.05
    current_cost = current_scope + PENALTY * conflict_count + SUM_WEIGHT * current_sum
    
    best_scope = current_scope
    best_conflicts = conflict_count
    best_sum = current_sum
    best_cost = current_cost
    
    start_time = time.time()
    temp = 20.0
    cooling = 0.9993
    
    row_diffs_list = [_row_diffs(row) for row in current_rows]
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 5:
            break
        
        move_type = rng.random()
        
        if move_type < 0.35:
            # Move: Rebuild a random row (greedy with constraints)
            ri = rng.randrange(n_rows)
            old_row = current_rows[ri]
            old_diffs = row_diffs_list[ri]
            
            for d in old_diffs:
                old_cnt = diff_counts[d]
                if old_cnt == 2:
                    conflict_count -= 1
                elif old_cnt > 2:
                    conflict_count -= 1
                if old_cnt == 1:
                    del diff_counts[d]
                else:
                    diff_counts[d] = old_cnt - 1
            
            used = set(diff_counts.keys())
            max_limit = current_scope + rng.randint(0, 10)
            new_row = _greedy_row(used, k, rng, max_limit)
            
            if len(new_row) < k + 1:
                for d in old_diffs:
                    old_cnt = diff_counts.get(d, 0)
                    if old_cnt == 1:
                        conflict_count += 1
                    elif old_cnt > 1:
                        conflict_count += 1
                    diff_counts[d] = old_cnt + 1
                continue
            
            new_diffs = _row_diffs(new_row)
            for d in new_diffs:
                old_cnt = diff_counts.get(d, 0)
                if old_cnt == 1:
                    conflict_count += 1
                elif old_cnt > 1:
                    conflict_count += 1
                diff_counts[d] = old_cnt + 1
            
            current_rows[ri] = new_row
            row_diffs_list[ri] = new_diffs
            
        elif move_type < 0.80:
            # Move: Shift a single entry in a random row - more aggressive compression
            ri = rng.randrange(n_rows)
            row = current_rows[ri]
            row_len = len(row)
            if row_len < 2:
                continue
            
            idx = rng.randint(1, row_len - 1)
            old_val = row[idx]
            
            # More aggressive compression: try larger reductions
            r = rng.random()
            if r < 0.6:
                # Try to reduce (compress) - larger range
                new_val = old_val - rng.randint(1, 6)
            elif r < 0.85:
                # Small random perturbation
                new_val = old_val + rng.randint(-4, 4)
            else:
                # Larger random perturbation
                new_val = old_val + rng.randint(-10, 10)
            
            if new_val <= 0 or new_val == old_val:
                continue
            if idx > 1 and new_val <= row[idx - 1]:
                continue
            if idx < row_len - 1 and new_val >= row[idx + 1]:
                continue
            
            old_diffs = row_diffs_list[ri]
            old_diff_set = set(old_diffs)
            
            new_row = list(row)
            new_row[idx] = new_val
            new_diffs = _row_diffs(new_row)
            new_diff_set = set(new_diffs)
            
            if len(new_diffs) != len(new_diff_set):
                continue
            
            removed = old_diff_set - new_diff_set
            added = new_diff_set - old_diff_set
            
            delta_conflict = 0
            for d in removed:
                cnt = diff_counts.get(d, 0)
                if cnt == 2:
                    delta_conflict -= 1
                elif cnt > 2:
                    delta_conflict -= 1
            for d in added:
                cnt = diff_counts.get(d, 0)
                if cnt == 1:
                    delta_conflict += 1
                elif cnt > 1:
                    delta_conflict += 1
            
            other_max = 0
            for i in range(n_rows):
                if i != ri:
                    other_max = max(other_max, current_rows[i][-1])
            new_scope = max(new_row[-1], other_max)
            
            delta_sum = new_val - old_val
            
            new_conflicts = conflict_count + delta_conflict
            new_cost = new_scope + PENALTY * new_conflicts + SUM_WEIGHT * (current_sum + delta_sum)
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta_cost / temp)):
                for d in removed:
                    cnt = diff_counts[d]
                    if cnt == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = cnt - 1
                for d in added:
                    old_cnt = diff_counts.get(d, 0)
                    diff_counts[d] = old_cnt + 1
                
                row[idx] = new_val
                row_diffs_list[ri] = new_diffs
                conflict_count = new_conflicts
                current_scope = new_scope
                current_sum += delta_sum
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope = current_scope
                    best_conflicts = conflict_count
                    best_sum = current_sum
                    best_rows = [list(r) for r in current_rows]
        
        else:
            # Move: Swap two entries between two different rows
            ri1 = rng.randrange(n_rows)
            ri2 = rng.randrange(n_rows)
            if ri1 == ri2:
                continue
            row1 = current_rows[ri1]
            row2 = current_rows[ri2]
            
            # Try swapping the last element of row1 with last element of row2
            v1 = row1[-1]
            v2 = row2[-1]
            if v1 == v2:
                continue
            
            if len(row1) > 1 and v2 <= row1[-2]:
                continue
            if len(row2) > 1 and v1 <= row2[-2]:
                continue
            
            old_diffs1 = row_diffs_list[ri1]
            old_diffs2 = row_diffs_list[ri2]
            
            new_row1 = list(row1)
            new_row1[-1] = v2
            new_row2 = list(row2)
            new_row2[-1] = v1
            
            new_diffs1 = _row_diffs(new_row1)
            new_diffs2 = _row_diffs(new_row2)
            
            if len(new_diffs1) != len(set(new_diffs1)) or len(new_diffs2) != len(set(new_diffs2)):
                continue
            
            old_set1 = set(old_diffs1)
            old_set2 = set(old_diffs2)
            new_set1 = set(new_diffs1)
            new_set2 = set(new_diffs2)
            
            removed = (old_set1 - new_set1) | (old_set2 - new_set2)
            added = (new_set1 - old_set1) | (new_set2 - old_set2)
            
            delta_conflict = 0
            for d in removed:
                cnt = diff_counts.get(d, 0)
                if cnt == 2:
                    delta_conflict -= 1
                elif cnt > 2:
                    delta_conflict -= 1
            for d in added:
                cnt = diff_counts.get(d, 0)
                if cnt == 1:
                    delta_conflict += 1
                elif cnt > 1:
                    delta_conflict += 1
            
            new_scope = max(new_row1[-1], new_row2[-1], max((current_rows[i][-1] for i in range(n_rows) if i not in (ri1, ri2)), default=0))
            delta_sum = 0
            
            new_conflicts = conflict_count + delta_conflict
            new_cost = new_scope + PENALTY * new_conflicts + SUM_WEIGHT * current_sum
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or (temp > 0.01 and rng.random() < pow(2.718281828, -delta_cost / temp)):
                for d in removed:
                    cnt = diff_counts[d]
                    if cnt == 1:
                        del diff_counts[d]
                    else:
                        diff_counts[d] = cnt - 1
                for d in added:
                    old_cnt = diff_counts.get(d, 0)
                    diff_counts[d] = old_cnt + 1
                
                current_rows[ri1] = new_row1
                current_rows[ri2] = new_row2
                row_diffs_list[ri1] = new_diffs1
                row_diffs_list[ri2] = new_diffs2
                conflict_count = new_conflicts
                current_scope = new_scope
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_scope = current_scope
                    best_conflicts = conflict_count
                    best_sum = current_sum
                    best_rows = [list(r) for r in current_rows]

        temp *= cooling
        if temp < 0.01:
            temp = 0.01

    if _is_valid(best_rows, k):
        return best_rows
    if _is_valid(current_rows, k):
        return current_rows
    return best_rows

def _local_search_polish(rows, k, rng, time_budget=10):
    """Final high-precision polish: try single-entry shifts to reduce scope.
    
    Searches all entry positions for any improvement in scope or sum of entries,
    using delta-based conflict checking. Also tries small random perturbations.
    """
    n_rows = len(rows)
    best_rows = [list(r) for r in rows]
    best_scope = _max_entry(best_rows)
    best_sum = _sum_entries(best_rows)
    
    diff_counts = {}
    for row in best_rows:
        for d in _row_diffs(row):
            diff_counts[d] = diff_counts.get(d, 0) + 1
    
    row_diff_sets = [_row_diffs_set(row) for row in best_rows]
    row_diff_lists = [_row_diffs(row) for row in best_rows]
    
    start_time = time.time()
    no_improve_count = 0
    MAX_NO_IMPROVE = 12
    
    while time.time() - start_time < time_budget:
        if _elapsed() > _DEADLINE - 3:
            break
        
        improved = False
        for ri in range(n_rows):
            row = best_rows[ri]
            row_len = len(row)
            for idx in range(1, row_len):
                old_val = row[idx]
                
                # Try reducing by 1 through 10 (wider search)
                for reduction in range(1, 11):
                    new_val = old_val - reduction
                    if new_val <= 0:
                        break
                    if idx > 1 and new_val <= row[idx - 1]:
                        break
                    if idx < row_len - 1 and new_val >= row[idx + 1]:
                        continue
                
                    new_row = list(row)
                    new_row[idx] = new_val
                    new_diffs = _row_diffs(new_row)
                    new_diff_set = set(new_diffs)
                    
                    if len(new_diffs) != len(new_diff_set):
                        continue
                    
                    old_diff_set = row_diff_sets[ri]
                    removed = old_diff_set - new_diff_set
                    added = new_diff_set - old_diff_set
                    
                    conflict = False
                    for d in added:
                        other_cnt = diff_counts.get(d, 0)
                        if d in old_diff_set:
                            other_cnt -= 1
                        if other_cnt >= 1:
                            conflict = True
                            break
                    
                    if conflict:
                        continue
                    
                    new_scope = max(new_row[-1], max((best_rows[i][-1] for i in range(n_rows) if i != ri), default=0))
                    delta_sum = new_val - old_val
                    new_sum = best_sum + delta_sum
                    
                    if new_scope < best_scope or (new_scope == best_scope and new_sum < best_sum):
                        for d in removed:
                            cnt = diff_counts[d]
                            if cnt == 1:
                                del diff_counts[d]
                            else:
                                diff_counts[d] = cnt - 1
                        for d in added:
                            old_cnt = diff_counts.get(d, 0)
                            diff_counts[d] = old_cnt + 1
                        
                        row[idx] = new_val
                        row_diff_sets[ri] = new_diff_set
                        row_diff_lists[ri] = new_diffs
                        best_scope = new_scope
                        best_sum = new_sum
                        improved = True
                        no_improve_count = 0
                        break
                
                # Try small random perturbations if no reduction worked
                if not improved:
                    for _ in range(5):
                        new_val = old_val + rng.randint(-8, 8)
                        if new_val <= 0 or new_val == old_val:
                            continue
                        if idx > 1 and new_val <= row[idx - 1]:
                            continue
                        if idx < row_len - 1 and new_val >= row[idx + 1]:
                            continue
                        
                        new_row = list(row)
                        new_row[idx] = new_val
                        new_diffs = _row_diffs(new_row)
                        new_diff_set = set(new_diffs)
                        
                        if len(new_diffs) != len(new_diff_set):
                            continue
                        
                        old_diff_set = row_diff_sets[ri]
                        removed = old_diff_set - new_diff_set
                        added = new_diff_set - old_diff_set
                        
                        conflict = False
                        for d in added:
                            other_cnt = diff_counts.get(d, 0)
                            if d in old_diff_set:
                                other_cnt -= 1
                            if other_cnt >= 1:
                                conflict = True
                                break
                        
                        if conflict:
                            continue
                        
                        new_scope = max(new_row[-1], max((best_rows[i][-1] for i in range(n_rows) if i != ri), default=0))
                        delta_sum = new_val - old_val
                        new_sum = best_sum + delta_sum
                        
                        if new_scope < best_scope or (new_scope == best_scope and new_sum < best_sum):
                            for d in removed:
                                cnt = diff_counts[d]
                                if cnt == 1:
                                    del diff_counts[d]
                                else:
                                    diff_counts[d] = cnt - 1
                            for d in added:
                                old_cnt = diff_counts.get(d, 0)
                                diff_counts[d] = old_cnt + 1
                            
                            row[idx] = new_val
                            row_diff_sets[ri] = new_diff_set
                            row_diff_lists[ri] = new_diffs
                            best_scope = new_scope
                            best_sum = new_sum
                            improved = True
                            no_improve_count = 0
                            break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= MAX_NO_IMPROVE:
                break
        else:
            no_improve_count = 0
    
    if _is_valid(best_rows, k):
        return best_rows
    return [list(r) for r in rows]

def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Phase 1: Multiple randomized greedy restarts to find a valid start
    best_rows = None
    best_score = float('inf')
    
    n_restarts = 60
    for i in range(n_restarts):
        if _elapsed() > _DEADLINE - 40:
            break
        rng2 = random.Random(12345 + i * 7919)
        rows = _greedy_full(n_rows, k, rng2)
        if rows is None:
            continue
        if not _is_valid(rows, k):
            continue
        score = _max_entry(rows)
        if score < best_score:
            best_score = score
            best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: pure greedy without randomization jumps
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
        best_score = _max_entry(best_rows)
    
    # Phase 2: Simulated Annealing (main optimization)
    if _elapsed() < _DEADLINE - 15:
        time_left = _DEADLINE - 15 - _elapsed()
        sa_time = min(110, time_left)
        improved_rows = _sa_optimize(best_rows, k, rng, time_budget=sa_time)
        
        if improved_rows and _is_valid(improved_rows, k):
            new_score = _max_entry(improved_rows)
            if new_score < best_score:
                best_rows = improved_rows
                best_score = new_score
    
    # Phase 3: Final high-precision polish (extended)
    if _elapsed() < _DEADLINE - 3:
        polish_time = min(18, _DEADLINE - 3 - _elapsed())
        if polish_time > 0:
            polished_rows = _local_search_polish(best_rows, k, rng, time_budget=polish_time)
            if polished_rows and _is_valid(polished_rows, k):
                new_score = _max_entry(polished_rows)
                if new_score < best_score:
                    best_rows = polished_rows
                    best_score = new_score
    
    # Final validation
    if not _is_valid(best_rows, k):
        # Shouldn't happen, but fallback to a known valid greedy
        used = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best_rows.append(row)
    
    return best_rows
