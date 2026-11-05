"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
import random
import math
from collections import deque

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses simulated annealing with multi-row rebuild moves and polish phase.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs_needed:
        lb += 1
    
    start_time = time.time()
    time_limit = 115.0
    
    def greedy_construct(n_rows, k):
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
    
    best_rows = greedy_construct(n_rows, k)
    best_max = max(max(r) for r in best_rows)
    
    def compute_all_diffs(rows):
        diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
        return diffs
    
    def get_max_entry(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    def compute_diffs_for_row(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def build_row_backtrack(remaining_used, max_entry, k):
        row = [0]
        def bt(current_row):
            if len(current_row) == k + 1:
                return True
            last = current_row[-1]
            rem = k + 1 - len(current_row)
            for x in range(last + 1, max_entry + 1):
                if max_entry - x < rem - 1:
                    break
                new_diffs = set()
                for a in current_row:
                    d = x - a
                    if d > 0:
                        new_diffs.add(d)
                if new_diffs & remaining_used:
                    continue
                current_row.append(x)
                if bt(current_row):
                    return True
                current_row.pop()
            return False
        if bt(row):
            return row
        return None
    
    def sa_phase(initial_rows, max_entry_cap, time_budget):
        deadline = time.time() + time_budget
        
        rows = []
        for r in initial_rows:
            truncated = [x for x in r if x <= max_entry_cap]
            if len(truncated) == k + 1:
                rows.append(truncated[:])
            else:
                rows.append(None)
        
        used_diffs = set()
        for i, r in enumerate(rows):
            if r is not None:
                used_diffs |= compute_diffs_for_row(r)
        
        for i in range(len(rows)):
            if rows[i] is None:
                built = build_row_backtrack(used_diffs, max_entry_cap, k)
                if built is None:
                    return None, None
                rows[i] = built
                used_diffs |= compute_diffs_for_row(built)
        
        current_max = get_max_entry(rows)
        if current_max > max_entry_cap:
            return None, None
        
        T = max_entry_cap * 0.5
        T_min = 0.5
        best_local = [r[:] for r in rows]
        best_local_max = current_max
        
        step = 0
        max_steps = 200000
        
        while time.time() < deadline and step < max_steps:
            step += 1
            T = max(T_min, T * 0.9995)
            
            move_type = random.random()
            
            if move_type < 0.6:
                row_idx = random.randint(0, n_rows - 1)
                pos_idx = random.randint(1, k)
                row = rows[row_idx]
                old_val = row[pos_idx]
                
                min_val = row[pos_idx - 1] + 1 if pos_idx > 0 else 1
                max_val = row[pos_idx + 1] - 1 if pos_idx < k else max_entry_cap
                
                if min_val < old_val:
                    new_val = random.randint(min_val, old_val - 1)
                    
                    new_diffs = set()
                    for i in range(len(row)):
                        if i != pos_idx:
                            d = abs(row[i] - new_val)
                            if d > 0:
                                new_diffs.add(d)
                    
                    old_diffs = set()
                    for i in range(len(row)):
                        if i != pos_idx:
                            d = abs(row[i] - old_val)
                            if d > 0:
                                old_diffs.add(d)
                    
                    other_diffs = used_diffs - old_diffs
                    if not (new_diffs & other_diffs):
                        row[pos_idx] = new_val
                        used_diffs = other_diffs | new_diffs
                        
                        new_max = get_max_entry(rows)
                        if new_max > current_max:
                            row[pos_idx] = old_val
                            used_diffs = compute_all_diffs(rows)
                        else:
                            current_max = new_max
                            if current_max < best_local_max:
                                best_local_max = current_max
                                best_local = [r[:] for r in rows]
            
            elif move_type < 0.85:
                row_idx = random.randint(0, n_rows - 1)
                old_row = rows[row_idx][:]
                old_diffs = compute_diffs_for_row(old_row)
                remaining = used_diffs - old_diffs
                
                new_row = build_row_backtrack(remaining, max_entry_cap, k)
                if new_row is not None:
                    new_diffs = compute_diffs_for_row(new_row)
                    new_max = max(get_max_entry([r for i, r in enumerate(rows) if i != row_idx]), new_row[-1])
                    
                    if new_max <= current_max or random.random() < math.exp(-(new_max - current_max) / T):
                        rows[row_idx] = new_row
                        used_diffs = remaining | new_diffs
                        current_max = new_max
                        if current_max < best_local_max:
                            best_local_max = current_max
                            best_local = [r[:] for r in rows]
                    else:
                        rows[row_idx] = old_row
                        used_diffs = compute_all_diffs(rows)
            
            else:
                if n_rows >= 2:
                    row_idx1, row_idx2 = random.sample(range(n_rows), 2)
                    old_row1 = rows[row_idx1][:]
                    old_row2 = rows[row_idx2][:]
                    old_diffs1 = compute_diffs_for_row(old_row1)
                    old_diffs2 = compute_diffs_for_row(old_row2)
                    remaining = used_diffs - old_diffs1 - old_diffs2
                    
                    new_row1 = build_row_backtrack(remaining, max_entry_cap, k)
                    if new_row1 is not None:
                        new_diffs1 = compute_diffs_for_row(new_row1)
                        remaining2 = remaining | new_diffs1
                        new_row2 = build_row_backtrack(remaining2, max_entry_cap, k)
                        if new_row2 is not None:
                            new_diffs2 = compute_diffs_for_row(new_row2)
                            new_max = max(get_max_entry([r for i, r in enumerate(rows) if i not in (row_idx1, row_idx2)]), new_row1[-1], new_row2[-1])
                            
                            if new_max <= current_max or random.random() < math.exp(-(new_max - current_max) / T):
                                rows[row_idx1] = new_row1
                                rows[row_idx2] = new_row2
                                used_diffs = remaining | new_diffs1 | new_diffs2
                                current_max = new_max
                                if current_max < best_local_max:
                                    best_local_max = current_max
                                    best_local = [r[:] for r in rows]
                            else:
                                rows[row_idx1] = old_row1
                                rows[row_idx2] = old_row2
                                used_diffs = compute_all_diffs(rows)
            
            if step % 500 == 0:
                used_diffs = compute_all_diffs(rows)
        
        if best_local_max <= max_entry_cap:
            return best_local, best_local_max
        return None, None
    
    for target_max in range(lb, best_max + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 5
        
        if remaining_time < 2.0:
            break
        
        dist = target_max - lb
        if dist == 0:
            budget = min(20.0, remaining_time)
        elif dist <= 3:
            budget = min(12.0, remaining_time)
        elif dist <= 8:
            budget = min(7.0, remaining_time)
        else:
            budget = min(4.0, remaining_time)
        
        if budget < 1.0:
            break
        
        sa_rows, sa_max = sa_phase(best_rows, target_max, budget)
        
        if sa_rows is not None and sa_max < best_max:
            best_rows = sa_rows
            best_max = sa_max
    
    elapsed = time.time() - start_time
    remaining_time = time_limit - elapsed - 3
    if remaining_time > 3.0 and best_max > lb:
        target = best_max - 1
        budget = min(remaining_time, 12.0)
        sa_rows, sa_max = sa_phase(best_rows, target, budget)
        if sa_rows is not None and sa_max < best_max:
            best_rows = sa_rows
            best_max = sa_max
    
    if not is_valid(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
