import random
import math
import time
import heapq

def construct(n_rows, k):
    row_len = k + 1
    MAX_DIFF_ESTIMATE = 200
    
    def is_valid_extension_array(row, new_val, used_diffs, size):
        if new_val <= row[-1]:
            return False
        for v in row:
            d = new_val - v
            if d >= size:
                return False
            if used_diffs[d]:
                return False
        return True

    def greedy_solution(randomize=False):
        rows = [[0] for _ in range(n_rows)]
        used = bytearray(MAX_DIFF_ESTIMATE)
        
        for step in range(n_rows * k):
            best_idx = -1
            best_key = None
            
            for i in range(n_rows):
                if len(rows[i]) < row_len:
                    noise = random.random() if randomize else 0
                    key = (len(rows[i]), rows[i][-1] + noise)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_idx = i
                        
            if best_idx == -1:
                break
                
            i = best_idx
            current_row = rows[i]
            x = current_row[-1] + 1
            
            while True:
                if is_valid_extension_array(current_row, x, used, MAX_DIFF_ESTIMATE):
                    current_row.append(x)
                    for v in current_row[:-1]:
                        d = x - v
                        if d < MAX_DIFF_ESTIMATE:
                            used[d] = 1
                    break
                x += 1
        return rows, max(max(r) for r in rows)

    start_time = time.time()
    timeout = 115
    
    best_solution = None
    best_cost = float('inf')
    
    # 1. Greedy Baseline
    for _ in range(10):
        if time.time() - start_time > timeout:
            break
        rows, cost = greedy_solution(randomize=True)
        if cost < best_cost:
            best_cost = cost
            best_solution = [r[:] for r in rows]
            
    rows, cost = greedy_solution(randomize=False)
    if cost < best_cost:
        best_cost = cost
        best_solution = [r[:] for r in rows]

    # 2. Optimized Beam Search
    beam_width = 5000 
    current_max_diff = best_cost
    
    initial_rows = [[0] for _ in range(n_rows)]
    initial_used = bytearray(current_max_diff + 1)
    beam = [(0, initial_rows, initial_used)]
    
    for step in range(n_rows * k):
        if time.time() - start_time > timeout:
            break
            
        next_beam = []
        
        for cost, rows, used_diffs in beam:
            # Identify target row
            target_idx = -1
            min_len = row_len
            min_last = float('inf')
            
            for i in range(n_rows):
                if len(rows[i]) < min_len:
                    min_len = len(rows[i])
                    min_last = rows[i][-1]
                    target_idx = i
                elif len(rows[i]) == min_len and rows[i][-1] < min_last:
                    min_last = rows[i][-1]
                    target_idx = i
                    
            if target_idx == -1:
                if cost < best_cost:
                    best_cost = cost
                    best_solution = [r[:] for r in rows]
                continue

            current_row = rows[target_idx]
            current_max_val = max(max(r) for r in rows)
            
            candidates = []
            x = current_row[-1] + 1
            limit_val = best_cost - 1
            
            while x <= limit_val and len(candidates) < 5:
                if is_valid_extension_array(current_row, x, used_diffs, len(used_diffs)):
                    candidates.append(x)
                x += 1
                
            if not candidates:
                continue

            for new_val in candidates:
                new_rows = [r[:] for r in rows]
                new_rows[target_idx].append(new_val)
                
                new_used_diffs = bytearray(used_diffs)
                for v in new_rows[target_idx]:
                    if v != new_val:
                        d = new_val - v
                        if d < len(new_used_diffs):
                            new_used_diffs[d] = 1
                            
                new_cost = max(current_max_val, new_val)
                
                if new_cost < best_cost:
                    next_beam.append((new_cost, new_rows, new_used_diffs))
        
        if not next_beam:
            break
            
        next_beam.sort(key=lambda x: x[0])
        
        if len(next_beam) > beam_width:
            from collections import defaultdict
            cost_groups = defaultdict(list)
            for item in next_beam:
                cost_groups[item[0]].append(item)
            
            for c, items in cost_groups.items():
                random.shuffle(items)
            
            next_beam = []
            for c in sorted(cost_groups.keys()):
                next_beam.extend(cost_groups[c])
                
            beam = next_beam[:beam_width]
        else:
            beam = next_beam
            
        for cost, rows, _ in beam:
            if all(len(r) == row_len for r in rows):
                if cost < best_cost:
                    best_cost = cost
                    best_solution = [r[:] for r in rows]
                    
    if best_solution is None:
        rows, _ = greedy_solution(randomize=False)
        best_solution = rows

    return best_solution
