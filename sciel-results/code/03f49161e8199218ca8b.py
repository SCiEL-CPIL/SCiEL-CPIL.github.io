import time
import random
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 138.0
    
    def row_diffs(row):
        out = set()
        n = len(row)
        for i in range(n):
            a = row[i]
            for j in range(i + 1, n):
                out.add(row[j] - a)
        return out
    
    def row_mask(row, max_diff):
        mask = 0
        n = len(row)
        for i in range(n):
            a = row[i]
            for j in range(i + 1, n):
                d = row[j] - a
                if d <= max_diff:
                    mask |= (1 << d)
        return mask

    def greedy_row(used_set, k, cap=None, rng=None):
        if rng is None:
            rng = random
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if cap is not None and x > cap:
                return None
            new = {x - a for a in row}
            if not (new & used_set):
                row.append(x)
                used_set |= new
        return row

    # Generate baseline
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(set(used), k)
        if row is None:
            row = [0] * (k+1)
        rows.append(row)
        used |= row_diffs(row)
    
    baseline_max = max(r[-1] for r in rows)
    best_rows = rows[:]
    best_max = baseline_max

    # Strategy 1: Pool generation and local search
    def generate_valid_rows(limit, k, time_limit):
        start = time.time()
        timeout = time_limit
        rows_list = []
        
        bit_of = [1 << d for d in range(limit + 1)]
        
        def backtrack(current_row, current_mask, last_val):
            if time.time() - start > timeout:
                return False
            
            if len(current_row) == k + 1:
                max_val = current_row[-1]
                rows_list.append((max_val, current_mask, tuple(current_row)))
                return True
            
            for x in range(last_val + 1, limit + 1):
                new_mask = 0
                valid = True
                for a in current_row:
                    d = x - a
                    if d > limit:
                        valid = False
                        break
                    bit = bit_of[d]
                    if current_mask & bit:
                        valid = False
                        break
                    new_mask |= bit
                
                if valid:
                    current_row.append(x)
                    if not backtrack(current_row, current_mask | new_mask, x):
                        return False
                    current_row.pop()
        
        if backtrack([0], 0, 0):
            return rows_list
        else:
            return None

    pool = None
    if time.time() < deadline - 100:
        pool = generate_valid_rows(baseline_max, k, 25.0)
        
        if pool is not None:
            pool.sort(key=lambda x: x[0])
            
            current_rows = [list(r) for r in best_rows]
            current_masks = []
            for r in current_rows:
                current_masks.append(row_mask(r, baseline_max))
            
            total_mask = 0
            for m in current_masks:
                total_mask |= m
                
            improved = True
            no_improve_count = 0
            rng = random.Random(42)
            
            while improved and time.time() < deadline - 8:
                improved = False
                no_improve_count += 1
                if no_improve_count > 100:
                    break
                
                max_idx = -1
                max_val = -1
                for i, r in enumerate(current_rows):
                    if r[-1] > max_val:
                        max_val = r[-1]
                        max_idx = i
                
                if max_idx == -1:
                    break
                
                others_mask = total_mask ^ current_masks[max_idx]
                
                best_cand = None
                best_cand_mask = 0
                best_cand_row = None
                
                for idx, (cand_max, cand_mask, cand_row) in enumerate(pool):
                    if cand_max >= max_val:
                        break
                    
                    if (cand_mask & others_mask) == 0:
                        best_cand = (cand_max, cand_mask, cand_row)
                        break
                
                if best_cand is not None:
                    cand_max, cand_mask, cand_row = best_cand
                    total_mask = (total_mask ^ current_masks[max_idx]) | cand_mask
                    current_rows[max_idx] = list(cand_row)
                    current_masks[max_idx] = cand_mask
                    improved = True
                    no_improve_count = 0
                    
                    current_global_max = max(r[-1] for r in current_rows)
                    if current_global_max < best_max:
                        best_max = current_global_max
                        best_rows = [list(r) for r in current_rows]

    # Strategy 2: Targeted local search with 2-row swaps
    def find_row_with_mask(others_mask, cap, limit, k, time_limit):
        """Find a valid row with max <= cap that doesn't conflict with others_mask"""
        start = time.time()
        timeout = time_limit
        bit_of = [1 << d for d in range(limit + 1)]
        
        def backtrack(current_row, current_mask, last_val):
            if time.time() - start > timeout:
                return None
            
            if len(current_row) == k + 1:
                return current_row[:]
            
            for x in range(last_val + 1, cap + 1):
                new_mask = 0
                valid = True
                for a in current_row:
                    d = x - a
                    if d > limit:
                        valid = False
                        break
                    bit = bit_of[d]
                    if current_mask & bit:
                        valid = False
                        break
                    if others_mask & bit:
                        valid = False
                        break
                    new_mask |= bit
                
                if valid:
                    current_row.append(x)
                    res = backtrack(current_row, current_mask | new_mask, x)
                    if res is not None:
                        return res
                    current_row.pop()
        
        return backtrack([0], 0, 0)

    # Phase 2: Iterated local search with restarts
    rng = random.Random(7)
    
    def current_state():
        used = set()
        for r in best_rows:
            used |= row_diffs(r)
        return used

    def try_improve(rows, used, max_iters=500):
        current_max = max(r[-1] for r in rows)
        best_local = [list(r) for r in rows]
        best_local_max = current_max
        no_improve = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            i = rng.randint(0, n_rows - 1)
            others_mask = 0
            for j in range(n_rows):
                if j != i:
                    others_mask |= row_mask(rows[j], best_local_max)
            
            cap = best_local_max - 1 if rng.random() < 0.7 else best_local_max
            cand = find_row_with_mask(others_mask, cap, best_local_max, k, 0.3)
            
            if cand is not None:
                new_last = cand[-1]
                max_others = 0
                for j in range(n_rows):
                    if j != i:
                        if rows[j][-1] > max_others:
                            max_others = rows[j][-1]
                
                new_global_max = max(max_others, new_last)
                if new_global_max < best_local_max:
                    rows[i] = cand
                    best_local = [list(r) for r in rows]
                    best_local_max = new_global_max
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 50:
                        break
        
        return best_local, best_local_max

    # Run multiple independent local search attempts
    attempts = 0
    while time.time() < deadline - 3:
        # Start from best found
        current = [list(r) for r in best_rows]
        improved_rows, improved_max = try_improve(current, set(), max_iters=300)
        
        if improved_max < best_max:
            best_max = improved_max
            best_rows = improved_rows
        
        attempts += 1
        if attempts > 20:
            break

    # Final validation
    def validate_solution(rows):
        all_diffs = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i] >= row[i+1]:
                    return False
            diffs = row_diffs(row)
            if diffs & all_diffs:
                return False
            all_diffs |= diffs
        return True

    if not validate_solution(best_rows):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = greedy_row(set(used), k)
            if row is None:
                row = [0] * (k+1)
            rows.append(row)
            used |= row_diffs(row)
        return rows

    return best_rows
