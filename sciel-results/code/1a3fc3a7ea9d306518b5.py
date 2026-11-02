import time
import random
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    # Each row has k+1 elements starting at 0, so k gaps
    # Total differences per row: k*(k+1)//2
    # Total differences across all rows: n_rows * k*(k+1)//2
    # The largest entry must be at least the total number of distinct differences needed
    # but more precisely, the differences range from 1 to max_val, and we need
    # n_rows * k*(k+1)//2 distinct values in [1, max_val]
    
    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    
    # Greedy construction with bitset tracking
    def greedy_construct(n_rows, k, seed=0):
        rng = random.Random(seed)
        used = [False] * 2000  # bitset of used differences
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Try adding the smallest possible next value
                found = False
                for x in range(current_max + 1, 150):
                    new_diffs = [x - a for a in row]
                    # Check if any new difference is already used
                    valid = True
                    for d in new_diffs:
                        if d >= len(used) or used[d]:
                            valid = False
                            break
                    if valid:
                        # Mark new differences as used
                        for d in new_diffs:
                            if d < len(used):
                                used[d] = True
                            else:
                                # Extend if needed
                                while len(used) <= d:
                                    used.append(False)
                                used[d] = True
                        row.append(x)
                        current_max = x
                        found = True
                        break
                if not found:
                    return None  # Failed
            rows.append(row)
        
        return rows
    
    def is_valid(rows, n_rows, k):
        """Check if rows satisfy all constraints."""
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            # Check strictly increasing
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            # Check all differences
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def get_max_val(rows):
        return max(max(row) for row in rows)
    
    # Try greedy first
    best_rows = None
    best_max = float('inf')
    
    # Try multiple seeds for greedy
    for seed in range(20):
        if time.time() > deadline:
            break
        rows = greedy_construct(n_rows, k, seed)
        if rows and is_valid(rows, n_rows, k):
            max_val = get_max_val(rows)
            if max_val < best_max:
                best_max = max_val
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: simple greedy without randomization
        used = [False] * 2000
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                found = False
                for x in range(current_max + 1, 200):
                    new_diffs = [x - a for a in row]
                    valid = True
                    for d in new_diffs:
                        if d >= len(used) or used[d]:
                            valid = False
                            break
                    if valid:
                        for d in new_diffs:
                            if d < len(used):
                                used[d] = True
                            else:
                                while len(used) <= d:
                                    used.append(False)
                                used[d] = True
                        row.append(x)
                        current_max = x
                        found = True
                        break
                if not found:
                    # Return what we have (may be invalid, but better than crashing)
                    return rows if rows else [[0] * (k + 1) for _ in range(n_rows)]
            rows.append(row)
        return rows
    
    # Local improvement: try to reduce max values
    def local_improve(rows, n_rows, k, max_iter=1000):
        if time.time() > deadline:
            return rows
        rows = [list(r) for r in rows]
        current_max = get_max_val(rows)
        improved = True
        iter_count = 0
        
        while improved and iter_count < max_iter and time.time() < deadline:
            improved = False
            iter_count += 1
            
            for ri in range(n_rows):
                for pos in range(1, len(rows[ri])):
                    orig_val = rows[ri][pos]
                    # Try to reduce this value
                    for new_val in range(orig_val - 1, -1, -1):
                        if new_val <= rows[ri][pos - 1]:
                            break
                        # Check if this change is valid
                        test_row = rows[ri][:]
                        test_row[pos] = new_val
                        # Check strictly increasing
                        valid = True
                        for i in range(len(test_row) - 1):
                            if test_row[i + 1] <= test_row[i]:
                                valid = False
                                break
                        if not valid:
                            continue
                        # Check differences
                        used = set()
                        all_valid = True
                        for r_idx, row in enumerate(rows):
                            test_r = test_row if r_idx == ri else row
                            for i in range(len(test_r)):
                                for j in range(i + 1, len(test_r)):
                                    d = test_r[j] - test_r[i]
                                    if d in used:
                                        all_valid = False
                                        break
                                    used.add(d)
                            if not all_valid:
                                break
                        if all_valid:
                            rows[ri] = test_row
                            improved = True
                            break
                    if time.time() > deadline:
                        return rows
        
        return rows
    
    best_rows = local_improve(best_rows, n_rows, k)
    return best_rows
