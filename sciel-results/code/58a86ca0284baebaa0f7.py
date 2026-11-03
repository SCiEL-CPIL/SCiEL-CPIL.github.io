"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    """
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]
    
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row

    def get_diffs(row):
        """Calculate the set of positive differences for a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def max_entry(rows):
        return max(max(r) for r in rows)

    def sum_entries(rows):
        return sum(sum(r) for r in rows)

    def greedy_row(used_diffs, max_val=None, randomize=False, skip_prob=0.1, prefer_small=True):
        """
        Build a row greedily.
        If randomize is True, it may skip valid candidates to explore.
        prefer_small: if True, pick smallest valid; if False, pick from valid candidates randomly.
        """
        row = [0]
        local_diffs = set()
        x = 0
        limit = max_val if max_val else 500
        
        while len(row) < k + 1:
            x += 1
            if max_val and x > max_val:
                return None, None
            
            new_diffs = {x - a for a in row}
            
            # Check against global and local
            if new_diffs & used_diffs:
                continue
            if new_diffs & local_diffs:
                continue
            
            # Acceptance probability
            if randomize and random.random() < skip_prob:
                continue
            
            row.append(x)
            local_diffs |= new_diffs
                
        return row, local_diffs

    def backtracking_row(used_diffs, max_val):
        """
        Try to build a row using backtracking for better packing.
        Returns (row, diffs) or (None, None).
        """
        row = [0]
        local_diffs = set()
        
        def backtrack(pos, current_row, current_diffs):
            if pos == k + 1:
                return True
            
            start = current_row[-1] + 1
            for x in range(start, max_val + 1):
                new_diffs = {x - a for a in current_row}
                if new_diffs & used_diffs or new_diffs & current_diffs:
                    continue
                current_row.append(x)
                old_diffs = set(current_diffs)
                current_diffs |= new_diffs
                if backtrack(pos + 1, current_row, current_diffs):
                    return True
                current_row.pop()
                current_diffs = old_diffs
            return False
        
        if backtrack(1, row, local_diffs):
            return row, local_diffs
        return None, None

    # 1. Initial Construction - try multiple strategies
    best_rows = None
    best_max = float('inf')
    
    # Strategy 0: Pure greedy (smallest first)
    used = set()
    rows = []
    valid = True
    for i in range(n_rows):
        row, row_diffs = greedy_row(used, randomize=False)
        if row is None:
            valid = False
            break
        rows.append(row)
        used |= row_diffs
    if valid:
        cur_max = max_entry(rows)
        if cur_max < best_max:
            best_max = cur_max
            best_rows = [r[:] for r in rows]

    # Strategy 1-5: Greedy with randomization
    for strategy in range(1, 6):
        random.seed(42 + strategy * 1000)
        used = set()
        rows = []
        valid = True
        skip_prob = 0.05 * strategy
        for i in range(n_rows):
            row, row_diffs = greedy_row(used, randomize=True, skip_prob=skip_prob)
            if row is None:
                valid = False
                break
            rows.append(row)
            used |= row_diffs
        if valid:
            cur_max = max_entry(rows)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [r[:] for r in rows]

    # Strategy 6-9: Backtracking for first few rows, greedy for rest
    for bt_rows in [1, 2, 3]:
        random.seed(999 + bt_rows)
        used = set()
        rows = []
        valid = True
        for i in range(n_rows):
            if i < bt_rows:
                row, row_diffs = backtracking_row(used, max_val=best_max + 20)
            else:
                row, row_diffs = greedy_row(used, randomize=False)
            if row is None:
                valid = False
                break
            rows.append(row)
            used |= row_diffs
        if valid:
            cur_max = max_entry(rows)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [r[:] for r in rows]

    if best_rows is None:
        # Fallback: very loose greedy
        used = set()
        rows = []
        for i in range(n_rows):
            row, row_diffs = greedy_row(used, max_val=1000)
            rows.append(row)
            used |= row_diffs
        return rows

    # 2. Local Search
    current_rows = [r[:] for r in best_rows]
    current_max = best_max
    current_sum = sum_entries(current_rows)
    
    # Pre-compute diffs
    row_diffs_list = [get_diffs(r) for r in current_rows]
    used_set = set()
    for d in row_diffs_list:
        used_set |= d
        
    start_time = time.time()
    time_limit = 115
    
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Strategy A: Try to shrink the row with the current max
        max_idx = max(range(n_rows), key=lambda i: current_rows[i][-1])
        
        # Try multiple approaches to rebuild
        old_diffs = row_diffs_list[max_idx]
        used_without_max = used_set - old_diffs
        
        # Try greedy with tighter bound
        new_row, new_diffs = greedy_row(used_without_max, max_val=current_max - 1)
        
        if new_row is None:
            # Try backtracking
            new_row, new_diffs = backtracking_row(used_without_max, current_max - 1)
        
        if new_row is not None:
            current_rows[max_idx] = new_row
            row_diffs_list[max_idx] = new_diffs
            used_set = used_without_max | new_diffs
            current_max = max_entry(current_rows)
            current_sum = sum_entries(current_rows)
            best_max = current_max
            best_rows = [r[:] for r in current_rows]
            improved = True
            no_improve_count = 0
            continue

        # Strategy B: Random Perturbation
        i = random.randint(0, n_rows - 1)
        
        old_diffs = row_diffs_list[i]
        used_without_i = used_set - old_diffs
        
        # Try with randomization
        skip_probs = [0.1, 0.2, 0.3, 0.5]
        sp = random.choice(skip_probs)
        new_row, new_diffs = greedy_row(used_without_i, max_val=current_max, randomize=True, skip_prob=sp)
        
        if new_row is None:
            # Try backtracking
            new_row, new_diffs = backtracking_row(used_without_i, current_max)
        
        if new_row is not None and new_row != current_rows[i]:
            new_max = max_entry([r if j != i else new_row for j, r in enumerate(current_rows)])
            new_sum = sum_entries([r if j != i else new_row for j, r in enumerate(current_rows)])
            
            accept = False
            if new_max < current_max:
                accept = True
            elif new_max == current_max:
                if new_sum < current_sum:
                    accept = True
                elif new_sum == current_sum:
                    if random.random() < 0.15:
                        accept = True
            
            if accept:
                current_rows[i] = new_row
                row_diffs_list[i] = new_diffs
                used_set = used_without_i | new_diffs
                current_max = new_max
                current_sum = new_sum
                
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in current_rows]
                    improved = True
                    no_improve_count = 0

        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Aggressive restart of worst row
                max_idx = max(range(n_rows), key=lambda i: current_rows[i][-1])
                old_diffs = row_diffs_list[max_idx]
                used_without_max = used_set - old_diffs
                new_row, new_diffs = backtracking_row(used_without_max, current_max)
                if new_row is None:
                    new_row, new_diffs = greedy_row(used_without_max, max_val=current_max, randomize=True, skip_prob=0.5)
                if new_row is not None:
                    current_rows[max_idx] = new_row
                    row_diffs_list[max_idx] = new_diffs
                    used_set = used_without_max | new_diffs
                    current_sum = sum_entries(current_rows)
                    no_improve_count = 0
                else:
                    break
        else:
            no_improve_count = 0
            
    return best_rows
# EVOLVE-BLOCK-END
