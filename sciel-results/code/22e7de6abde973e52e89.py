import time
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    # Use 140s to be safe against the 150s kill limit
    deadline = start_time + 140.0

    def greedy_construct():
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = 0
                for a in row:
                    new_diffs |= (1 << (x - a))
                if (new_diffs & used) == 0:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    baseline_max = max(max(row) for row in baseline_rows)

    # If k is 0, just return [[0] for _ in range(n_rows)]
    if k == 0:
        return [[0] for _ in range(n_rows)]

    total_diffs_needed = n_rows * (k * (k + 1) // 2)
    lower_bound = total_diffs_needed

    # If baseline is already at lower bound, it's optimal (or close)
    if baseline_max <= lower_bound:
        return baseline_rows

    def solve_with_limit(limit):
        rows_acc = []

        def _dfs(row_idx, pos_in_row, current_row, used_diffs):
            if time.time() > deadline:
                return None

            if row_idx == n_rows:
                return rows_acc + [current_row]

            if pos_in_row == k + 1:
                rows_acc.append(current_row)
                res = _dfs(row_idx + 1, 0, [0], used_diffs)
                if res:
                    return res
                rows_acc.pop()
                return None

            last_val = current_row[-1]
            # Pruning: Can we fit the remaining k - pos_in_row integers?
            # We need k - pos_in_row more integers.
            # Smallest they can be: x+1, ..., x + (k - pos_in_row)
            # Largest must be <= limit.
            
            # We iterate x from last_val + 1 to limit
            # If x + (k - pos_in_row) > limit, we break
            max_x_allowed = limit - (k - pos_in_row)
            
            for x in range(last_val + 1, max_x_allowed + 1):
                new_diffs = 0
                for a in current_row:
                    new_diffs |= (1 << (x - a))
                
                if new_diffs & used_diffs:
                    continue
                
                current_row.append(x)
                res = _dfs(row_idx, pos_in_row + 1, current_row, used_diffs | new_diffs)
                if res:
                    return res
                current_row.pop()
            
            return None

        return _dfs(0, 0, [0], 0)

    found = None
    # Try to find a solution with max < baseline_max
    # Iterate limits from lower_bound to baseline_max - 1
    # To avoid spending too much time on a single limit that fails, 
    # we rely on the global deadline.
    
    # We can also add a check: if baseline_max - lower_bound is large, 
    # maybe binary search? But linear is simpler and finds the *best* (smallest max) 
    # if we stop at the first success.
    
    for limit in range(lower_bound, baseline_max):
        if time.time() > deadline - 5.0:
            break
        res = solve_with_limit(limit)
        if res:
            found = res
            break

    if found:
        return found
    else:
        return baseline_rows
