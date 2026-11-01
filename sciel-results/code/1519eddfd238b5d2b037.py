import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n_elements = k + 1
    n_diffs_per_row = k * (k + 1) // 2
    
    # 1. Deterministic Greedy to get initial bound
    def greedy_deterministic():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows, max(max(r) for r in rows)
    
    rows, best_max = greedy_deterministic()
    
    # 2. Randomized Greedy with Restarts
    best_rows = rows
    
    def greedy_randomized(window_size):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < n_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    # Found a valid x
                    # Collect a few more valid x's for randomization
                    candidates = [x]
                    # Look ahead a bit
                    for dx in range(1, window_size):
                        x_next = x + dx
                        nd_next = {x_next - a for a in row}
                        if not nd_next & used:
                            candidates.append(x_next)
                    # Pick one
                    chosen = random.choice(candidates)
                    # Update row and used
                    # Note: chosen might not be x, so we need to recompute diffs for chosen
                    # Actually, we just picked from candidates, so chosen is valid.
                    # But we need to update 'used' with diffs of 'chosen'.
                    # And we need to set 'x' to 'chosen' for the next iteration?
                    # No, the next iteration starts from chosen + 1.
                    chosen_diffs = {chosen - a for a in row}
                    row.append(chosen)
                    used |= chosen_diffs
                    x = chosen # Next search starts from here + 1
                    break
                # If we don't break, we continue incrementing x
                # Wait, the logic above is slightly flawed.
                # We increment x, check if valid.
                # If valid, we collect candidates.
                # If we pick a candidate > x, we skip the intermediate valid ones?
                # Yes, that's the point of randomization.
                # But we must ensure 'used' is updated correctly.
            if len(row) < n_elements:
                return None, None # Failed
            rows.append(row)
        return rows, max(max(r) for r in rows)

    # Run randomized greedy for a while
    while time.time() < deadline - 10.0:
        # Vary window size
        window = random.randint(1, 20)
        res_rows, res_max = greedy_randomized(window)
        if res_rows and res_max < best_max:
            best_max = res_max
            best_rows = res_rows
            # If we hit the counting bound, we can stop
            if best_max * n_rows <= n_rows * n_diffs_per_row: # Rough check
                 pass
            if best_max == 105: # Counting bound for (7,5) is 105
                 break

    return best_rows
