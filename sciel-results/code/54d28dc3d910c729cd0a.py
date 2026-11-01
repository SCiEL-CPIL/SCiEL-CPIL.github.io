import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety

    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def greedy_row_with_update(used_diffs, k, randomize=False, bias=0.8):
        """
        Constructs a single row of length k+1 starting at 0.
        Returns (row, new_used_diffs) or (None, used_diffs) if failed.
        """
        row = [0]
        local_used = set(used_diffs)
        for _ in range(k):
            candidates = []
            x = row[-1] + 1
            # We don't know the upper bound, but for small k it's small.
            # We can stop if we have enough candidates for randomization
            max_candidates_to_check = 50 if randomize else 1000
            checked = 0
            while checked < max_candidates_to_check:
                checked += 1
                valid = True
                for a in row:
                    d = x - a
                    if d in local_used:
                        valid = False
                        break
                if valid:
                    candidates.append(x)
                    if not randomize:
                        break
                    if len(candidates) >= 5:
                        break
                x += 1
            
            if not candidates:
                return None, used_diffs
            
            if randomize:
                if random.random() < bias and len(candidates) > 0:
                    next_val = candidates[0]
                else:
                    next_val = random.choice(candidates)
            else:
                next_val = candidates[0]
                
            row.append(next_val)
            for a in row[:-1]:
                local_used.add(next_val - a)
        
        return row, local_used

    def randomized_greedy_construction(n_rows, k, randomize=True, bias=0.7):
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row, new_used = greedy_row_with_update(used_diffs, k, randomize, bias)
            if row is None:
                return None
            rows.append(row)
            used_diffs = new_used
        return rows

    def local_search(rows, n_rows, k, time_limit):
        used_diffs = set()
        for r in rows:
            used_diffs |= get_diffs(r)
        
        current_scope = max(max(r) for r in rows)
        best_rows = [r[:] for r in rows]
        best_scope = current_scope
        
        temp = 10.0
        cool_rate = 0.99
        
        while time.time() < time_limit:
            temp *= cool_rate
            if temp < 0.1:
                break
                
            max_val = max(max(r) for r in rows)
            max_rows = [i for i, r in enumerate(rows) if max(r) == max_val]
            idx = random.choice(max_rows)
            
            old_row = rows[idx]
            old_diffs = get_diffs(old_row)
            used_diffs -= old_diffs
            
            new_row, new_used = greedy_row_with_update(used_diffs, k, randomize=True, bias=random.uniform(0.3, 0.9))
            
            if new_row is not None:
                new_scope = max(max(r) for r in rows[:idx] + [new_row] + rows[idx+1:])
                delta = new_scope - current_scope
                
                if delta < 0:
                    rows[idx] = new_row
                    used_diffs = new_used
                    current_scope = new_scope
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r[:] for r in rows]
                else:
                    if random.random() < math.exp(-delta / temp):
                        rows[idx] = new_row
                        used_diffs = new_used
                        current_scope = new_scope
            
        return best_rows, best_scope

    # Main Loop
    best_rows = None
    best_scope = float('inf')
    
    # Initial deterministic greedy
    init_rows = randomized_greedy_construction(n_rows, k, randomize=False)
    if init_rows:
        init_scope = max(max(r) for r in init_rows)
        if init_scope < best_scope:
            best_scope = init_scope
            best_rows = init_rows
            
    # Randomized restarts and local search
    while time.time() < deadline:
        # Generate a new candidate
        bias = random.uniform(0.3, 0.9)
        cand_rows = randomized_greedy_construction(n_rows, k, randomize=True, bias=bias)
        
        if cand_rows:
            cand_scope = max(max(r) for r in cand_rows)
            if cand_scope < best_scope:
                best_scope = cand_scope
                best_rows = [r[:] for r in cand_rows]
            
            # Try to improve the current best
            if time.time() < deadline - 10:
                ls_rows, ls_scope = local_search([r[:] for r in best_rows], n_rows, k, time.time() + 5)
                if ls_scope < best_scope:
                    best_scope = ls_scope
                    best_rows = ls_rows

    if best_rows is None:
        # Fallback
        return randomized_greedy_construction(n_rows, k, randomize=False)
        
    return best_rows
