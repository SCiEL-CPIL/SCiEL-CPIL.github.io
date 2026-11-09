import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_rows = None
    best_max = float('inf')
    
    # Initial greedy solution to set a good upper bound
    used_diffs = set()
    greedy_rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used_diffs:
                row.append(x)
                used_diffs |= new_diffs
        greedy_rows.append(row)
    best_max = max(max(r) for r in greedy_rows)
    best_rows = greedy_rows
    
    # Bitset-based CSP with forward checking and randomized restarts
    # We use bitmasks for efficient difference checks.
    # Max possible diff is bounded by best_max initially, but we search for lower targets.
    # Let's define a maximum search bound. The problem size is small (k=8, n_rows=4).
    # Max entry ~134. Diffs are 1..133.
    # We can use an integer bitmask where bit d is set if diff d is used.
    
    def get_diff_mask_for_row(row):
        mask = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                d = row[j] - row[i]
                mask |= (1 << d)
        return mask

    def try_construct(target_max, n_rows, k, random_state):
        used_diff_mask = 0
        rows = []
        
        def backtrack_row(idx, row, used_diff_mask):
            if idx == k:
                return row[:]
            
            last = row[-1]
            
            # Collect valid candidates
            valid_candidates = []
            for x in range(last + 1, target_max + 1):
                # Check if adding x creates any conflict
                # Diffs are x - a for a in row
                # We need to check if any of these diffs are already in used_diff_mask
                # And they must be distinct (they are, since row is strictly increasing)
                
                conflict = False
                for a in row:
                    d = x - a
                    if (used_diff_mask >> d) & 1:
                        conflict = True
                        break
                
                if not conflict:
                    valid_candidates.append(x)
                    
            if not valid_candidates:
                return None
            
            # Heuristic: Prefer smaller x, but randomize
            # We can shuffle a subset of the smallest candidates
            W = 10
            candidates_to_try = valid_candidates[:W] if len(valid_candidates) > W else valid_candidates
            random_state.shuffle(candidates_to_try)
            
            for x in candidates_to_try:
                new_row = row + [x]
                # Calculate new diffs mask
                new_diffs_mask = 0
                for a in row:
                    d = x - a
                    new_diffs_mask |= (1 << d)
                
                res = backtrack_row(idx + 1, new_row, used_diff_mask | new_diffs_mask)
                if res is not None:
                    return res
                    
            return None

        # Main loop for rows
        for i in range(n_rows):
            row = [0]
            res = backtrack_row(0, row, used_diff_mask)
            if res is None:
                return None
            # Update global mask
            used_diff_mask |= get_diff_mask_for_row(res)
            rows.append(res)
            
        return rows

    # Local search / perturbation to improve a given solution
    def local_search(rows, max_iter=10000, time_limit=None):
        if time_limit is None:
            time_limit = deadline - 5
        current_rows = [r[:] for r in rows]
        current_max = max(max(r) for r in current_rows)
        
        def get_all_diffs(rows):
            diffs = set()
            for row in rows:
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        diffs.add(row[j] - row[i])
            return diffs
        
        def is_valid(rows):
            diffs = get_all_diffs(rows)
            expected_count = n_rows * k * (k + 1) // 2
            return len(diffs) == expected_count
        
        if not is_valid(current_rows):
            return current_rows, current_max
            
        for _ in range(max_iter):
            if time.time() > time_limit:
                break
                
            # Try to reduce the max entry
            # Find a row with the largest entry
            max_row_idx = max(range(n_rows), key=lambda i: max(current_rows[i]))
            row = current_rows[max_row_idx]
            
            if max(row) <= 1:
                break
                
            # Try to reduce the last element of the row
            # We need to find a smaller value for row[-1] such that all diffs remain unique
            last_val = row[-1]
            new_val = None
            
            # Try values smaller than last_val
            for candidate in range(last_val - 1, max(row[:-1]), -1):
                test_row = row[:-1] + [candidate]
                # Check if this creates any duplicate diffs
                test_diffs = set()
                for i in range(len(test_row)):
                    for j in range(i+1, len(test_row)):
                        d = test_row[j] - test_row[i]
                        test_diffs.add(d)
                
                # Check against other rows
                other_diffs = set()
                for idx in range(n_rows):
                    if idx == max_row_idx:
                        continue
                    for i in range(len(current_rows[idx])):
                        for j in range(i+1, len(current_rows[idx])):
                            other_diffs.add(current_rows[idx][j] - current_rows[idx][i])
                
                if not (test_diffs & other_diffs):
                    new_val = candidate
                    break
            
            if new_val is not None:
                row[-1] = new_val
                new_max = max(max(r) for r in current_rows)
                if new_max < current_max:
                    current_max = new_max
                    if current_max == best_max:
                        # We might be able to do better, continue
                        pass
                # Even if max doesn't change, the structure might allow further reductions
            else:
                # Try perturbing another position
                # Pick a random position in the max row (not the first)
                if len(row) > 2:
                    pos = random_state.randint(1, len(row) - 2)
                    orig_val = row[pos]
                    # Try to shift it
                    for delta in random_state.sample(range(-5, 6), 10):
                        if delta == 0:
                            continue
                        new_candidate = orig_val + delta
                        if new_candidate <= row[pos-1] or new_candidate >= row[pos+1]:
                            continue
                        test_row = row[:]
                        test_row[pos] = new_candidate
                        test_diffs = set()
                        for i in range(len(test_row)):
                            for j in range(i+1, len(test_row)):
                                test_diffs.add(test_row[j] - test_row[i])
                        
                        other_diffs = set()
                        for idx in range(n_rows):
                            if idx == max_row_idx:
                                continue
                            for i in range(len(current_rows[idx])):
                                for j in range(i+1, len(current_rows[idx])):
                                    other_diffs.add(current_rows[idx][j] - current_rows[idx][i])
                        
                        if not (test_diffs & other_diffs):
                            row[pos] = new_candidate
                            break
        
        return current_rows, current_max

    # Main search loop
    # We want to minimize the maximum entry.
    # Start from best_max - 1 and go down.
    # If we find a solution, update best_max and continue.
    
    current_target = best_max - 1
    random_state = random.Random()
    
    while time.time() < deadline and current_target >= 0:
        found = False
        # Try multiple restarts for this target
        max_restarts = 20
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            random_state = random.Random()
            res = try_construct(current_target, n_rows, k, random_state)
            if res is not None:
                # Try to improve with local search
                improved_rows, improved_max = local_search(res, max_iter=5000, time_limit=deadline - 10)
                if improved_max < best_max:
                    best_max = improved_max
                    best_rows = improved_rows
                found = True
                break
        
        if found:
            current_target = best_max - 1
        else:
            # If we fail, we just keep the current best.
            break

    # Final local search polish on the best solution found
    if best_rows is not None:
        polished_rows, polished_max = local_search(best_rows, max_iter=20000, time_limit=deadline - 5)
        if polished_max < best_max:
            best_max = polished_max
            best_rows = polished_rows

    return best_rows
