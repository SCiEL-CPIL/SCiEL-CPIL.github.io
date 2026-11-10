import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Greedy Baseline
    best_rows = greedy_construct(n_rows, k)
    best_max = max(max(r) for r in best_rows)
    
    # If greedy is already very good, we might skip heavy search?
    # But we want to try to beat it.
    
    # 2. Generate all Golomb rulers of length k+1 with max <= 112
    # We use 112 as a hard cap for the search to ensure we find the known optimum if possible.
    # If we can't find 112, we might find 113, etc.
    # But generating up to 112 is fast.
    MAX_SEARCH_LIMIT = 112
    
    def get_diffs(row):
        diffs = 0
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs |= 1 << (row[j] - row[i])
        return diffs

    def generate_rulers(limit):
        rulers = []
        def backtrack(row, last, used):
            if len(row) == k + 1:
                if row[-1] <= limit:
                    rulers.append(list(row))
                return
            for x in range(last + 1, limit + 1):
                new_diffs = 0
                valid = True
                for a in row:
                    d = x - a
                    if (used >> d) & 1:
                        valid = False
                        break
                    new_diffs |= 1 << d
                if valid:
                    row.append(x)
                    backtrack(row, x, used | new_diffs)
                    row.pop()
        backtrack([0], 0, 0)
        return rulers

    all_rulers = generate_rulers(MAX_SEARCH_LIMIT)
    
    if not all_rulers:
        return best_rows

    # Sort by max value to find tight solutions first
    all_rulers.sort(key=lambda r: r[-1])
    ruler_diffs = [get_diffs(r) for r in all_rulers]
    
    # 3. Backtracking search for a set of n_rows disjoint rulers
    # We fix the first ruler to break symmetry.
    
    def search_set(start_idx, used_diffs, current_set, current_max):
        nonlocal best_rows, best_max, deadline
        
        if time.time() > deadline:
            return False
            
        if len(current_set) == n_rows:
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(r) for r in current_set]
                # If we hit the theoretical lower bound or known optimum, we can stop
                if current_max <= 105: # Counting bound
                    return True
            return False
            
        for i in range(start_idx, len(all_rulers)):
            r = all_rulers[i]
            d = ruler_diffs[i]
            
            new_max = max(current_max, r[-1])
            if new_max >= best_max:
                continue
                
            if d & used_diffs:
                continue
                
            current_set.append(r)
            # Recurse
            search_set(i + 1, used_diffs | d, current_set, new_max)
            current_set.pop()
            
            # Optional: If we found a solution with max 112, we can break?
            # Since we sorted by max, if we found 112, we can't find < 112 
            # unless there are rulers with max < 112 that we haven't tried in this branch?
            # But we are iterating i in increasing order of max (due to sort).
            # So if we found a solution, it used rulers with max <= new_max.
            # Any other solution in this branch would use rulers with index > i, 
            # which have max >= r[-1].
            # So we might find a solution with the same max or higher.
            # We don't need to find higher.
            # So we can break if we found a solution?
            # No, because a different combination of *earlier* rulers (smaller index) 
            # might have been skipped? No, we are iterating i.
            # The recursion explores all combinations of rulers with index >= i.
            # If we found a solution, it's valid.
            # We update best_max.
            # We can continue to see if we find an even smaller max?
            # Since we sorted by max, and we prune `new_max >= best_max`,
            # if we found a solution with max 112, `best_max` becomes 112.
            # Any further branch with `new_max >= 112` is pruned.
            # So we will only explore branches that *could* have max < 112.
            # So we should NOT break. We should let it finish to find < 112 if possible.
            
        return False

    # Run search
    # We can randomize the order of rulers to avoid getting stuck?
    # But we sorted by max.
    # Let's just run it.
    
    search_set(0, 0, [], 0)
    
    return best_rows

def greedy_construct(n_rows, k):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            valid = True
            for a in row:
                d = x - a
                if (used >> d) & 1:
                    valid = False
                    break
                new |= 1 << d
            if valid:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
