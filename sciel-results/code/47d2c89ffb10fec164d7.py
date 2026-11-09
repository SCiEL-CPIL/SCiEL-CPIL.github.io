"""Rewritten solver for difference triangle set (n_rows, k) using randomized greedy with
windowed candidate selection and time-bounded local search."""
import time
import random
import sys

def construct(n_rows, k):
    # Deadline: evaluator kills at 150s; use 135s as hard limit to be safe.
    deadline = time.time() + 135.0
    
    # For the specific task (4,7): n_rows=4, k=7, so each row has 8 entries.
    # But we must handle general n_rows, k.
    
    # Total differences per row: k*(k+1)/2. For k=7: 28.
    # Total differences across all rows: n_rows * k*(k+1)/2.
    
    best_rows = None
    best_scope = float('inf')
    
    # Helper: check if a full set of rows is valid
    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def compute_scope(rows):
        return max(max(row) for row in rows)
    
    # Randomized greedy: build rows one by one, choosing smallest available mark
    # but with some randomization in the window of candidates.
    def greedy_randomized(n_rows, k, window=1, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try candidates in a window around x
                found = False
                for offset in range(window):
                    candidate = x + offset
                    if candidate <= 0:
                        continue
                    new_diffs = {candidate - a for a in row}
                    if not new_diffs & used:
                        row.append(candidate)
                        used |= new_diffs
                        x = candidate
                        found = True
                        break
                if not found:
                    x += 1
            rows.append(row)
        return rows
    
    # Local search: try to reduce the scope by modifying the last entry or
    # swapping some marks.
    def local_search(rows, n_rows, k, deadline):
        # Simple hill climbing: try to reduce the maximum entry
        best = [list(r) for r in rows]
        best_scope = compute_scope(best)
        
        # Identify the row with the maximum entry
        max_idx = max(range(n_rows), key=lambda i: max(best[i]))
        
        # Try to reduce the last element of the max row
        row = best[max_idx]
        last_val = row[-1]
        
        # Compute all differences in other rows
        used_by_others = set()
        for i in range(n_rows):
            if i == max_idx:
                continue
            for j in range(len(best[i])):
                for l in range(j+1, len(best[i])):
                    used_by_others.add(best[i][l] - best[i][j])
        
        # Differences within the row except those involving the last element
        used_by_row_no_last = set()
        for i in range(len(row)-1):
            for j in range(i+1, len(row)-1):
                used_by_row_no_last.add(row[j] - row[i])
        
        # Try to find a smaller value for the last element
        for new_val in range(last_val - 1, row[-2], -1):
            # Check if new_val creates differences that conflict
            new_diffs = {new_val - a for a in row[:-1]}
            if not (new_diffs & used_by_others) and not (new_diffs & used_by_row_no_last):
                # Also check that new_val > row[-2] (strictly increasing)
                if new_val > row[-2]:
                    row[-1] = new_val
                    # Verify full validity
                    if is_valid(best):
                        new_scope = compute_scope(best)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            break
                    else:
                        row[-1] = last_val
                        break
        
        return best, best_scope
    
    # Phase 1: Run randomized greedy with various windows and seeds
    windows_to_try = [1, 2, 3, 5, 8, 12, 20]
    num_seeds = 50
    
    for window in windows_to_try:
        for seed in range(num_seeds):
            if time.time() > deadline - 10:
                break
            rows = greedy_randomized(n_rows, k, window=window, seed=seed)
            if is_valid(rows):
                scope = compute_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
        if time.time() > deadline - 10:
            break
    
    # Phase 2: Local search on the best solution found
    if best_rows is not None and time.time() < deadline - 5:
        best_rows, best_scope = local_search(best_rows, n_rows, k, deadline)
    
    # Phase 3: More aggressive randomized restarts if time remains
    if best_rows is not None and time.time() < deadline - 15:
        # Try more diverse randomizations
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            # Random window between 1 and 30
            window = random.randint(1, 30)
            seed = random.randint(0, 10**9)
            rows = greedy_randomized(n_rows, k, window=window, seed=seed)
            if is_valid(rows):
                scope = compute_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
                    # Try local search on this new best
                    if time.time() < deadline - 5:
                        best_rows, best_scope = local_search(best_rows, n_rows, k, deadline)
    
    # Fallback: pure greedy if no valid solution found
    if best_rows is None:
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        best_rows = rows
    
    return best_rows
