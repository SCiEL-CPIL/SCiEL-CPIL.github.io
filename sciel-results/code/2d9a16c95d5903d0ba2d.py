import time
import random
import math
from itertools import combinations

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Total number of differences needed
    num_diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Lower bound from counting: need total_diffs distinct values in [1, max]
    # So max >= total_diffs, but also each row's max >= k*(k+1)/2 roughly
    lower_bound = max(total_diffs, k * (k + 1) // 2)
    
    # Greedy construction: build rows one by one, choosing smallest possible next element
    def greedy_build(n_rows, k, randomize=False):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                if randomize and random.random() < 0.1:
                    x += random.randint(1, 3)
                else:
                    x += 1
                new_diffs = set()
                for a in row:
                    d = x - a
                    if d in used:
                        break
                    new_diffs.add(d)
                else:
                    row.append(x)
                    used.update(new_diffs)
        return rows, max(max(r) for r in rows)
    
    # Local search: try to reduce the maximum by adjusting elements
    def local_search(rows, k, n_rows, max_iter=5000):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        
        current_max = max(max(r) for r in rows)
        best_rows = [r[:] for r in rows]
        best_max = current_max
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random row and try to reduce a random element
            ri = random.randint(0, n_rows - 1)
            # Try to reduce the largest element of this row
            row = rows[ri]
            
            # Try reducing the last element
            old_val = row[-1]
            # Find the smallest value that works
            found = False
            for new_val in range(row[-2] + 1, old_val):
                # Check if new_val works
                new_diffs = set()
                valid = True
                for a in row[:-1]:
                    d = new_val - a
                    if d in used:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    # Remove old diffs involving old_val
                    for a in row[:-1]:
                        used.discard(old_val - a)
                    # Add new diffs
                    used.update(new_diffs)
                    row[-1] = new_val
                    found = True
                    break
            
            if found:
                new_max = max(max(r) for r in rows)
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [r[:] for r in rows]
            else:
                # Try a different element - pick a random position
                if len(row) > 2:
                    pi = random.randint(1, len(row) - 2)
                    old_val = row[pi]
                    for new_val in range(row[pi-1] + 1, old_val):
                        new_diffs = set()
                        valid = True
                        for j, a in enumerate(row):
                            if j == pi:
                                continue
                            if j < pi:
                                d = new_val - a
                            else:
                                d = a - new_val
                            if d in used:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            # Remove old diffs involving old_val at position pi
                            for j, a in enumerate(row):
                                if j == pi:
                                    continue
                                if j < pi:
                                    used.discard(old_val - a)
                                else:
                                    used.discard(a - old_val)
                            # Add new diffs
                            used.update(new_diffs)
                            row[pi] = new_val
                            found = True
                            break
                    if found:
                        new_max = max(max(r) for r in rows)
                        if new_max < best_max:
                            best_max = new_max
                            best_rows = [r[:] for r in rows]
        
        return best_rows, best_max
    
    # Backtracking search with pruning
    def backtrack_search(n_rows, k, max_limit):
        """Try to find a valid configuration with max entry <= max_limit."""
        used = set()
        rows = []
        
        def dfs(row_idx, pos_idx, row):
            if time.time() > deadline:
                return False
            
            if row_idx >= n_rows:
                return True
            
            if pos_idx > k:
                rows.append(row[:])
                return dfs(row_idx + 1, 0, [0])
            
            if pos_idx == 0:
                return dfs(row_idx, 1, [0])
            
            # Start from previous element + 1
            start = row[-1] + 1 if row else 1
            
            for x in range(start, max_limit + 1):
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    row.append(x)
                    used.update(new_diffs)
                    
                    if dfs(row_idx, pos_idx + 1, row):
                        return True
                    
                    row.pop()
                    used.difference_update(new_diffs)
            
            return False
        
        return dfs(0, 0, [0]), rows
    
    # First, do a greedy build to get an initial solution
    best_rows, best_max = greedy_build(n_rows, k)
    
    # Try to improve with local search
    improved_rows, improved_max = local_search([r[:] for r in best_rows], k, n_rows)
    if improved_max < best_max:
        best_rows = improved_rows
        best_max = improved_max
    
    # Try backtracking with decreasing max limits
    # Start from current best and try to reduce
    for target_max in range(best_max - 1, lower_bound - 1, -1):
        if time.time() > deadline - 5:
            break
        # Try backtracking with this limit
        success, rows = backtrack_search(n_rows, k, target_max)
        if success:
            best_rows = rows
            best_max = target_max
            # Continue trying to go lower
            continue
    
    # Final local search polish
    if time.time() < deadline - 5:
        final_rows, final_max = local_search([r[:] for r in best_rows], k, n_rows, max_iter=10000)
        if final_max < best_max:
            best_rows = final_rows
            best_max = final_max
    
    return best_rows
