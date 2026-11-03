import time
import random
import sys
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs
    
    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 2. Local improvement to reduce scope
    best_solution = _improve_scope(best_solution, n_rows, k, deadline)
    best_scope = max(max(row) for row in best_solution)
    
    # 3. Try to find better solutions with randomized backtracking
    # for target scopes below current best
    while time.time() < deadline - 5.0:
        target_scope = best_scope - 1
        if target_scope < lower_bound:
            break
        
        # Try multiple random restarts for this target scope
        found = False
        for attempt in range(30):
            if time.time() > deadline - 5.0:
                break
            
            solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                new_scope = max(max(row) for row in solution)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_solution = solution
                    found = True
                    # Try to improve further
                    improved = _improve_scope(best_solution, n_rows, k, deadline)
                    new_scope = max(max(row) for row in improved)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = improved
                    if best_scope <= lower_bound:
                        return best_solution
                    break
        
        if not found and target_scope <= lower_bound + 10:
            # If we can't find anything near the lower bound, 
            # try a slightly larger target to make progress
            target_scope = best_scope - 1
            if target_scope < lower_bound:
                break
    
    return best_solution

def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search to build n_rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, and max element <= max_scope.
    Uses a boolean array for fast difference lookup.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    # Use a bytearray for fast membership testing
    used_diffs = bytearray(max_scope + 1)  # index 0 unused, indices 1..max_scope
    rows = []
    
    def build_row(row_idx, pos, current_mark, row_so_far):
        if time.time() > deadline:
            return None
            
        if pos == k + 1:
            return list(row_so_far)
        
        # Find valid candidates efficiently
        valid_candidates = []
        for x in range(current_mark + 1, max_scope + 1):
            # Check if any difference conflicts
            conflict = False
            for a in row_so_far:
                d = x - a
                if d > max_scope or used_diffs[d]:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
        
        if not valid_candidates:
            return None
        
        # Randomize the order of candidates to escape local minima
        # But bias towards smaller values to keep scope small
        n_smallest = min(15, len(valid_candidates))
        smallest = valid_candidates[:n_smallest]
        
        # With some probability, also consider larger candidates
        if len(valid_candidates) > n_smallest and rng.random() < 0.4:
            extra = rng.sample(valid_candidates[n_smallest:], min(8, len(valid_candidates) - n_smallest))
            candidates_to_try = smallest + extra
        else:
            candidates_to_try = smallest
        
        rng.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            # Compute new differences
            new_diffs = [x - a for a in row_so_far]
            
            # Check if we can actually complete the row with remaining positions
            # Simple heuristic: the largest possible value is max_scope,
            # so we need room for (k+1 - pos - 1) more elements
            # Each needs at least 1 more than the previous, so we need
            # x + (k+1 - pos - 1) <= max_scope
            if x + (k + 1 - pos - 1) > max_scope:
                continue
            
            # Mark differences as used
            for d in new_diffs:
                used_diffs[d] = 1
            
            res = build_row(row_idx, pos + 1, x, row_so_far + [x])
            if res is not None:
                return res
            
            # Unmark differences
            for d in new_diffs:
                used_diffs[d] = 0
        
        return None
    
    for i in range(n_rows):
        row = build_row(i, 1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        # Update used_diffs for the completed row
        for j in range(len(row)):
            for jj in range(j + 1, len(row)):
                d = row[jj] - row[j]
                if d <= max_scope:
                    used_diffs[d] = 1
    
    return rows

def _improve_scope(solution, n_rows, k, deadline):
    """
    Try to reduce the scope of a valid solution by local search.
    We try to shift elements down to reduce the maximum value.
    Uses a boolean array for fast difference tracking.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    max_possible_scope = best_scope
    
    # Collect all used differences using a bytearray
    used_diffs = bytearray(max_possible_scope + 1)
    for row in best:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d <= max_possible_scope:
                    used_diffs[d] = 1
    
    # Try to reduce scope by shifting elements
    # Strategy: for each row, try to shift elements down
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(len(row) - 1, 0, -1):  # from end to start
                # Try to shift this element down
                original = row[pos]
                for new_val in range(row[pos - 1] + 1, original):
                    # Check if new value creates conflicts
                    new_diffs = []
                    for i in range(pos):
                        d = new_val - row[i]
                        if d <= 0 or d > max_possible_scope:
                            new_diffs = None
                            break
                        new_diffs.append(d)
                    if new_diffs is None:
                        continue
                    
                    # Also need to check diffs with elements after pos
                    for j in range(pos + 1, len(row)):
                        d = row[j] - new_val
                        if d <= 0 or d > max_possible_scope:
                            new_diffs = None
                            break
                        new_diffs.append(d)
                    if new_diffs is None:
                        continue
                    
                    # Remove old diffs from tracking
                    old_diffs = []
                    for i in range(pos):
                        d = original - row[i]
                        if d <= max_possible_scope:
                            old_diffs.append(d)
                    for j in range(pos + 1, len(row)):
                        d = row[j] - original
                        if d <= max_possible_scope:
                            old_diffs.append(d)
                    
                    # Temporarily unmark old diffs
                    for d in old_diffs:
                        used_diffs[d] = 0
                    
                    # Check if new_diffs conflict with remaining used_diffs
                    conflict = False
                    for d in new_diffs:
                        if used_diffs[d]:
                            conflict = True
                            break
                    
                    if conflict:
                        # Re-mark old diffs
                        for d in old_diffs:
                            used_diffs[d] = 1
                        continue
                    
                    # Apply the change
                    for d in new_diffs:
                        used_diffs[d] = 1
                    row[pos] = new_val
                    improved = True
                    break
    
    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        # Revert if no improvement
        return [list(row) for row in solution]
