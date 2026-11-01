import time
import random
import sys

def construct(n_rows, k):
    # Calculate a lower bound for M
    total_diffs_needed = n_rows * k * (k + 1) // 2
    M_lower = int((2 * total_diffs_needed) ** 0.5)
    while M_lower * (M_lower + 1) // 2 < total_diffs_needed:
        M_lower += 1
    
    deadline = time.time() + 115.0
    best_solution = None
    best_M = float('inf')
    
    random.seed(42)
    
    # We will try M from M_lower upwards
    # We limit the number of restarts per M to avoid getting stuck on an impossible M
    # If M is impossible, we waste time. But M_lower is a tight bound.
    # We'll allow a few seconds per M.
    
    current_M = M_lower
    max_M = M_lower + 50 # Safety cap
    
    while current_M <= max_M and time.time() < deadline:
        # Try to find a solution with max entry <= current_M
        # Number of restarts depends on time left
        time_left = deadline - time.time()
        if time_left < 1.0:
            break
            
        # Allocate time for this M
        # If this is the first M, give it more time?
        # Or just fixed restarts?
        # Let's say 20 restarts per M, or until 5 seconds pass.
        restarts = 0
        m_start_time = time.time()
        
        while restarts < 50 and (time.time() - m_start_time) < 5.0 and time.time() < deadline:
            rows = try_construct(n_rows, k, current_M)
            if rows is not None:
                # Valid solution found
                if current_M < best_M:
                    best_M = current_M
                    best_solution = rows
                # Since we iterate M upwards, the first success is the best M we've seen.
                # We can return immediately if we are confident.
                # However, to be safe against "missed" solutions at lower M (if we skipped), 
                # but we didn't skip, we started at lower bound.
                # So if we find one at M, it's optimal for the search so far.
                # We can return it.
                return rows
            restarts += 1
        
        # If we didn't find a solution, increment M
        current_M += 1
        
    # If we didn't return inside the loop, we might have a best_solution from a previous iteration?
    # No, the logic above returns immediately on success.
    # So if we exit the loop, it means we exhausted M or time without finding a solution.
    # In that case, we should return the greedy fallback.
    
    # Fallback: Greedy construction
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def try_construct(n_rows, k, M):
    """
    Attempts to construct n_rows rows with max entry <= M.
    Returns list of rows or None.
    """
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = find_row(used_diffs, M, k)
        if row is None:
            return None
        rows.append(row)
        # Update used_diffs
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used_diffs.add(row[j] - row[i])
                
    return rows

def find_row(used_diffs, M, k):
    """
    Finds a row of k+1 integers starting at 0, strictly increasing, max <= M,
    such that all differences are not in used_diffs and distinct within the row.
    """
    row = [0]
    current_diffs = set()
    
    def dfs():
        if len(row) == k + 1:
            return True
        
        last = row[-1]
        candidates = []
        
        # Generate valid candidates
        for x in range(last + 1, M + 1):
            valid = True
            # Check against global used diffs and current row diffs
            for a in row:
                d = x - a
                if d in used_diffs or d in current_diffs:
                    valid = False
                    break
            if valid:
                candidates.append(x)
                
        if not candidates:
            return False
            
        # Heuristic: 
        # 80% chance: pick the smallest candidate (tight packing)
        # 20% chance: pick a random candidate (exploration)
        if random.random() < 0.8:
            x = candidates[0]
        else:
            x = random.choice(candidates)
            
        # Add x
        row.append(x)
        # Update current_diffs
        for a in row[:-1]:
            current_diffs.add(x - a)
            
        if dfs():
            return True
            
        # Backtrack
        for a in row[:-1]:
            current_diffs.discard(x - a)
        row.pop()
        
        # If the heuristic choice failed, we should try other candidates?
        # The current DFS only tries ONE candidate per node because of the heuristic selection.
        # This is NOT a full DFS. It's a greedy randomized search.
        # To make it a proper search, we need to iterate through candidates.
        
        # Let's refactor to a proper DFS with ordering.
        return False

    # The above find_row is flawed because it doesn't backtrack over candidate choices properly.
    # It picks one candidate, and if that fails, it returns False, but doesn't try the next candidate.
    # We need to loop through candidates.
    
    def dfs_corrected():
        if len(row) == k + 1:
            return True
        
        last = row[-1]
        candidates = []
        
        for x in range(last + 1, M + 1):
            valid = True
            for a in row:
                d = x - a
                if d in used_diffs or d in current_diffs:
                    valid = False
                    break
            if valid:
                candidates.append(x)
                
        if not candidates:
            return False
            
        # Order candidates: 
        # We want to try tight packing first, but also allow exploration.
        # Let's shuffle a subset or use a weighted pick?
        # For backtracking, we must try all candidates if the first one fails.
        # To improve efficiency, we order them.
        # Sort by x (ascending) is good for packing.
        # But pure ascending might get stuck.
        # Let's shuffle the candidates, but keep the smallest few at the front?
        # Or just shuffle entirely?
        # Shuffling entirely is standard for randomized DFS.
        random.shuffle(candidates)
        
        for x in candidates:
            row.append(x)
            new_diffs = set()
            for a in row[:-1]:
                d = x - a
                new_diffs.add(d)
                current_diffs.add(d)
                
            if dfs_corrected():
                return True
                
            # Backtrack
            current_diffs.difference_update(new_diffs)
            row.pop()
            
        return False
        
    if dfs_corrected():
        return row
    return None
