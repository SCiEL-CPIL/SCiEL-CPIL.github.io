import time
import random
import sys

def construct(n_rows, k):
    # Calculate a lower bound for M
    # Total differences needed: n_rows * k * (k+1) / 2
    # Max differences available up to M: M * (M+1) / 2
    # So M * (M+1) >= n_rows * k * (k+1)
    total_diffs_needed = n_rows * k * (k + 1) // 2
    # Solve M(M+1)/2 >= total_diffs_needed
    # M approx sqrt(2 * total_diffs_needed)
    M_lower = int((2 * total_diffs_needed) ** 0.5)
    # Adjust M_lower to be safe
    while M_lower * (M_lower + 1) // 2 < total_diffs_needed:
        M_lower += 1
    
    M = M_lower
    deadline = time.time() + 110.0
    random.seed(42) # Fixed seed for reproducibility in first attempt, then random
    
    while time.time() < deadline:
        used_diffs = set()
        rows = []
        success = True
        
        for _ in range(n_rows):
            row = find_row(used_diffs, M, k)
            if row is None:
                success = False
                break
            rows.append(row)
            # Update used_diffs
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        if success:
            return rows
        
        # If we failed, try again with a new randomization
        # If we are close to deadline, increase M
        if time.time() > deadline - 10.0:
            M += 1
            # Reset random seed to explore new space
            random.seed()
        else:
            # Just retry, find_row uses randomization internally
            pass

    # If we run out of time, return the best found so far? 
    # The problem requires a valid set. If we didn't find one, we should return a valid one if possible.
    # But if we didn't find one, we might have to return an invalid one or crash.
    # The prompt says "Valid when...". If invalid, score 0.
    # So we must ensure we return a valid set if possible.
    # If we ran out of time, we should probably return the greedy solution as a fallback.
    
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

def find_row(used_diffs, M, k):
    """
    Finds a row of k+1 integers starting at 0, strictly increasing, max <= M,
    such that all differences are not in used_diffs.
    Uses randomized backtracking.
    """
    row = [0]
    
    def dfs():
        if len(row) == k + 1:
            return True
        
        last = row[-1]
        # Candidates for next element: last + 1 to M
        # We can optimize by checking validity first
        candidates = []
        for x in range(last + 1, M + 1):
            # Check if differences to all previous elements are unused
            valid = True
            for a in row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
            if valid:
                candidates.append(x)
        
        if not candidates:
            return False
        
        # Randomize candidates to explore different branches
        random.shuffle(candidates)
        
        for x in candidates:
            row.append(x)
            # We don't update used_diffs here because we are just finding ONE row
            # The caller will update the global used_diffs.
            # However, we must ensure that differences *within this row* are distinct.
            # The check `d in used_diffs` ensures they are distinct from previous rows.
            # But what about differences within the current row?
            # If x1 - x0 = d and x2 - x1 = d, then x2 - x0 = 2d.
            # The check `x - a in used_diffs` only checks against GLOBAL used_diffs.
            # It does NOT check against differences already formed within the CURRENT row.
            # We need to track differences within the current row.
            
            # Let's refine: 
            # We need `x - a` to not be in `used_diffs` AND not be in `current_row_diffs`.
            
            # Let's restart the DFS logic with this correction.
            # Actually, if we just check against `used_diffs`, we might pick x such that x - a is already in the row.
            # Example: row=[0, 3]. used_diffs={}. 
            # Try x=6. 6-3=3 (in row? no, 3 is a mark, 3 is a diff? 3-0=3).
            # If we pick x=6, diffs are 6, 3. 3 is already a diff (3-0).
            # So we need to track current row diffs.
            
            # Let's modify the inner loop to track current row diffs.
            
            # This approach is getting complicated inside the closure.
            # Let's rewrite find_row properly.
            
            pass
        
        row.pop()
        return False

    # Rewrite find_row to be correct
    def find_row_corrected(used_diffs, M, k):
        row = [0]
        current_diffs = set()
        
        def dfs():
            if len(row) == k + 1:
                return True
            
            last = row[-1]
            candidates = []
            for x in range(last + 1, M + 1):
                valid = True
                new_diffs = set()
                for a in row:
                    d = x - a
                    if d in used_diffs or d in current_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    candidates.append((x, new_diffs))
            
            if not candidates:
                return False
            
            random.shuffle(candidates)
            
            for x, new_diffs in candidates:
                row.append(x)
                # Add new_diffs to current_diffs
                current_diffs.update(new_diffs)
                
                if dfs():
                    return True
                
                # Backtrack
                current_diffs.difference_update(new_diffs)
                row.pop()
            
            return False
        
        if dfs():
            return row
        return None

    return find_row_corrected(used_diffs, M, k)

# Re-define construct to use the corrected find_row
def construct(n_rows, k):
    total_diffs_needed = n_rows * k * (k + 1) // 2
    M_lower = int((2 * total_diffs_needed) ** 0.5)
    while M_lower * (M_lower + 1) // 2 < total_diffs_needed:
        M_lower += 1
    
    M = M_lower
    deadline = time.time() + 110.0
    random.seed(42)
    
    while time.time() < deadline:
        used_diffs = set()
        rows = []
        success = True
        
        for _ in range(n_rows):
            row = find_row(used_diffs, M, k)
            if row is None:
                success = False
                break
            rows.append(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        if success:
            return rows
        
        if time.time() > deadline - 10.0:
            M += 1
            random.seed()
        else:
            pass

    # Fallback greedy
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
