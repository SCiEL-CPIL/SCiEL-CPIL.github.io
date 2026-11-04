import time
import random
import sys

def construct(n_rows, k):
    """
    Construct a difference triangle set: n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = time.time() + 140.0  # 10s margin from 150s limit
    
    # Total number of differences needed
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Lower bound on scope: we need total_diffs_needed distinct differences in [1, M]
    # So M >= total_diffs_needed
    # But geometric constraints make it larger.
    # For n_rows=7, k=5: diffs_per_row=15, total=105. Best known=112.
    
    lower_bound = total_diffs_needed
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False
                    all_diffs.add(d)
        return True
    
    def row_diffs(row):
        """Return the set of all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def greedy_construct():
        """Deterministic greedy construction."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Start with greedy solution
    best_rows = greedy_construct()
    best_scope = get_scope(best_rows)
    
    # If greedy already achieves lower bound, return immediately
    if best_scope <= lower_bound:
        return best_rows
    
    def randomized_search_with_backtracking(max_scope):
        """
        Try to construct rows where each row's max entry is <= max_scope.
        Uses randomized backtracking.
        Returns (rows, success) where rows is the constructed list if successful.
        """
        used = set()
        rows = []
        
        def backtrack_row(row_index, current_row, current_last):
            """Recursively build the current_row starting from current_row."""
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                rows.append(list(current_row))
                if row_index + 1 >= n_rows:
                    return True
                return backtrack_row(row_index + 1, [0], 0)
            
            # Determine the range of possible next values
            # Must be > current_last, and such that max entry <= max_scope
            start_val = current_last + 1
            end_val = max_scope
            
            # For efficiency, we can limit the search space
            # If current_row has r elements, we need (k+1 - r) more elements
            # The minimum possible max entry if we add consecutive integers:
            min_possible_max = current_last + (k + 1 - len(current_row))
            if min_possible_max > max_scope:
                return False
            
            # Randomize the order of trying values
            candidates = list(range(start_val, end_val + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline:
                    return False
                
                new_diffs = {x - a for a in current_row}
                if new_diffs & used:
                    continue
                
                # Check if we can still fit remaining elements
                # After adding x, we need (k+1 - len(current_row) - 1) more elements
                remaining = k + 1 - len(current_row) - 1
                if remaining > 0:
                    # The next element must be > x, so max entry will be at least x + remaining
                    if x + remaining > max_scope:
                        # Try larger x values instead
                        continue
                
                current_row.append(x)
                used.update(new_diffs)
                
                if backtrack_row(row_index, current_row, x):
                    return True
                
                current_row.pop()
                used.difference_update(new_diffs)
            
            return False
        
        success = backtrack_row(0, [0], 0)
        if success:
            return rows, True
        return None, False
    
    def local_search(rows, max_iterations=5000):
        """
        Local search: try to improve scope by adjusting individual elements.
        """
        if not is_valid(rows):
            return rows, get_scope(rows)
        
        current_rows = [list(r) for r in rows]
        current_scope = get_scope(current_rows)
        best_local_rows = [list(r) for r in current_rows]
        best_local_scope = current_scope
        
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Pick a random row and a random position to modify
            row_idx = random.randint(0, n_rows - 1)
            row = current_rows[row_idx]
            
            # We can only modify positions 1..k (position 0 is always 0)
            pos = random.randint(1, k)
            
            old_val = row[pos]
            
            # Try changing this value
            # New value must be > row[pos-1] and < row[pos+1] (if pos < k)
            min_val = row[pos-1] + 1
            max_val = row[pos+1] - 1 if pos < k else get_scope(current_rows)
            
            if min_val >= max_val:
                continue
            
            # Try a random new value in range
            # Bias towards values that reduce scope
            if old_val == get_scope(current_rows):
                # Try to reduce the max
                new_val = random.randint(min_val, old_val - 1)
            else:
                new_val = random.randint(min_val, max_val)
            
            if new_val == old_val:
                continue
            
            # Check if this change maintains validity
            # Compute old diffs for this row
            old_row_diffs = row_diffs(row)
            
            # Temporarily modify
            row[pos] = new_val
            new_row_diffs = row_diffs(row)
            
            # Check for conflicts
            # Remove old diffs, add new diffs, check for duplicates
            all_other_diffs = set()
            for i, r in enumerate(current_rows):
                if i != row_idx:
                    all_other_diffs |= row_diffs(r)
            
            # Check if new_row_diffs has any conflict with all_other_diffs or within itself
            # (within itself is fine since it's a set)
            if new_row_diffs & all_other_diffs:
                row[pos] = old_val
                continue
            
            # Also need to check that removing old diffs and adding new diffs
            # doesn't create issues. Since we're only changing one value,
            # the key check is: new_row_diffs should not intersect with all_other_diffs
            # AND new_row_diffs should be a set (no duplicates within, which is guaranteed)
            
            # Accept the change
            new_scope = get_scope(current_rows)
            if new_scope < best_local_scope:
                best_local_scope = new_scope
                best_local_rows = [list(r) for r in current_rows]
            
            # With some probability, revert to avoid getting stuck
            if random.random() < 0.3:
                row[pos] = old_val
        
        if best_local_scope < current_scope:
            return best_local_rows, best_local_scope
        return current_rows, current_scope
    
    # Main search loop
    # Strategy: 
    # 1. Try randomized backtracking with decreasing max_scope
    # 2. Use local search to refine solutions
    
    # First, try to find a solution with scope = best_scope - 1, then -2, etc.
    # But also do randomized restarts with local search
    
    scope_to_try = best_scope - 1
    
    while time.time() < deadline:
        # Try backtracking with current target scope
        if scope_to_try >= lower_bound:
            rows, success = randomized_search_with_backtracking(scope_to_try)
            if success:
                new_scope = get_scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = rows
                    if best_scope <= lower_bound:
                        break
                scope_to_try -= 1
                continue
        
        # Do local search on current best
        improved_rows, improved_scope = local_search(best_rows, max_iterations=2000)
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
            if best_scope <= lower_bound:
                break
            # Reset scope_to_try to try to improve further
            scope_to_try = best_scope - 1
        
        # Also try randomized construction with some randomness
        # Then local search
        if random.random() < 0.5:
            # Randomized greedy
            used = set()
            rows = []
            success = True
            for _ in range(n_rows):
                row = [0]
                last = 0
                while len(row) < k + 1:
                    found = False
                    # Try values in a randomized window
                    window_start = last + 1
                    window_end = min(last + 50, best_scope)
                    candidates = list(range(window_start, window_end + 1))
                    random.shuffle(candidates)
                    for x in candidates:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            last = x
                            found = True
                            break
                    if not found:
                        success = False
                        break
                if not success:
                    break
                rows.append(row)
            
            if success and len(rows) == n_rows:
                scope = get_scope(rows)
                if scope < best_scope:
                    # Refine with local search
                    refined_rows, refined_scope = local_search(rows, max_iterations=3000)
                    if refined_scope < best_scope:
                        best_scope = refined_scope
                        best_rows = refined_rows
                        if best_scope <= lower_bound:
                            break
    
    return best_rows
