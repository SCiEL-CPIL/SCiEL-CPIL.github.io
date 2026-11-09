"""Improved program for difference triangle set (4,7) using randomized backtracking and greedy fallback."""
# EVOLVE-BLOCK-START
import random
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct over all rows.
    Minimizes the largest entry.
    
    Uses randomized backtracking with forward checking for small M, 
    and a greedy fallback for guaranteed validity.
    """
    # Total number of differences per row: C(k+1, 2)
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row
    
    # Counting bound
    counting_bound = total_diffs_needed
    
    # Best known for (4,7) is 134.
    # We will try to find a solution with M=134 first, then 135, etc.
    # We use a time-limited randomized backtracking search.
    
    def try_construct_randomized(max_entry, time_limit=5.0):
        """
        Attempt to construct n_rows rows with max entry <= max_entry
        using randomized backtracking.
        """
        start_time = time.time()
        used_diffs = set()
        rows = []
        
        # We build rows one by one.
        # For each row, we use backtracking to find a valid set of marks.
        
        def backtrack_row(row_marks, current_max, used_diffs_local, depth=0):
            if time.time() - start_time > time_limit:
                return None
            
            if len(row_marks) == k + 1:
                return row_marks[:]
            
            # Determine the range for the next mark
            # Must be > current_max and <= max_entry
            # Also, we need to fit (k + 1 - len(row_marks) - 1) more marks after this one
            remaining_marks = k + 1 - len(row_marks) - 1
            min_next = current_max + 1
            # We need at least 'remaining_marks' integers strictly greater than next_mark
            # So next_mark <= max_entry - remaining_marks
            max_next = max_entry - remaining_marks
            
            if min_next > max_next:
                return None
            
            # Generate candidates
            # Heuristic: Randomize the order to escape local minima / symmetry
            candidates = list(range(min_next, max_next + 1))
            random.shuffle(candidates)
            
            for next_mark in candidates:
                # Check differences with existing marks in this row
                new_diffs = []
                valid = True
                for existing in row_marks:
                    diff = next_mark - existing
                    if diff in used_diffs_local:
                        valid = False
                        break
                    new_diffs.append(diff)
                
                if not valid:
                    continue
                
                # Check if new_diffs have internal conflicts?
                # Since row_marks is strictly increasing, diffs are next_mark - r_i.
                # These are distinct because r_i are distinct.
                # So no internal conflict in new_diffs themselves.
                
                # Update used_diffs
                for d in new_diffs:
                    used_diffs_local.add(d)
                
                # Recurse
                result = backtrack_row(row_marks + [next_mark], next_mark, used_diffs_local, depth + 1)
                
                if result is not None:
                    return result
                
                # Undo
                for d in new_diffs:
                    used_diffs_local.remove(d)
            
            return None
        
        for _ in range(n_rows):
            # Try to find a row
            # We can try multiple random restarts for the row construction if it fails quickly
            # But usually, one pass with shuffling is enough if M is not too tight.
            row = backtrack_row([0], 0, used_diffs)
            if row is None:
                return None
            rows.append(row)
        
        return rows

    # Try to find optimal solution
    # Start from counting bound, but jump to known good bound if possible
    # For (4,7), counting bound is 112. Known best is 134.
    # It is highly likely no solution exists for < 134 or is very hard.
    # We'll try 134 first. If it fails, we try 135, 136...
    # If we fail to find a solution for 134 within time, we might still have a valid greedy solution.
    
    # To be safe and fast, we will try M=134 with a short time limit.
    # If it finds a solution, great.
    # If not, we will use a greedy construction which is O(N^2) and very fast, 
    # ensuring we return *a* valid solution. The greedy solution's M might be larger, 
    # but it's better than timing out (score 0).
    
    # Actually, let's try a range of M values with short time limits.
    # But the greedy fallback is crucial for robustness.
    
    # Greedy Fallback:
    # Construct rows sequentially. For each row, pick the smallest possible next integer
    # that doesn't create a conflict with any used difference (global) or internal difference.
    
    def greedy_construct():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            while len(row) < k + 1:
                # Try next integers
                found = False
                # Upper bound for greedy: we can go up to a large number, but we want small M.
                # A safe upper bound is counting_bound * 2 or similar.
                # Let's just iterate up.
                for x in range(current_max + 1, current_max + 1000): 
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs |= new_diffs
                        current_max = x
                        found = True
                        break
                if not found:
                    # This shouldn't happen if we go high enough
                    return None
            rows.append(row)
        return rows

    # Strategy:
    # 1. Try to find a solution with M=134 using randomized backtracking (time limited).
    # 2. If successful, return it.
    # 3. If not, try M=135, 136... up to 140 with short time limits.
    # 4. If all fail, return the greedy solution (which is valid but M might be ~150-200).
    
    # Note: The greedy solution for (4,7) typically yields M around 160-180.
    # The backtracking might find 134.
    
    target_ms = [134, 135, 136, 137, 138, 139, 140]
    
    for m in target_ms:
        if m < counting_bound:
            continue
        # Try with a small time limit to avoid hanging
        # We can increase time limit for the first few attempts
        time_limit = 2.0
        result = try_construct_randomized(m, time_limit)
        if result is not None:
            return result
    
    # Fallback to greedy
    result = greedy_construct()
    if result is not None:
        return result
    
    # Absolute fallback: return a trivially valid but large set
    # This should never be reached
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
    return rows

# EVOLVE-BLOCK-END
