import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin
    
    # We need n_rows rows, each with k+1 marks starting at 0
    # All pairwise differences within each row must be globally unique
    
    def solve(n_rows, k, max_scope):
        """Try to find a valid set with all marks <= max_scope"""
        if time.time() > deadline:
            return None
        
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        def backtrack(row_idx, pos_in_row):
            if time.time() > deadline:
                return False
            
            if row_idx == n_rows:
                # All rows complete
                if pos_in_row == k + 1:
                    return True
                return False
            
            if pos_in_row == k + 1:
                # This row is complete, move to next
                return backtrack(row_idx + 1, 1)
            
            row = rows[row_idx]
            last = row[-1]
            
            # Try next marks from last+1 to max_scope
            for x in range(last + 1, max_scope + 1):
                new_diffs = set()
                conflict = False
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        conflict = True
                        break
                    new_diffs.add(d)
                
                if conflict:
                    continue
                
                # Place this mark
                row.append(x)
                used_diffs.update(new_diffs)
                
                if backtrack(row_idx, pos_in_row + 1):
                    return True
                
                # Undo
                row.pop()
                used_diffs.difference_update(new_diffs)
            
            return False
        
        # Start with first row, position 1 (position 0 is already 0)
        return backtrack(0, 1)
    
    # First, get a baseline with greedy
    def greedy(n_rows, k):
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
    
    greedy_result = greedy(n_rows, k)
    best_scope = max(max(r) for r in greedy_result)
    
    # Try to improve with backtracking
    # Use a heuristic: build rows in parallel, or use smarter ordering
    
    # Try backtracking with increasing scope bounds
    # First try to beat the greedy result
    for scope in range(best_scope, 0, -1):
        if time.time() > deadline - 5:
            break
        
        # Reset
        rows = [[0] for _ in range(n_rows)]
        used_diffs = set()
        
        # Use a different strategy: interleave row construction
        # Try building all rows simultaneously, one mark at a time
        # Track how many marks each row has
        marks_count = [1] * n_rows  # each row has 0 initially
        
        def backtrack_parallel():
            if time.time() > deadline - 3:
                return False
            
            # Find which row to extend next (the one with fewest marks, or round-robin)
            # Try all possibilities
            min_marks = min(marks_count)
            candidates = [i for i in range(n_rows) if marks_count[i] == min_marks]
            
            for row_idx in candidates:
                if marks_count[row_idx] >= k + 1:
                    continue
                    
                row = rows[row_idx]
                last = row[-1]
                
                for x in range(last + 1, scope + 1):
                    new_diffs = set()
                    conflict = False
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    
                    if conflict:
                        continue
                    
                    row.append(x)
                    used_diffs.update(new_diffs)
                    marks_count[row_idx] += 1
                    
                    # Check if all rows are complete
                    if all(m >= k + 1 for m in marks_count):
                        return True
                    
                    if backtrack_parallel():
                        return True
                    
                    # Undo
                    row.pop()
                    used_diffs.difference_update(new_diffs)
                    marks_count[row_idx] -= 1
            
            return False
        
        if backtrack_parallel():
            return rows
    
    # If backtracking didn't find anything better, return greedy
    return greedy_result
