import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety
    
    K = k + 1  # Number of elements per row (8)
    
    # Lower bound from problem description
    min_scope = 112
    
    best_solution = None

    def find_row_randomized(used_diffs, max_scope, time_limit, rng):
        """
        Finds a row of K integers starting at 0, strictly increasing, max <= max_scope,
        such that all pairwise differences are distinct and not in used_diffs.
        Returns the row (list of ints) or None.
        """
        def backtrack(current_row, current_diffs):
            if time.time() > time_limit:
                return None
            
            if len(current_row) == K:
                return current_row[:]
            
            last_val = current_row[-1]
            remaining = K - len(current_row)
            max_val_allowed = max_scope - (remaining - 1)
            
            # Generate candidates
            candidates = list(range(last_val + 1, max_val_allowed + 1))
            rng.shuffle(candidates)
            
            for next_val in candidates:
                valid = True
                new_diffs = []
                for prev_val in current_row:
                    diff = next_val - prev_val
                    if diff in current_diffs:
                        valid = False
                        break
                    if diff in used_diffs:
                        valid = False
                        break
                    new_diffs.append(diff)
                
                if not valid:
                    continue
                
                current_row.append(next_val)
                current_diffs.update(new_diffs)
                
                result = backtrack(current_row, current_diffs)
                if result is not None:
                    return result
                
                current_row.pop()
                for d in new_diffs:
                    current_diffs.discard(d)
            
            return None
        
        return backtrack([0], set())

    # Iterate M from min_scope upwards
    # We will try to find a solution for each M.
    # If found, we return it immediately since we want to minimize M.
    
    for M in range(min_scope, 150):  # Upper bound 150 is safe (best known 134)
        
        # Time check
        if time.time() > deadline:
            break
        
        # We will try to construct n_rows rows for this M.
        # We use randomization to explore different solutions.
        # Limit attempts to fit within time budget.
        
        # Estimate time per attempt: 4 rows * small time.
        # Let's allow a few attempts per M.
        
        found_rows = None
        
        # Use a fixed number of attempts, but check time.
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            rng = random.Random(attempt + M)
            current_used_diffs = set()
            current_rows = []
            
            # Time limit for the whole set of rows for this attempt
            # We want to finish all 4 rows within a small fraction of the M-budget.
            # Let's say we give 2 seconds for the whole set of 4 rows.
            attempt_deadline = time.time() + 2.0
            
            success = True
            for r_idx in range(n_rows):
                if time.time() > attempt_deadline:
                    success = False
                    break
                
                # Allocate remaining time to this row search
                row_time_limit = attempt_deadline
                
                row = find_row_randomized(current_used_diffs, M, row_time_limit, rng)
                
                if row is None:
                    success = False
                    break
                
                # Add differences of this row to current_used_diffs
                for i in range(K):
                    for j in range(i+1, K):
                        current_used_diffs.add(row[j] - row[i])
                        
                current_rows.append(row)
            
            if success and len(current_rows) == n_rows:
                found_rows = current_rows
                break
        
        if found_rows:
            return found_rows
            
    # If we didn't find a solution, return an empty list (invalid, scores 0)
    # But we should try to return something? No, invalid is 0.
    return []
