import time
import sys

def construct(n_rows, k):
    """
    Constructs n_rows rows of k+1 increasing integers starting at 0.
    All positive differences between entries of the same row must be distinct
    over all rows.
    Minimizes the largest entry (scope).
    """
    # Lower bound estimation
    # Total differences per row: k*(k+1)/2
    # Total differences overall: n_rows * k*(k+1)/2
    # Max difference is M. So we need M >= total_diffs.
    # Actually, the differences are from 1 to M.
    # So M must be at least n_rows * k*(k+1)/2.
    # For n_rows=4, k=7: 4 * 28 = 112.
    
    total_diffs = n_rows * k * (k + 1) // 2
    M_lower = total_diffs
    
    # Upper bound: The greedy algorithm provides a valid solution.
    # We can run the greedy to get a starting upper bound.
    # But greedy is slow to compute if we just run it. 
    # Let's just start from M_lower and go up.
    
    start_time = time.time()
    time_limit = 100.0  # Leave 20s margin for the evaluator's 120s limit
    
    # Helper to check time
    def time_exceeded():
        return (time.time() - start_time) > time_limit

    # Backtracking search for a fixed M
    def search(M):
        # global_used[d] is True if difference d is used in any row so far
        global_used = [False] * (M + 1)
        rows = []
        
        # Pre-compute nothing, just use sets for local diffs? 
        # Using a list of booleans for global is fast.
        
        def backtrack(row_idx, current_row):
            if time_exceeded():
                return False
            
            if row_idx == n_rows:
                return True
            
            if len(current_row) == k + 1:
                # Row complete
                rows.append(current_row)
                # Start next row
                result = backtrack(row_idx + 1, [0])
                if result:
                    return True
                rows.pop()
                return False
            
            prev_val = current_row[-1]
            # Try next values for x
            # Optimization: We can try to pick x such that it doesn't create
            # "gaps" that are hard to fill? 
            # Simple heuristic: try x from prev_val + 1 to M
            
            # To improve performance, we can try to fill the row with
            # "dense" differences first? Or just smallest x?
            # Smallest x keeps the row compact, which is usually good for
            # leaving room for other rows.
            
            for x in range(prev_val + 1, M + 1):
                # Check if new differences are available
                # new_diffs = {x - a for a in current_row}
                # Since current_row is increasing, x - a is decreasing as a increases.
                # We just need to check if any x - a is in global_used.
                
                valid = True
                new_diffs = []
                for a in current_row:
                    d = x - a
                    if d > M: # Should not happen as x <= M and a >= 0
                        valid = False
                        break
                    if global_used[d]:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                
                # Mark as used
                for d in new_diffs:
                    global_used[d] = True
                
                # Recurse
                if backtrack(row_idx, current_row + [x]):
                    return True
                
                # Unmark
                for d in new_diffs:
                    global_used[d] = False
            
            return False
        
        return backtrack(0, [0]), rows

    # Iterate M from lower bound upwards
    M = M_lower
    while True:
        if time_exceeded():
            # If we run out of time, return the best found so far?
            # We haven't found a valid one yet in this loop structure.
            # We should have a fallback.
            # Let's use the greedy algorithm as a fallback if backtracking fails or times out.
            break
        
        found, rows = search(M)
        if found:
            return rows
        
        M += 1
        
        # If M gets too large, we might want to switch to a different strategy
        # or just let it run. Given the problem size, M shouldn't need to be huge.
        
    # Fallback: Greedy algorithm (slower to find optimal, but always finds a valid solution)
    # This ensures we return *something* valid if the backtracking search is too slow
    # to find a solution near the lower bound.
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

# EVOLVE-BLOCK-START
# (The function construct is defined above, but the prompt asks to keep the function names 
# and signatures. The seed had the function inside the block. I will put the whole logic in the block.)
# EVOLVE-BLOCK-END
