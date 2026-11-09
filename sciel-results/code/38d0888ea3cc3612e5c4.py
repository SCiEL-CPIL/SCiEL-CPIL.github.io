import time
import sys

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 100.0 # Safety margin
    
    # 1. Get a baseline greedy solution
    # (Reuse the logic from the seed or write a simple one)
    baseline_rows = greedy_construct(n_rows, k)
    best_rows = baseline_rows
    best_scope = max(max(r) for r in best_rows) if best_rows else 0
    
    # 2. Determine lower bound for M
    # n_rows * C(k+1, 2) <= M * (M+1) / 2
    num_diffs = n_rows * (k * (k + 1) // 2)
    # M^2 + M - 2*num_diffs >= 0
    # M = (-1 + sqrt(1 + 8*num_diffs)) / 2
    import math
    lb = int(math.ceil((-1 + math.sqrt(1 + 8 * num_diffs)) / 2))
    
    # 3. Binary search or linear search for M
    # Since M is likely close to lb, linear search from lb is good.
    # But we need to stop if time runs out.
    
    for M in range(lb, lb + 50): # Search up to lb+50
        if time.time() - start_time > time_limit - 10:
            break
            
        # Check if M is feasible
        solution = backtrack_solve(n_rows, k, M, time_limit - (time.time() - start_time))
        if solution:
            return solution
            
    return best_rows

def greedy_construct(n_rows, k):
    # Simple greedy to get a valid solution quickly
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

def backtrack_solve(n_rows, k, M, time_budget):
    # Uses bitmask for used differences
    # Max diff is M
    # Total bits needed M+1
    # Python int is arbitrary precision
    
    start_time = time.time()
    
    def can_place(row, x, used_mask):
        # Check if adding x to row creates conflicts
        # New diffs: x - y for y in row
        # We need to check if any of these bits are set in used_mask
        # Also, they must be distinct? 
        # If y1 != y2, x-y1 != x-y2. So they are distinct from each other.
        # We just need to check against used_mask.
        new_mask = 0
        for y in row:
            d = x - y
            if d > M: return False # Should not happen if x<=M and y>=0
            bit = 1 << d
            if used_mask & bit:
                return False
            new_mask |= bit
        return new_mask
    
    # State:
    # rows: list of completed rows
    # current_row: list of elements in current row
    # used_mask: int bitmask of used differences
    
    rows = []
    
    def backtrack(row_idx, current_row, used_mask):
        if time.time() - start_time > time_budget:
            return None # Timeout
        
        if row_idx == n_rows:
            return rows + [current_row]
            
        if len(current_row) == k + 1:
            # Move to next row
            rows.append(current_row)
            result = backtrack(row_idx + 1, [0], used_mask)
            if result:
                return result
            rows.pop()
            return None
            
        # Determine next element x
        last = current_row[-1]
        start_x = last + 1
        
        # Symmetry breaking:
        # If this is the 2nd element (index 1) of the row, 
        # and we have previous rows, x must be >= previous_row[1]
        if len(current_row) == 1 and row_idx > 0:
            prev_row = rows[-1]
            if len(prev_row) > 1:
                start_x = max(start_x, prev_row[1])
        
        # Heuristic: Try smaller x first? 
        # Or try to spread out?
        # Let's try smaller x first to keep numbers small.
        
        for x in range(start_x, M + 1):
            new_mask = can_place(current_row, x, used_mask)
            if new_mask is not False:
                # Place x
                current_row.append(x)
                result = backtrack(row_idx, current_row, used_mask | new_mask)
                if result:
                    return result
                current_row.pop()
                
        return None
        
    # Initial call
    return backtrack(0, [0], 0)
