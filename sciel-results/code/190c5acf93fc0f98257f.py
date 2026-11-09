import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Problem parameters
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    # The counting bound suggests M >= 105. Best known is 112.
    # We will search for solutions with M starting from a lower bound up to a limit.
    
    # We want to find the minimum M such that a valid set exists.
    # We can use a backtracking algorithm that tries to fill rows.
    # To speed up, we can try to construct rows one by one, ensuring global uniqueness of differences.
    
    # State:
    # used_diffs: set of all differences used so far
    # rows: list of rows constructed so far
    
    # To make it efficient, we can precompute valid rows for a given M?
    # No, M is small (around 105-112), but the number of combinations is huge.
    # However, k is small (5), so row length is 6.
    # Number of possible rows with max entry M is C(M, k) roughly.
    # For M=112, k=5, C(112, 5) ~ 112^5 / 120 ~ 10^8. Too many to precompute.
    
    # Alternative: Backtracking on the entire set.
    # We fill rows one by one.
    # For each row, we generate candidate rows (subsets of size k+1 from {0..M} starting with 0) 
    # that do not conflict with used_diffs.
    # To avoid generating all C(M, k) candidates, we can build the row incrementally.
    
    # Let's implement a randomized backtracking search.
    # We will try to find a solution for a given M.
    # We will start M from a lower bound and increase it.
    # Lower bound: total_diffs. For (7,5), total_diffs = 7 * 15 = 105.
    # So M must be at least 105.
    # We will try M = 105, 106, ..., up to some limit (e.g., 120).
    # If we find a solution for M, we return it.
    # If we time out, we return the best valid solution found (or a greedy one if none valid).
    
    # Actually, the task asks to minimize the largest entry.
    # So we should try to find the smallest M.
    # We can run a search for M=105, if it fails, try M=106, etc.
    # But backtracking might be slow.
    # We can use a heuristic: randomize the order of candidates.
    
    # Let's define a function to check if a row is valid given used_diffs.
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Backtracking search for a fixed M
    def solve_for_M(M, max_time=None):
        if max_time is None:
            max_time = deadline - time.time()
        
        used_diffs = set()
        rows = []
        
        # We will use a recursive backtracking function.
        # To avoid deep recursion limits or stack issues, we can use an iterative approach or just recursion with small depth (7 rows).
        # Depth is n_rows, which is 7. This is safe.
        
        def backtrack(row_idx):
            if time.time() > max_time:
                return False
            
            if row_idx == n_rows:
                return True
            
            # Generate candidate rows for this position
            # A row is a list of k+1 integers: 0 = r0 < r1 < ... < rk <= M
            # We can generate them by choosing k elements from 1..M.
            # To speed up, we can iterate through combinations.
            # But C(M, k) is large.
            # Heuristic: We can prioritize rows with smaller max element? Or randomize?
            # Let's generate random valid rows and check.
            # Or, better: iterate through combinations in a randomized order?
            # Generating all combinations is expensive.
            # Let's try a different approach:
            # Build the row element by element.
            
            # State for current row construction:
            # current_row: list
            # current_diffs: set of diffs within current_row
            # We need to ensure current_diffs is disjoint from used_diffs.
            
            # Let's use a helper to extend a partial row.
            
            def extend_row(current_row, current_diffs, next_idx):
                if time.time() > max_time:
                    return False
                
                if next_idx == k + 1:
                    # Row complete
                    # Check if current_diffs conflicts with used_diffs
                    if current_diffs & used_diffs:
                        return False
                    # Accept this row
                    rows.append(list(current_row))
                    used_diffs.update(current_diffs)
                    
                    if backtrack(row_idx + 1):
                        return True
                    
                    # Backtrack
                    rows.pop()
                    used_diffs.difference_update(current_diffs)
                    return False
                
                # Choose next element x
                # x must be > current_row[-1] and <= M
                # Also, the new diffs (x - y for y in current_row) must not be in used_diffs or current_diffs
                
                min_x = current_row[-1] + 1
                max_x = M
                
                # Optimization: If we are picking the last element, we can be more selective?
                # Let's just iterate x from min_x to max_x.
                # To add randomness, we can shuffle the range or pick random x's.
                # But systematic search is more likely to find a solution if one exists.
                # However, we might get stuck in a local minimum.
                # Let's try systematic first. If it fails, we might need to randomize.
                
                # Actually, for small M and k, systematic might be fast enough if we prune early.
                # Let's iterate x.
                
                # To improve performance, we can check conflicts early.
                for x in range(min_x, max_x + 1):
                    new_diffs = set()
                    conflict = False
                    for y in current_row:
                        d = x - y
                        if d in used_diffs or d in current_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    
                    if conflict:
                        continue
                    
                    # Recurse
                    current_row.append(x)
                    current_diffs.update(new_diffs)
                    
                    if extend_row(current_row, current_diffs, next_idx + 1):
                        return True
                    
                    # Backtrack
                    current_row.pop()
                    current_diffs.difference_update(new_diffs)
                
                return False
            
            # Start building row 0
            # Row 0 must start with 0.
            # So current_row = [0], current_diffs = set()
            # next_idx = 1 (we need to pick k more elements)
            
            return extend_row([0], set(), 1)
        
        # We might need to randomize the order of x choices to escape local minima.
        # Let's modify extend_row to shuffle the candidate x values.
        # But shuffling a range is expensive.
        # Alternative: Use a randomized greedy approach with backtracking.
        
        # Let's restart the backtracking with randomization if the first attempt fails.
        # Actually, let's just run the systematic backtracking. If it's too slow, we'll switch.
        # Given M ~ 105, k=5, the branching factor is ~105.
        # Depth is 5 (elements per row after 0).
        # 105^5 is ~ 10^10. Too slow for systematic.
        
        # We need a smarter search.
        # Heuristic: Choose x that minimizes future conflicts?
        # Or: Randomized restarts.
        
        # Let's try a different strategy:
        # 1. Generate a pool of valid rows (rows that don't conflict with themselves, which is always true).
        # 2. We need to select n_rows rows from this pool such that their diff sets are disjoint.
        # This is a set packing problem.
        
        # Since M is small, we can generate all possible rows?
        # C(112, 5) is ~ 10^8. Too many.
        
        # Let's go back to backtracking but with randomized order of x.
        
        def backtrack_random(row_idx):
            if time.time() > max_time:
                return False
            
            if row_idx == n_rows:
                return True
            
            # We will try to build a row.
            # To add randomness, we can shuffle the range of x.
            # But we need to be careful with backtracking.
            
            # Let's use a different approach:
            # Try to complete the row by picking random valid x's.
            # If we get stuck, backtrack.
            
            def extend_row_random(current_row, current_diffs, next_idx):
                if time.time() > max_time:
                    return False
                
                if next_idx == k + 1:
                    if current_diffs & used_diffs:
                        return False
                    rows.append(list(current_row))
                    used_diffs.update(current_diffs)
                    
                    if backtrack_random(row_idx + 1):
                        return True
                    
                    rows.pop()
                    used_diffs.difference_update(current_diffs)
                    return False
                
                min_x = current_row[-1] + 1
                max_x = M
                
                # Create a list of valid candidates
                candidates = []
                for x in range(min_x, max_x + 1):
                    new_diffs = set()
                    conflict = False
                    for y in current_row:
                        d = x - y
                        if d in used_diffs or d in current_diffs:
                            conflict = True
                            break
                        new_diffs.add(d)
                    if not conflict:
                        candidates.append((x, new_diffs))
                
                if not candidates:
                    return False
                
                # Shuffle candidates to add randomness
                random.shuffle(candidates)
                
                for x, new_diffs in candidates:
                    current_row.append(x)
                    current_diffs.update(new_diffs)
                    
                    if extend_row_random(current_row, current_diffs, next_idx + 1):
                        return True
                    
                    current_row.pop()
                    current_diffs.difference_update(new_diffs)
                
                return False
            
            return extend_row_random([0], set(), 1)
        
        # Try a few times with randomization
        for _ in range(10):
            if backtrack_random(0):
                return rows
            # Clear state for next try?
            # The state is local to solve_for_M, so it's fine.
            # But used_diffs and rows are modified in place.
            # We need to reset them.
            used_diffs.clear()
            rows.clear()
            
        return None

    # We will try M from lower_bound to some upper limit
    # Lower bound is total_diffs.
    # Upper limit: let's say 150.
    
    best_solution = None
    best_M = float('inf')
    
    # First, try to find a valid solution with a greedy method to have a fallback
    # Greedy: pick the smallest possible x for each element in each row
    def greedy_solve():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        return rows
    
    greedy_rows = greedy_solve()
    greedy_M = max(max(row) for row in greedy_rows)
    best_solution = greedy_rows
    best_M = greedy_M
    
    # Now try to improve with backtracking
    # Start from lower bound
    # Note: The counting bound is 105.
    # But is 105 achievable? Probably not.
    # We will try M from 105 to best_M - 1.
    
    for M in range(total_diffs, best_M):
        if time.time() > deadline - 5:
            break
        
        # Allocate a small time budget for each M
        # If M is close to best_M, we might spend more time?
        # Let's just try a few random restarts for each M.
        
        # We need to be careful with the time.
        # Let's give 5 seconds to each M.
        sol = solve_for_M(M, max_time=5.0)
        if sol:
            best_solution = sol
            best_M = M
            break # Found the smallest M, so we can stop
    
    # If we didn't find a better M, return the greedy solution
    # Or if we found a better M, return that.
    
    # Ensure the solution is valid
    def is_valid(rows):
        used = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                ri = row[i]
                for j in range(i + 1, n):
                    d = row[j] - ri
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    if not is_valid(best_solution):
        # Fallback to greedy
        best_solution = greedy_solve()
    
    return best_solution

# Test locally
if __name__ == "__main__":
    # (7, 5)
    rows = construct(7, 5)
    max_val = max(max(r) for r in rows)
    print(f"Max value: {max_val}")
    # Print rows
    for r in rows:
        print(r)
