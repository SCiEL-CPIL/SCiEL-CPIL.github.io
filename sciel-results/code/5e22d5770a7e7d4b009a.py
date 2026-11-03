import time
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct globally.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = 145.0 # Safety margin for 150s limit
    
    m = k + 1 # Number of elements in each row
    
    # Lower bound for M
    # Total diffs = n_rows * m*(m-1)/2
    # These must fit in 1..M
    # M >= n_rows * m*(m-1)/2
    lb = n_rows * m * (m - 1) // 2
    
    # A slightly tighter bound might be derived from the fact that
    # the largest diff in each row is <= M.
    # But let's start with lb and iterate up.
    
    best_solution = None
    best_scope = float('inf')
    
    # We will search for a solution with scope M.
    # We iterate M from lb upwards.
    
    def dfs(row_idx, current_row, used_diffs, current_scope):
        nonlocal best_solution, best_scope, deadline
        
        if time.time() > deadline:
            return False
        
        if row_idx == n_rows:
            # All rows constructed
            if current_scope < best_scope:
                best_scope = current_scope
                best_solution = [list(r) for r in current_rows] # Need to pass current_rows
            return True
        
        # Heuristic: If we are building row `row_idx`, and we have a symmetry constraint
        # e.g., row_idx == 0 or current_row[1] >= prev_row[1]
        
        # To keep it simple and robust, we don't enforce strict ordering of rows initially,
        # but we can enforce that the first element (after 0) is non-decreasing to break symmetry.
        # Let's assume rows are processed in order 0..n_rows-1.
        # Symmetry break: row[i][1] >= row[i-1][1] for i > 0.
        
        if row_idx > 0 and len(current_row) > 0 and len(prev_rows[row_idx-1]) > 0:
             # This logic is tricky inside DFS if we don't track prev_rows easily.
             # Let's just rely on the search space being small enough for small n_rows, k.
             # For n_rows=7, k=5, m=6.
             # Diffs per row = 15. Total 105 diffs.
             # M is around 105-120.
             # Search space is huge. We need strong pruning.
             pass
        
        # We build the current row element by element.
        # current_row starts as [0]
        # We need to add m-1 more elements.
        
        # This structure implies we should DFS over the elements of the current row,
        # and only move to the next row when the current one is complete.
        
        # Let's restructure: 
        # State: (row_idx, pos_in_row, current_row, used_diffs)
        # If pos_in_row == m:
        #    if row_idx == n_rows-1: success
        #    else: dfs(row_idx+1, 0, [0], used_diffs)
        
        # But we need to enforce symmetry between rows.
        # Let's store the completed rows in a list `rows` passed by reference or closure.
        
        pass

    # Let's implement a cleaner recursive solver.
    
    rows = []
    
    def solve(M):
        """
        Try to find a valid set of rows with max element <= M.
        Returns True if found, and updates `rows`.
        """
        nonlocal deadline, best_solution, best_scope
        
        if time.time() > deadline:
            return False
            
        used_diffs = set()
        rows.clear()
        
        def dfs(row_idx, current_row):
            nonlocal used_diffs, rows
            
            if time.time() > deadline:
                return False
                
            if row_idx == n_rows:
                # Check if we improved best
                # The scope is determined by the max of all elements
                # Since we enforce max <= M in the search, and we iterate M,
                # the first success is the optimal M? Not necessarily if we don't search M in order?
                # We are searching for a specific M.
                # If we find a solution for M, we return True.
                return True
            
            # Symmetry breaking: 
            # If row_idx > 0, current_row[1] must be >= rows[row_idx-1][1]
            # Note: current_row starts as [0], so current_row[1] is the first mark we add.
            # But we add marks one by one.
            # Let's enforce: The first non-zero element of row i must be >= first non-zero of row i-1.
            
            min_first_mark = 0
            if row_idx > 0:
                if rows[row_idx-1]:
                    # rows[row_idx-1] is the full previous row
                    min_first_mark = rows[row_idx-1][1]
                else:
                    return False # Should not happen
            
            # If we are at the start of a new row (current_row == [0])
            if len(current_row) == 1:
                # We need to choose the next mark x such that x >= min_first_mark
                # And x - 0 = x is not in used_diffs
                # Also, we should try to keep x small? Or large?
                # Small x leaves more room for subsequent marks?
                # Usually, greedy smallest-first works well for packing.
                
                for x in range(min_first_mark, M + 1):
                    if x == 0: continue
                    if x in used_diffs: continue
                    
                    # Check if we can complete the row with this first mark?
                    # Forward checking:
                    # The remaining marks must be > x.
                    # The max possible mark is M.
                    # We need m-2 more marks.
                    # The differences will be x, and (x2-x), (x2-0), etc.
                    
                    # Tentative add
                    new_used = used_diffs | {x}
                    new_row = current_row + [x]
                    
                    if dfs(row_idx, new_row):
                        return True
                    # Note: We don't need to explicitly remove x from used_diffs 
                    # if we pass new_used, but here we are mutating?
                    # Let's use immutable sets or careful mutation.
                    # For speed, mutation with backtracking is better.
                    
            else:
                # Continue current row
                last_mark = current_row[-1]
                # Next mark x must be > last_mark
                # Differences: x - a for a in current_row
                # These must not be in used_diffs
                
                # Optimization: 
                # The smallest possible next mark is last_mark + 1.
                # The largest is M.
                
                # We can prune if even the smallest possible differences conflict?
                
                for x in range(last_mark + 1, M + 1):
                    # Check differences
                    valid = True
                    diffs_to_check = []
                    for a in current_row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        diffs_to_check.append(d)
                    
                    if not valid:
                        continue
                        
                    # Additional Pruning:
                    # Can we still fit the remaining marks?
                    # Remaining marks: m - len(current_row) - 1
                    # They must be > x.
                    # Max available space: M - x
                    # Min diff between consecutive marks is 1.
                    # So we need at least (remaining marks) * 1 space?
                    # Actually, just M - x >= remaining_marks is a weak bound.
                    
                    # Stronger:
                    # The new differences added are diffs_to_check.
                    # Do any of these conflict with *potential* future differences?
                    # Hard to check.
                    
                    # Let's just do the mutation
                    used_diffs.update(diffs_to_check)
                    current_row.append(x)
                    
                    if dfs(row_idx, current_row):
                        return True
                    
                    # Backtrack
                    current_row.pop()
                    used_diffs.difference_update(diffs_to_check)
                
            return False
        
        # Start DFS
        # We must ensure the first row's first mark is >= 0 (always true)
        # But wait, the symmetry break `min_first_mark` for row 0 is 0.
        # So row 0 can start with any x >= 1.
        
        if dfs(0, [0]):
            # Save solution
            return True
        
        return False

    # Iterate M from lower bound
    # We need a good lower bound.
    # lb = n_rows * m * (m-1) // 2
    # For n_rows=7, m=6, lb = 7 * 15 = 105.
    # The prompt says best known is 112.
    
    # To speed up, we can jump M.
    # But linear scan from lb is safe if DFS is fast.
    
    # However, DFS with Python sets might be slow for large M.
    # We should use a bitset for used_diffs if M is small (< 1000).
    # M is around 100-200. A Python int can act as a bitset.
    
    # Let's rewrite the core using bitsets for speed.
    
    def solve_bitset(M):
        nonlocal deadline, best_solution, best_scope
        
        if time.time() > deadline:
            return False
            
        used_bits = 0
        rows.clear()
        
        def dfs(row_idx, current_row):
            nonlocal used_bits, rows
            
            if time.time() > deadline:
                return False
                
            if row_idx == n_rows:
                return True
            
            min_first_mark = 0
            if row_idx > 0:
                min_first_mark = rows[row_idx-1][1]
            
            if len(current_row) == 1:
                # Choose first mark
                for x in range(min_first_mark, M + 1):
                    if x == 0: continue
                    bit = 1 << x
                    if used_bits & bit: continue
                    
                    # Try
                    new_used = used_bits | bit
                    new_row = current_row + [x]
                    
                    if dfs(row_idx, new_row):
                        return True
            
            else:
                last_mark = current_row[-1]
                # Precompute bits for differences?
                # For each x, we need to check if any (x-a) is in used_bits.
                
                for x in range(last_mark + 1, M + 1):
                    valid = True
                    new_diff_bits = 0
                    for a in current_row:
                        d = x - a
                        bit = 1 << d
                        if used_bits & bit:
                            valid = False
                            break
                        new_diff_bits |= bit
                    
                    if not valid:
                        continue
                        
                    # Check if new_diff_bits conflicts with each other?
                    # No, differences within a row are distinct by construction?
                    # x - a1 and x - a2 are distinct if a1 != a2.
                    # But do they conflict with *each other*? No, they are new.
                    # Do they conflict with *existing* used_bits? We checked that.
                    
                    # Mutation
                    used_bits |= new_diff_bits
                    current_row.append(x)
                    
                    if dfs(row_idx, current_row):
                        return True
                    
                    current_row.pop()
                    used_bits &= ~new_diff_bits
            
            return False
        
        if dfs(0, [0]):
            return True
        return False

    # Start searching
    # We can parallelize or just run sequentially.
    # Given the 150s limit, sequential is fine if efficient.
    
    # To avoid getting stuck in a very hard M, we can limit the time per M.
    # But usually, the optimal M is close to the lower bound.
    
    # Let's try M from lb to lb + 50 or so.
    # If we don't find it, we expand.
    
    # Actually, let's just iterate M and break on first success.
    # Since we want the *smallest* M, the first success is the answer.
    
    # Optimization: Randomize the order of x?
    # No, deterministic is better for reproducibility and finding the "canonical" small solution.
    
    # One issue: The symmetry break `rows[row_idx-1][1]` assumes we fill rows sequentially.
    # Is it possible that the optimal solution has rows NOT sorted by their first mark?
    # Yes, but since the rows are indistinguishable, we can WLOG sort them by their first mark.
    # So enforcing this order does not exclude any optimal solution.
    
    # What if two rows have the same first mark?
    # Then we should break tie by second mark, etc.
    # For simplicity, `>=` is sufficient to prune permutations.
    
    # Let's run the search.
    
    # To handle the case where lb is a weak bound and search takes too long,
    # we can use a timeout per M.
    
    max_M_to_try = lb + 100 # Heuristic upper limit for initial scan
    found = False
    
    for M in range(lb, max_M_to_try + 1):
        if time.time() > deadline - 5: # Leave 5s for cleanup
            break
            
        # Reset state
        rows = []
        if solve_bitset(M):
            best_scope = M
            best_solution = [list(r) for r in rows]
            found = True
            break
        else:
            # If time is running out, we might want to save the best partial?
            # No, we need a valid full set.
            pass
            
    if not found:
        # If we didn't find a solution in the initial range, 
        # we might have to do a more extensive search or return a greedy solution.
        # But the task implies a solution exists near the bound.
        # Let's try a greedy fallback to ensure we return *something* valid, 
        # even if not optimal, to avoid 0 score.
        
        # Greedy construction
        used = set()
        greedy_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            greedy_rows.append(row)
        
        if best_solution is None:
            best_solution = greedy_rows
            best_scope = max(max(r) for r in greedy_rows)
        else:
            # Compare scopes
            greedy_scope = max(max(r) for r in greedy_rows)
            if greedy_scope < best_scope:
                best_solution = greedy_rows
                best_scope = greedy_scope

    return best_solution
