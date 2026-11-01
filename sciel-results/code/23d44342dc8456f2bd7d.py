import time
import random
import sys
from collections import defaultdict

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows, each of length k+1.
    Strategy: Iterated Local Search with Targeted Scope Reduction.
    1. Find an initial valid solution using greedy/backtracking.
    2. Iteratively identify the row with the maximum element.
    3. Attempt to re-solve that specific row with a constraint that its max element 
       is strictly less than the current max, while avoiding all diffs from other rows.
    4. If direct shrink fails, perturb a random other row to change the forbidden diff set.
    5. Keep the best solution found.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Problem parameters
    # Each row has k+1 elements: 0 = x_0 < x_1 < ... < x_k
    # Differences: x_j - x_i for 0 <= i < j <= k
    # Total diffs per row: k*(k+1)/2
    # Total diffs: n_rows * k*(k+1)/2
    
    def get_diffs(row):
        """Calculate all positive differences in a row."""
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def is_valid_solution(rows):
        """Check if all differences across all rows are distinct."""
        all_diffs = []
        for row in rows:
            all_diffs.extend(get_diffs(row))
        return len(all_diffs) == len(set(all_diffs))

    def find_row_avoiding(forbidden_diffs, max_limit, node_limit=50000):
        """
        Finds a row of length k+1 starting with 0, strictly increasing,
        with distinct internal differences, none of which are in forbidden_diffs,
        and max element <= max_limit.
        Returns the row if found, else None.
        Uses randomized backtracking to escape local minima if needed, 
        but primarily tries to find *any* valid row under the limit.
        """
        # We use a simple backtracking with a node limit.
        # To make it "randomized" for diversity, we can shuffle the candidate x values 
        # if we allow it, but for finding a valid row under a tight limit, 
        # deterministic ascending is usually best for finding *a* solution if one exists.
        # However, for the "perturb" step, we want a *different* row.
        # So we will implement a version that can be seeded for randomness.
        
        row = [0]
        current_diffs = set()
        node_count = 0
        
        # Helper to check if a candidate x is valid
        def check_x(x, row, current_diffs, forbidden_diffs):
            new_diffs = []
            for a in row:
                d = x - a
                if d in forbidden_diffs or d in current_diffs:
                    return None
                new_diffs.append(d)
            return new_diffs

        def backtrack(idx, curr_max):
            nonlocal node_count
            node_count += 1
            if node_count > node_limit:
                return False
            
            if idx == k + 1:
                return True
            
            min_next = curr_max + 1
            # We must leave room for (k - idx) more elements
            max_next = max_limit - (k - idx)
            
            if min_next > max_next:
                return False
            
            # Generate candidates
            # For the "shrink" step, we want the smallest possible max, so ascending is good.
            # For the "perturb" step, we might want diversity. 
            # Here we stick to ascending for reliability, but the caller controls max_limit.
            candidates = list(range(min_next, max_next + 1))
            
            # If we want to force diversity in perturbation, we could shuffle here.
            # But let's keep it deterministic for the main shrink logic.
            
            for x in candidates:
                new_diffs = check_x(x, row, current_diffs, forbidden_diffs)
                if new_diffs is None:
                    continue
                
                row.append(x)
                current_diffs.update(new_diffs)
                
                if backtrack(idx + 1, x):
                    return True
                
                row.pop()
                current_diffs.difference_update(new_diffs)
            
            return False

        if backtrack(1, 0):
            return row[:]
        return None

    def find_row_any(forbidden_diffs, node_limit=50000):
        """
        Finds a valid row avoiding forbidden_diffs, without a strict max_limit,
        but tries to keep it small. 
        We binary search on max_limit to find the smallest possible row.
        """
        # Lower bound: k
        # Upper bound: estimate based on greedy
        # Greedy estimate
        used = set(forbidden_diffs)
        row = [0]
        curr_max = 0
        for _ in range(k):
            x = curr_max + 1
            while True:
                valid = True
                new_diffs = []
                for a in row:
                    d = x - a
                    if d in used or d in [nd for nd in new_diffs]: # local check
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    row.append(x)
                    used.update(new_diffs)
                    curr_max = x
                    break
                x += 1
        greedy_max = curr_max
        
        # Binary search between k and greedy_max + 10
        low = k
        high = greedy_max + 20
        
        best_row = None
        # We just want a valid row. Let's try to find the smallest one.
        while low <= high and node_limit > 0:
            mid = (low + high) // 2
            row = find_row_avoiding(forbidden_diffs, mid, node_limit=node_limit//2)
            if row is not None:
                best_row = row
                high = mid - 1
            else:
                low = mid + 1
        
        if best_row is None:
             # If binary search failed, just return a greedy one
             return row # The greedy one from above is valid
        return best_row

    # 1. Generate Initial Solution
    # Use a simple greedy or backtracking to get a valid start.
    # Greedy is fast but might have a large scope.
    # Let's use a backtracking with a high limit to get a "tighter" initial solution.
    
    initial_rows = []
    used_diffs = set()
    
    for i in range(n_rows):
        # Find a row avoiding used_diffs
        # Start with a generous limit, then shrink? 
        # For init, just find *any* valid row.
        row = find_row_avoiding(used_diffs, 500, node_limit=100000)
        if row is None:
             # Fallback to greedy
             row = [0]
             curr_max = 0
             while len(row) < k + 1:
                 x = curr_max + 1
                 while True:
                     valid = True
                     for a in row:
                         if (x - a) in used_diffs:
                             valid = False
                             break
                     if valid:
                         row.append(x)
                         used_diffs.update(x - a for a in row[:-1])
                         curr_max = x
                         break
                     x += 1
        initial_rows.append(row)
        used_diffs.update(get_diffs(row))
    
    if not is_valid_solution(initial_rows):
        # Should not happen if find_row_avoiding works correctly
        pass

    best_rows = [r[:] for r in initial_rows]
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Iterated Improvement Loop
    # Strategy: 
    # - Find the row with the max element.
    # - Try to shrink it.
    # - If fail, perturb a random other row.
    
    current_rows = [r[:] for r in initial_rows]
    current_scope = best_scope
    
    while time.time() < deadline - 5.0:
        # Identify the row to shrink
        # Find the index of the row with the maximum element
        max_val = -1
        target_row_idx = -1
        for i, row in enumerate(current_rows):
            if max(row) > max_val:
                max_val = max(row)
                target_row_idx = i
        
        # Collect forbidden diffs from all OTHER rows
        forbidden_diffs = set()
        for i in range(n_rows):
            if i != target_row_idx:
                forbidden_diffs.update(get_diffs(current_rows[i]))
        
        # Target: Find a new row for target_row_idx with max < max_val
        # We can try to shrink by 1, or by a larger amount.
        # Let's try to find the smallest possible row avoiding forbidden_diffs.
        # If the smallest possible row's max is < max_val, we improve.
        
        # We use find_row_avoiding with a limit of max_val - 1.
        # If it finds a row, its max will be <= max_val - 1.
        new_row = find_row_avoiding(forbidden_diffs, max_val - 1, node_limit=20000)
        
        if new_row is not None:
            # Success! Update the row
            current_rows[target_row_idx] = new_row
            new_scope = max(max(r) for r in current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
            # We can continue to try to shrink further in the next iteration
            # The loop will naturally pick the new max (which might be the same row or another)
        else:
            # Failed to shrink the target row directly.
            # Perturb a random OTHER row to change the landscape.
            # Choose a random row different from target_row_idx
            if n_rows > 1:
                other_indices = [i for i in range(n_rows) if i != target_row_idx]
                perturb_idx = random.choice(other_indices)
                
                # Collect forbidden diffs for the row we are about to perturb
                # (All rows except the one being perturbed)
                forbidden_for_perturb = set()
                for i in range(n_rows):
                    if i != perturb_idx:
                        forbidden_for_perturb.update(get_diffs(current_rows[i]))
                
                # Find a *different* valid row for perturb_idx.
                # To ensure it's different, we can try to find a row with a max 
                # slightly different from the current one, or just re-find one.
                # Simple approach: Find a new row avoiding forbidden_for_perturb.
                # To avoid finding the exact same row, we can bias the search 
                # or just accept that it might find the same one (which is a no-op).
                # To force change, we can try to find a row with max > current_max + 1 
                # or max < current_max - 1? 
                # Actually, just finding *any* valid row is fine. If it's the same, 
                # the state doesn't change, and we'll try again.
                # To increase diversity, we can randomize the backtracking in find_row_avoiding?
                # For now, let's just re-find. If we get the same row, we try a different perturb_idx next time.
                
                old_row_max = max(current_rows[perturb_idx])
                
                # Try to find a row that is NOT the same as the current one.
                # We can do this by trying to find a row with max != old_row_max.
                # Or simpler: just re-run find_row_avoiding. 
                # If it returns the same row, we skip the update for this iteration.
                
                new_perturb_row = find_row_avoiding(forbidden_for_perturb, 500, node_limit=20000)
                
                if new_perturb_row is not None and new_perturb_row != current_rows[perturb_idx]:
                    current_rows[perturb_idx] = new_perturb_row
                    # Update best if this perturbation accidentally improved scope? 
                    # Unlikely, but check.
                    pert_scope = max(max(r) for r in current_rows)
                    if pert_scope < best_scope:
                        best_scope = pert_scope
                        best_rows = [r[:] for r in current_rows]
                else:
                    # If we couldn't find a different row, try to force a change 
                    # by adding a temporary constraint? 
                    # Or just move on.
                    pass
            else:
                # Only 1 row? Trivial case, but task implies n_rows >= 2 usually.
                # If n_rows == 1, we just minimize the single row.
                # find_row_avoiding with empty forbidden set gives the optimal row.
                new_row = find_row_avoiding(set(), 500, node_limit=20000)
                if new_row is not None:
                    current_rows[0] = new_row
                    best_scope = max(new_row)
                    best_rows = [new_row[:]]

    return best_rows

# Test harness for local validation (not run by evaluator)
if __name__ == "__main__":
    rows = construct(4, 7)
    scope = max(max(r) for r in rows)
    print(f"Scope: {scope}")
    print(rows)
    # Validate
    all_diffs = []
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                all_diffs.append(r[j] - r[i])
    if len(all_diffs) == len(set(all_diffs)):
        print("Valid")
    else:
        print("Invalid")
