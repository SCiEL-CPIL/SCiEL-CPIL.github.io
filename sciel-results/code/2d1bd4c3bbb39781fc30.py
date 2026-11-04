import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All differences between entries in the same row must be distinct over all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    best_rows = None
    best_scope = float('inf')
    
    def get_scope(rows):
        return max(max(r) for r in rows) if rows else 0

    def is_valid_move(rows, used_diffs, r_idx, j_idx, new_val):
        """
        Check if changing rows[r_idx][j_idx] to new_val is valid.
        """
        row = rows[r_idx]
        # Check strict increase
        if j_idx > 0 and new_val <= row[j_idx - 1]:
            return False
        if j_idx < k and new_val >= row[j_idx + 1]:
            return False
        
        # Calculate old diffs for this row
        old_diffs = set()
        for i in range(k + 1):
            for l in range(i + 1, k + 1):
                old_diffs.add(row[l] - row[i])
        
        # Calculate new diffs
        new_row = row.copy()
        new_row[j_idx] = new_val
        new_diffs = set()
        for i in range(k + 1):
            for l in range(i + 1, k + 1):
                new_diffs.add(new_row[l] - new_row[i])
        
        # Differences that are removed
        removed = old_diffs - new_diffs
        # Differences that are added
        added = new_diffs - old_diffs
        
        # The global set of used diffs, excluding the ones from this row
        # used_diffs currently contains old_diffs.
        # So used_diffs - removed gives the diffs used by OTHER rows.
        other_diffs = used_diffs - removed
        
        # Check if any added diff conflicts with other_diffs
        if added & other_diffs:
            return False
        
        return True

    def apply_move(rows, used_diffs, r_idx, j_idx, new_val):
        """
        Apply the move and update used_diffs.
        """
        row = rows[r_idx]
        old_val = row[j_idx]
        
        # Calculate old diffs
        old_diffs = set()
        for i in range(k + 1):
            for l in range(i + 1, k + 1):
                old_diffs.add(row[l] - row[i])
        
        # Update row
        row[j_idx] = new_val
        
        # Calculate new diffs
        new_diffs = set()
        for i in range(k + 1):
            for l in range(i + 1, k + 1):
                new_diffs.add(row[l] - row[i])
        
        # Update used_diffs
        # Remove old diffs that are no longer present
        # Add new diffs that were not present
        # Note: used_diffs is a global set.
        # We need to be careful.
        # The diffs in `removed` are no longer in this row.
        # But they might be in other rows? No, the problem states all differences are distinct over all rows.
        # So if a diff is in `removed`, it was ONLY in this row.
        # So we can safely remove it from `used_diffs`.
        # Similarly, `added` diffs were not in this row, and since the move was valid, they are not in other rows.
        
        # Actually, simpler:
        # used_diffs = (used_diffs - old_diffs) | new_diffs
        # But this is O(|used_diffs|) which is ~1000. It's fine.
        # However, doing set difference every time might be slow if we do it many times.
        # Let's just do:
        used_diffs.difference_update(old_diffs)
        used_diffs.update(new_diffs)
        
        return used_diffs

    def greedy_construct(seed):
        random.seed(seed)
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Try to add x
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
                else:
                    # If conflict, we might want to try a larger x?
                    # The standard greedy just increments x.
                    # To add randomness, we could pick a random x > current x?
                    # But that might blow up the scope.
                    # Let's stick to simple greedy for initialization, but maybe shuffle the order of rows?
                    # Rows are indistinguishable in terms of constraints, but the "used" set depends on order.
                    # So order matters.
                    pass
            rows.append(row)
        return rows, used

    def local_search(initial_rows, initial_used, time_budget):
        nonlocal best_rows, best_scope
        
        rows = [r.copy() for r in initial_rows]
        used = set(initial_used)
        
        # Precompute diffs for each row to speed up move checks?
        # is_valid_move recalculates diffs. That's O(k^2). k=5, so 15 ops. Very fast.
        
        current_scope = get_scope(rows)
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [r.copy() for r in rows]
            
        no_improve_count = 0
        max_no_improve = 5000 # Threshold to restart
        
        while time.time() - start_time < time_budget:
            # Pick a random row and a random element to decrease
            # Or pick the row with max element and the element with max value
            # Let's try: 50% chance pick global max element, 50% chance pick random non-zero element
            
            if random.random() < 0.5:
                # Find global max
                max_val = -1
                max_r = -1
                max_j = -1
                for r_idx, row in enumerate(rows):
                    for j_idx, val in enumerate(row):
                        if val > max_val:
                            max_val = val
                            max_r = r_idx
                            max_j = j_idx
                # Try to decrease max_r, max_j
                if max_j == 0:
                    continue
                r_idx, j_idx = max_r, max_j
                new_val = rows[r_idx][j_idx] - 1
            else:
                r_idx = random.randint(0, n_rows - 1)
                # Pick random index > 0
                j_idx = random.randint(1, k)
                new_val = rows[r_idx][j_idx] - 1
                
            if new_val < 0:
                continue
                
            if is_valid_move(rows, used, r_idx, j_idx, new_val):
                # Apply move
                used = apply_move(rows, used, r_idx, j_idx, new_val)
                
                new_scope = get_scope(rows)
                if new_scope < current_scope:
                    current_scope = new_scope
                    no_improve_count = 0
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [r.copy() for r in rows]
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
                
            if no_improve_count > max_no_improve:
                break
                
        return best_rows, best_scope

    # Strategy:
    # 1. Generate a few initial solutions with different seeds.
    # 2. Run local search on the best one (or all, depending on time).
    
    initial_candidates = []
    for i in range(10): # 10 initial guesses
        rows, used = greedy_construct(i)
        scope = get_scope(rows)
        initial_candidates.append((scope, rows, used))
        
    # Sort by scope
    initial_candidates.sort(key=lambda x: x[0])
    
    # Run local search on the best initial candidate
    # We can also run on the second best if time permits
    
    # Let's run local search on the best one first
    best_rows_ls, best_scope_ls = local_search(initial_candidates[0][1], initial_candidates[0][2], time_limit * 0.8)
    
    # If time remains, try another initial candidate
    if time.time() - start_time < time_limit * 0.9 and len(initial_candidates) > 1:
        # Pick the second best distinct one?
        # Just try the second one
        best_rows_ls2, best_scope_ls2 = local_search(initial_candidates[1][1], initial_candidates[1][2], time_limit * 0.1)
        if best_scope_ls2 < best_scope_ls:
            best_rows_ls = best_rows_ls2
            best_scope_ls = best_scope_ls2
            
    if best_rows_ls is None:
        # Fallback to greedy
        return initial_candidates[0][1]
        
    return best_rows_ls

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to wrap it in the EVOLVE-BLOCK if required by the prompt structure?
# The prompt says "Keep the function names and signatures of the seed program."
# And "Output the COMPLETE program in a single python code block."
# The seed program had:
# """Initial program..."""
# # EVOLVE-BLOCK-START
# def construct(...):
#     ...
# # EVOLVE-BLOCK-END

# So I will output the whole file content.
# EVOLVE-BLOCK-END
