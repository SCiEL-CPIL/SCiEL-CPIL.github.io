import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]
    
    # Total differences per row: C(k+1, 2)
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs
    
    # For (4,7): C(8,2)=28, total=112. Best known 134.
    
    def get_row_diffs(row):
        return set(row[j] - row[i] for i in range(len(row)) for j in range(i + 1, len(row)))
    
    def check_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def greedy_row(used_diffs, k, max_val=200):
        """
        Greedily construct a single row given the set of already used differences.
        Returns the row or None if impossible within max_val.
        """
        row = [0]
        current_diffs = set()
        
        for _ in range(k):
            x = row[-1] + 1
            forbidden = used_diffs | current_diffs
            while x < max_val:
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden):
                    row.append(x)
                    current_diffs |= new_diffs
                    break
                x += 1
            else:
                return None
        
        return row

    def randomized_row(used_diffs, k, max_val=200, p_smallest=0.7, window=5):
        """
        Construct a row with some randomization among valid candidates.
        """
        row = [0]
        current_diffs = set()
        
        for _ in range(k):
            candidates = []
            x = row[-1] + 1
            forbidden = used_diffs | current_diffs
            
            while x < max_val:
                new_diffs = {x - a for a in row}
                if not (new_diffs & forbidden):
                    candidates.append(x)
                    if len(candidates) >= window:
                        break
                x += 1
            
            if not candidates:
                return None
            
            if random.random() < p_smallest:
                next_x = candidates[0]
            else:
                next_x = random.choice(candidates)
            
            row.append(next_x)
            current_diffs |= {next_x - a for a in row[:-1]}
            
        return row

    # 1. Initial Solution: Deterministic Greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(used, k, max_val=300)
        if row is None:
            return None # Should not happen for reasonable inputs
        rows.append(row)
        used |= get_row_diffs(row)
        
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Simulated Annealing
    # State: list of rows
    # Objective: minimize max(max(r) for r in rows)
    # Move: Regenerate one row entirely using randomized_row, respecting used diffs of other rows.
    
    T = 5.0
    cooling = 0.998
    min_T = 0.01
    iterations = 0
    no_improve_count = 0
    
    current_scope = best_scope
    
    # Precompute diffs for current rows to speed up updates?
    # We'll just recompute row diffs when needed, or maintain a global used set.
    # Maintaining global used set:
    # used_all = union of diffs of all rows.
    # When swapping row i:
    # 1. Remove diffs of row i from used_all.
    # 2. Generate new row i against used_all.
    # 3. If successful, add new diffs to used_all.
    
    used_all = set()
    for r in rows:
        used_all |= get_row_diffs(r)
        
    row_diffs_list = [get_row_diffs(r) for r in rows]
    
    while time.time() < deadline:
        iterations += 1
        r_idx = random.randint(0, n_rows - 1)
        
        # Remove current row's diffs
        used_all -= row_diffs_list[r_idx]
        
        # Generate new row
        # Use a slightly larger max_val for generation to allow exploration,
        # but the score depends on the actual max entry.
        # If we generate a row with a large entry, the scope increases.
        # We want to encourage smaller entries.
        # randomized_row picks from a window. Smaller x is more likely.
        new_row = randomized_row(used_all, k, max_val=300, p_smallest=0.8, window=3)
        
        if new_row is None:
            # Rollback
            used_all |= row_diffs_list[r_idx]
            # Reduce temperature if stuck?
            T *= cooling
            if T < min_T:
                T = min_T
            continue
        
        new_row_diffs = get_row_diffs(new_row)
        
        # Calculate new scope
        # We need max of (max of other rows, max of new_row)
        # To avoid O(n) scan, we could maintain a multiset of row maxes, but n is small.
        other_max = 0
        for i in range(n_rows):
            if i != r_idx:
                if max(rows[i]) > other_max:
                    other_max = max(rows[i])
        
        new_global_scope = max(other_max, max(new_row))
        delta = new_global_scope - current_scope
        
        accept = False
        if delta < 0:
            accept = True
        else:
            if delta == 0:
                # Accept with high probability to explore
                accept = random.random() < 0.5
            else:
                prob = math.exp(-delta / T)
                accept = random.random() < prob
        
        if accept:
            rows[r_idx] = new_row
            row_diffs_list[r_idx] = new_row_diffs
            used_all |= new_row_diffs
            current_scope = new_global_scope
            
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in rows]
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Rollback
            used_all |= row_diffs_list[r_idx]
        
        T *= cooling
        if T < min_T:
            T = min_T
            
        # Optional: Restart if stuck for too long?
        # The prompt says "Spare time does not establish a bottleneck...".
        # But if we are stuck at a local minimum, restarting might help.
        # However, SA with low T is good for local refinement.
        # Let's rely on the diversity of randomized_row.
        
    # 3. Local Search / Hill Climbing on the best solution found
    # Try to reduce the scope further by targeted moves.
    # Move: Reduce a specific element in a specific row.
    
    final_rows = [r[:] for r in best_rows]
    final_scope = best_scope
    used_final = set()
    for r in final_rows:
        used_final |= get_row_diffs(r)
    final_row_diffs = [get_row_diffs(r) for r in final_rows]
    
    # Local search deadline: use remaining time
    local_deadline = deadline
    improved = True
    while improved and time.time() < local_deadline:
        improved = False
        # Try to reduce the largest element of each row
        # Or try to reduce any element?
        # Reducing an element might force increasing later elements if it creates conflicts?
        # No, reducing an element creates smaller differences.
        # If we reduce row[i][j], the differences row[i][j] - row[i][m] for m<j decrease.
        # The differences row[i][n] - row[i][j] for n>j increase.
        # This can cause conflicts.
        
        # A simpler local move: Swap two rows? No, rows are distinct sets.
        # Move: Regenerate the row with the largest max entry, but with a tighter constraint?
        # Or: Try to find a smaller valid value for the last element of each row.
        
        for r_idx in range(n_rows):
            if time.time() > local_deadline:
                break
            row = final_rows[r_idx]
            last_val = row[-1]
            
            # Try to reduce last_val
            # We need to check if we can pick a smaller value for row[-1]
            # that is > row[-2] and valid.
            
            # Remove current last element's contribution?
            # It's easier to just try replacing the whole row or the last few elements.
            # Let's try replacing the last element.
            
            old_last_diffs = {row[-1] - a for a in row[:-1]}
            used_wo_last = used_final - old_last_diffs
            
            for new_val in range(last_val - 1, row[-2], -1):
                # Check if new_val is valid
                new_diffs = {new_val - a for a in row[:-1]}
                if not (new_diffs & used_wo_last):
                    # Also need to ensure new_diffs are distinct?
                    # Yes, they are distinct because a is distinct.
                    # And they are positive.
                    
                    # Check if new_val > row[-2]
                    if new_val > row[-2]:
                        # Accept this move
                        final_rows[r_idx][-1] = new_val
                        final_row_diffs[r_idx] = get_row_diffs(final_rows[r_idx])
                        
                        # Update used_final
                        used_final -= old_last_diffs
                        used_final |= new_diffs
                        
                        new_scope = max(max(r) for r in final_rows)
                        if new_scope < final_scope:
                            final_scope = new_scope
                            improved = True
                        break # Move to next row
            
    # Final validation
    if not check_valid(final_rows):
        # Fallback to best_rows if something went wrong (shouldn't happen)
        return best_rows
        
    return final_rows
