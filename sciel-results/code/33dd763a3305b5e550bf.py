import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = 118.0  # Keep a margin from the 150s limit

    # Total differences required
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # First, get a valid greedy solution as fallback
    used_diffs_greedy = set()
    greedy_rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            x = current_max + 1
            while True:
                new_diffs = {x - a for a in row}
                if not new_diffs & used_diffs_greedy:
                    break
                x += 1
            row.append(x)
            current_max = x
            used_diffs_greedy |= new_diffs
        greedy_rows.append(row)

    greedy_scope = max(max(r) for r in greedy_rows)
    best_rows = [r[:] for r in greedy_rows]
    best_scope = greedy_scope

    def get_scope(rows):
        return max(max(r) for r in rows)

    def compute_all_diffs(rows):
        """Compute the set of all differences across all rows."""
        all_diffs = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    all_diffs.add(row[j] - row[i])
        return all_diffs

    def get_row_diffs(row):
        """Get all differences within a single row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_move(current_rows, row_idx, pos, new_val):
        """
        Check if changing current_rows[row_idx][pos] to new_val is valid.
        Returns True if the move doesn't create any duplicate differences.
        """
        row = current_rows[row_idx]
        
        # Ensure strictly increasing
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return False
        
        # Compute old differences involving position pos
        old_diffs = set()
        for j in range(len(row)):
            if j != pos:
                if j < pos:
                    old_diffs.add(row[pos] - row[j])
                else:
                    old_diffs.add(row[j] - row[pos])
        
        # Compute new differences involving position pos
        new_diffs = set()
        for j in range(len(row)):
            if j != pos:
                if j < pos:
                    new_diffs.add(new_val - row[j])
                else:
                    new_diffs.add(row[j] - new_val)
        
        # Check for internal conflicts in the same row (differences not involving pos)
        same_row_other_diffs = set()
        for a in range(len(row)):
            for b in range(a + 1, len(row)):
                if a != pos and b != pos:
                    same_row_other_diffs.add(row[b] - row[a])
        
        # New diffs must not conflict with same-row other diffs
        if new_diffs & same_row_other_diffs:
            return False
        
        # Check conflicts with other rows
        # Build set of all diffs from other rows
        other_diffs = set()
        for i in range(n_rows):
            if i == row_idx:
                continue
            r = current_rows[i]
            for a in range(len(r)):
                for b in range(a + 1, len(r)):
                    other_diffs.add(r[b] - r[a])
        
        # New diffs must not conflict with other rows' diffs
        if new_diffs & other_diffs:
            return False
        
        return True

    def local_search(rows, max_iterations=2000):
        """
        Iterated Local Search: repeatedly try single-element moves to reduce scope.
        Returns the best rows found.
        """
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        
        # Precompute all diffs for quick reference
        all_diffs = compute_all_diffs(current_rows)
        
        for it in range(max_iterations):
            if time.time() > deadline - 8.0:
                break
            
            # Try to find a single move that reduces the scope
            found_improvement = False
            
            # Collect candidate moves: focus on rows that achieve the current scope
            scope_rows = [i for i in range(n_rows) if max(current_rows[i]) == current_scope]
            
            # Try moves on scope-achieving rows first, then others
            row_order = scope_rows + [i for i in range(n_rows) if i not in scope_rows]
            random.shuffle(row_order)
            
            for row_idx in row_order:
                if time.time() > deadline - 8.0:
                    break
                if found_improvement:
                    break
                
                row = current_rows[row_idx]
                row_max = max(row)
                
                # For each position, try to reduce the value
                # Focus on positions that contribute to the max
                for pos in range(len(row) - 1, 0, -1):  # Try from end to start
                    if time.time() > deadline - 8.0:
                        break
                    if found_improvement:
                        break
                    
                    old_val = row[pos]
                    
                    # Try reducing this value
                    # How much can we reduce? Must stay > row[pos-1]
                    min_val = row[pos - 1] + 1 if pos > 0 else 1
                    # Also must stay < row[pos+1] if pos < len-1
                    max_allowed = row[pos + 1] - 1 if pos < len(row) - 1 else old_val
                    
                    # Try values from old_val-1 down to min_val
                    # Limit search range for efficiency
                    search_lo = max(min_val, old_val - 15)
                    search_hi = min(max_allowed, old_val - 1)
                    
                    if search_lo > search_hi:
                        continue
                    
                    # Try a few random candidates in range plus the best (smallest) option
                    candidates = set()
                    candidates.add(search_hi)  # smallest valid (most reduction)
                    # Add some random candidates
                    for _ in range(5):
                        c = random.randint(search_lo, search_hi)
                        candidates.add(c)
                    
                    for new_val in sorted(candidates, key=lambda x: x):  # Try most reducing first
                        if is_valid_move(current_rows, row_idx, pos, new_val):
                            # Check if this improves scope
                            new_row_max = max(row[:pos] + [new_val] + row[pos+1:])
                            new_scope = max(
                                max(max(r) for i, r in enumerate(current_rows) if i != row_idx),
                                new_row_max
                            )
                            if new_scope < current_scope:
                                # Accept the move
                                current_rows[row_idx][pos] = new_val
                                current_scope = new_scope
                                if current_scope < best_scope:
                                    best_scope = current_scope
                                    best_rows = [r[:] for r in current_rows]
                                found_improvement = True
                                break
                            elif new_scope == current_scope:
                                # Accept neutral moves sometimes to escape local optima
                                if random.random() < 0.3:
                                    current_rows[row_idx][pos] = new_val
                                    # current_scope unchanged
                                    break
                        # If new_val is too high, no point trying higher values
                        if new_val > old_val - 1:
                            break
                # End position loop
            # End row loop
            
            if not found_improvement:
                # No improvement found; apply a kick perturbation
                # Randomly perturb 1-3 elements in random rows
                n_kicks = random.randint(1, 3)
                for _ in range(n_kicks):
                    if time.time() > deadline - 8.0:
                        break
                    row_idx = random.randint(0, n_rows - 1)
                    row = current_rows[row_idx]
                    if len(row) <= 1:
                        continue
                    pos = random.randint(1, len(row) - 1)
                    old_val = row[pos]
                    
                    # Generate a new random value
                    min_val = row[pos - 1] + 1 if pos > 0 else 1
                    max_val = row[pos + 1] - 1 if pos < len(row) - 1 else old_val + 20
                    
                    if min_val > max_val:
                        continue
                    
                    # Try several random values
                    for attempt in range(20):
                        new_val = random.randint(min_val, max_val)
                        if new_val == old_val:
                            continue
                        if is_valid_move(current_rows, row_idx, pos, new_val):
                            current_rows[row_idx][pos] = new_val
                            current_scope = get_scope(current_rows)
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_rows = [r[:] for r in current_rows]
                            break
        # End iteration loop
        
        return best_rows, best_scope

    def construct_rows_greedy_with_bound(used_diffs, max_scope, n_rows_to_build):
        """
        Construct n_rows_to_build rows using greedy approach, trying to keep scope <= max_scope.
        Returns (rows, used_diffs) or None if failed.
        """
        rows = []
        for _ in range(n_rows_to_build):
            row = [0]
            current_max = 0
            success = True
            while len(row) < k + 1:
                x = current_max + 1
                found = False
                while x <= max_scope:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        current_max = x
                        used_diffs |= new_diffs
                        found = True
                        break
                    x += 1
                if not found:
                    success = False
                    break
            if not success:
                return None
            rows.append(row)
        return rows, used_diffs

    # Main optimization loop using ILS with multiple restarts
    best_found = best_scope
    
    # Phase 1: Try to construct rows with a tighter bound than greedy
    # The greedy scope is our initial upper bound
    for target_scope in range(best_found - 1, max(best_found - 30, 0), -1):
        if time.time() > deadline - 15.0:
            break
        
        # Try multiple random orderings of row construction
        for attempt in range(3):
            if time.time() > deadline - 12.0:
                break
            
            used_diffs = set()
            rows = []
            success = True
            
            # Randomize the order in which we try to place elements
            for row_idx in range(n_rows):
                if time.time() > deadline - 10.0:
                    success = False
                    break
                
                row = [0]
                current_max = 0
                while len(row) < k + 1:
                    x = current_max + 1
                    found = False
                    while x <= target_scope:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used_diffs:
                            row.append(x)
                            current_max = x
                            used_diffs |= new_diffs
                            found = True
                            break
                        x += 1
                    if not found:
                        success = False
                        break
                
                if not success:
                    break
                rows.append(row)
            
            if success:
                # We found a valid construction with scope <= target_scope
                actual_scope = get_scope(rows)
                if actual_scope < best_found:
                    best_found = actual_scope
                    best_rows = [r[:] for r in rows]
                # Now try to improve with local search
                if time.time() > deadline - 12.0:
                    continue
                
                improved_rows, improved_scope = local_search(rows, max_iterations=1500)
                if improved_scope < best_found:
                    best_found = improved_scope
                    best_rows = [r[:] for r in improved_rows]
                break  # No need to try other attempts for this target_scope
    
    # Phase 2: Run ILS on the best solution found so far
    if time.time() < deadline - 20.0:
        # Run multiple ILS rounds
        num_ils_rounds = 5
        for round_idx in range(num_ils_rounds):
            if time.time() > deadline - 10.0:
                break
            
            # Start from best found
            start_rows = [r[:] for r in best_rows]
            
            # Run local search with more iterations
            improved_rows, improved_scope = local_search(start_rows, max_iterations=3000)
            
            if improved_scope < best_found:
                best_found = improved_scope
                best_rows = [r[:] for r in improved_rows]
            else:
                # No improvement; apply a stronger kick and retry
                if time.time() > deadline - 10.0:
                    break
                
                # Perturb the best solution more aggressively
                perturbed = [r[:] for r in best_rows]
                n_perturb = random.randint(3, 6)
                for _ in range(n_perturb):
                    row_idx = random.randint(0, n_rows - 1)
                    row = perturbed[row_idx]
                    if len(row) <= 1:
                        continue
                    pos = random.randint(1, len(row) - 1)
                    old_val = row[pos]
                    min_val = row[pos - 1] + 1 if pos > 0 else 1
                    max_val = row[pos + 1] - 1 if pos < len(row) - 1 else old_val + 30
                    if min_val > max_val:
                        continue
                    for attempt in range(30):
                        new_val = random.randint(min_val, max_val)
                        if new_val == old_val:
                            continue
                        if is_valid_move(perturbed, row_idx, pos, new_val):
                            perturbed[row_idx][pos] = new_val
                            break
                
                # Run local search on perturbed solution
                improved_rows, improved_scope = local_search(perturbed, max_iterations=2000)
                if improved_scope < best_found:
                    best_found = improved_scope
                    best_rows = [r[:] for r in improved_rows]
    
    # Phase 3: Final polish with more iterations if time allows
    if time.time() < deadline - 15.0 and best_found < greedy_scope:
        final_rows, final_scope = local_search(best_rows, max_iterations=5000)
        if final_scope < best_found:
            best_found = final_scope
            best_rows = [r[:] for r in final_rows]
    
    # Return the best solution
    if best_found >= greedy_scope:
        return greedy_rows
    
    return best_rows
