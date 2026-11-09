"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _interleaved_greedy(n_rows, k, seed=0):
    """Interleaved greedy: build all rows simultaneously, one element at a time."""
    random.seed(seed)
    rows = [[0] for _ in range(n_rows)]
    used = set()
    
    # Build up to k+1 elements in each row
    for step in range(1, k + 1):
        # Shuffle row order for diversity
        order = list(range(n_rows))
        random.shuffle(order)
        for r in order:
            row = rows[r]
            x = row[-1] + 1
            while True:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    break
    
    return rows

def _construct_from_gaps(n_rows, k, gap_list):
    """
    Construct rows from a list of gaps. Each row has k gaps between k+1 elements.
    gap_list[r][g] is the gap between element g and g+1 in row r.
    """
    rows = []
    for r in range(n_rows):
        row = [0]
        for g in range(k):
            row.append(row[-1] + gap_list[r][g])
        rows.append(row)
    return rows

def _validate_rows(rows, n_rows, k):
    """Check if all differences are distinct across all rows."""
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _optimize(rows, n_rows, k, time_limit=100.0, seed=42):
    """
    Optimized local search using difference counts.
    Returns the best rows found.
    """
    random.seed(seed)
    start_time = time.time()
    
    current_rows = [list(r) for r in rows]
    
    # Helper to compute current scope
    def get_scope():
        return max(max(r) for r in current_rows)
    
    # Initialize difference counts
    max_val = max(max(r) for r in current_rows)
    diff_size = max_val + 20
    diff_counts = [0] * (diff_size + 1)
    
    # Populate initial counts
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()
    
    def try_move(r, pos, new_val):
        """Attempt to move current_rows[r][pos] to new_val. Returns True if successful."""
        row = current_rows[r]
        old_val = row[pos]
        
        # Check increasing constraint
        if pos > 0 and new_val <= row[pos - 1]:
            return False
        if pos < len(row) - 1 and new_val >= row[pos + 1]:
            return False
        
        # Calculate old and new differences
        old_diffs = []
        new_diffs = []
        for i in range(len(row)):
            if i == pos:
                continue
            old_diffs.append(abs(old_val - row[i]))
            new_diffs.append(abs(new_val - row[i]))
        
        # Remove old diffs
        for d in old_diffs:
            diff_counts[d] -= 1
            
        # Check new diffs
        valid = True
        for d in new_diffs:
            if d >= len(diff_counts):
                 pass
            if diff_counts[d] > 0:
                valid = False
                break
        
        if not valid:
            # Restore old diffs
            for d in old_diffs:
                diff_counts[d] += 1
            return False
        
        # Apply move
        row[pos] = new_val
        for d in new_diffs:
            diff_counts[d] += 1
            
        return True

    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Iterate rows and positions in random order for diversity
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current_rows[r]
            # Try to reduce elements from right to left? Or random?
            # Let's try to reduce the largest elements first as they contribute most to scope.
            # But we must maintain order.
            # Let's just iterate all positions.
            for pos in range(len(row) - 1, 0, -1): # Right to left
                if (time.time() - start_time) > time_limit:
                    break
                old_val = row[pos]
                # Try to reduce to smallest valid value
                # Lower bound: row[pos-1] + 1 (if pos > 0)
                lower_bound = row[pos - 1] + 1 if pos > 0 else 1
                # Upper bound: old_val - 1
                # We want the smallest new_val >= lower_bound such that move is valid.
                
                found = False
                # Try values from lower_bound to old_val - 1
                for new_val in range(lower_bound, old_val):
                    if try_move(r, pos, new_val):
                        improved = True
                        found = True
                        break
                if not found:
                    pass
                    
    # Update best
    current_scope = get_scope()
    if current_scope < best_scope:
        best_scope = current_scope
        best_rows = [list(r) for r in current_rows]
        
    return best_rows

def _optimize_simulated_annealing(rows, n_rows, k, time_limit=100.0, seed=42):
    """
    Simulated annealing local search. Allows worsening moves early on to escape local minima.
    """
    random.seed(seed)
    start_time = time.time()
    
    current_rows = [list(r) for r in rows]
    
    def get_scope():
        return max(max(r) for r in current_rows)
    
    max_val = max(max(r) for r in current_rows)
    diff_size = max_val + 20
    diff_counts = [0] * (diff_size + 1)
    
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()
    current_scope = best_scope
    
    # Annealing parameters
    T = 10.0
    T_min = 0.1
    alpha = 0.99
    
    while (time.time() - start_time) < time_limit:
        if T <= T_min:
            break
            
        r = random.randint(0, n_rows - 1)
        pos = random.randint(1, k) # 1 to k, since pos 0 is always 0
        
        row = current_rows[r]
        old_val = row[pos]
        
        # Generate a random new value within a reasonable range
        # Keep it close to old_val to maintain connectivity
        delta = random.randint(-10, 10)
        new_val = old_val + delta
        
        # Ensure validity of range
        lower_bound = row[pos - 1] + 1
        upper_bound = row[pos + 1] - 1 if pos < len(row) - 1 else old_val + 20
        
        if new_val < lower_bound:
            new_val = lower_bound
        if new_val > upper_bound:
            new_val = upper_bound
            
        if new_val == old_val:
            continue
            
        # Try the move
        # We need a helper that can revert
        old_diffs = []
        new_diffs = []
        for i in range(len(row)):
            if i == pos:
                continue
            old_diffs.append(abs(old_val - row[i]))
            new_diffs.append(abs(new_val - row[i]))
            
        # Check if move is valid (no conflicts)
        # We need to check if new_diffs conflict with existing counts AFTER removing old_diffs
        # This is slightly different from try_move which assumes we want to apply it.
        # Here we just want to know if it's a valid state.
        
        # Temporarily remove old
        for d in old_diffs:
            diff_counts[d] -= 1
            
        valid = True
        for d in new_diffs:
            if diff_counts[d] > 0:
                valid = False
                break
        
        if not valid:
            # Restore old
            for d in old_diffs:
                diff_counts[d] += 1
            continue
            
        # Move is valid. Calculate new scope
        # Apply move
        row[pos] = new_val
        for d in new_diffs:
            diff_counts[d] += 1
            
        new_scope = get_scope()
        
        if new_scope < current_scope:
            # Accept
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        else:
            # Worsening move
            delta_e = new_scope - current_scope
            if random.random() < pow(2.71828, -delta_e / T):
                # Accept
                current_scope = new_scope
            else:
                # Reject, revert
                row[pos] = old_val
                # Revert counts: remove new, add old
                for d in new_diffs:
                    diff_counts[d] -= 1
                for d in old_diffs:
                    diff_counts[d] += 1
                    
        T *= alpha
        
    return best_rows

def _final_polish(rows, n_rows, k, time_limit=10.0):
    """
    Final high-precision polish: repeatedly try to reduce each element to its minimum valid value.
    This is a deterministic pass that can find small improvements missed by stochastic search.
    """
    start_time = time.time()
    current_rows = [list(r) for r in rows]
    
    # Build diff counts
    max_val = max(max(r) for r in current_rows)
    diff_counts = [0] * (max_val + 20)
    
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d >= len(diff_counts):
                    diff_counts.extend([0] * (d - len(diff_counts) + 1))
                diff_counts[d] += 1
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for r in range(n_rows):
            if (time.time() - start_time) > time_limit:
                break
            row = current_rows[r]
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                lower_bound = row[pos - 1] + 1
                if lower_bound >= old_val:
                    continue
                # Try to find the smallest valid value >= lower_bound
                for new_val in range(lower_bound, old_val):
                    # Check if new_val is valid
                    valid = True
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        d = abs(new_val - row[i])
                        # Check if this difference is already used by another pair in any row
                        # We need to be careful: if this difference is used by the same row at another position, it's fine
                        # But if it's used by any other pair, it's not fine.
                        # Actually, the constraint is that ALL differences across ALL rows must be distinct.
                        # So if diff_counts[d] > 0, and that count is not from the current pair (r, i, pos), then it's invalid.
                        # But since we're changing row[pos], the old differences involving row[pos] are currently in diff_counts.
                        # We need to check if new_val - row[i] (or row[i] - new_val) conflicts with any OTHER difference.
                        # The old differences involving row[pos] are: old_val - row[i] for all i != pos.
                        # When we change to new_val, the new differences are new_val - row[i] for all i != pos.
                        # We must ensure that none of the new differences equals any difference that is NOT being removed.
                        # The differences being removed are exactly the old differences involving row[pos].
                        # So we can check: for each i != pos, let d_new = abs(new_val - row[i]).
                        # Let d_old = abs(old_val - row[i]).
                        # We remove d_old from consideration (it's being replaced).
                        # We add d_new.
                        # The conflict is if d_new is present in diff_counts AFTER removing all old differences involving row[pos].
                        # This is complex to check incrementally. Instead, let's do a simple check:
                        # For each i != pos, compute d_new = abs(new_val - row[i]).
                        # Check if there exists any pair (r2, i2, j2) != (r, i, pos) such that row2[j2] - row2[i2] == d_new.
                        # This is O(n^3) which might be too slow.
                        
                        # Alternative: use diff_counts but account for the fact that we're replacing old diffs.
                        # Let's do a temporary check:
                        # 1. Decrement diff_counts for all old differences involving row[pos].
                        # 2. Check if any new difference has diff_counts[d] > 0.
                        # 3. If valid, apply the move.
                        # 4. If not, restore diff_counts.
                        
                        pass # Placeholder, will implement below
                    
                    # Implementation of the check
                    old_diffs = [abs(old_val - row[i]) for i in range(len(row)) if i != pos]
                    new_diffs = [abs(new_val - row[i]) for i in range(len(row)) if i != pos]
                    
                    # Temporarily remove old
                    for d in old_diffs:
                        diff_counts[d] -= 1
                    
                    valid = True
                    for d in new_diffs:
                        if diff_counts[d] > 0:
                            valid = False
                            break
                    
                    if valid:
                        # Apply move
                        row[pos] = new_val
                        for d in new_diffs:
                            diff_counts[d] += 1
                        improved = True
                        break # Move to next position
                    else:
                        # Restore old
                        for d in old_diffs:
                            diff_counts[d] += 1
    
    return current_rows

def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 increasing integers."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_rows = None
    best_scope = float('inf')
    
    # 1. Generate diverse initial solutions using interleaved greedy
    # We reserve time for optimization.
    # Let's try 5-10 seeds for greedy.
    max_greedy_seeds = 15
    
    for seed in range(max_greedy_seeds):
        if time.time() > deadline - 80.0: # Leave 80s for optimization
            break
            
        rows = _interleaved_greedy(n_rows, k, seed=seed)
        sc = max(max(r) for r in rows)
        
        if sc < best_scope:
            best_scope = sc
            best_rows = [list(r) for r in rows]
            
    if best_rows is None:
        # Fallback to simple greedy
        rows = []
        used = set()
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    # 2. Optimization Phase
    # We have best_rows from greedy.
    # We can try to optimize the best one found, or multiple ones.
    # Given the time, let's split remaining time between:
    # a) Hill Climbing on the best solution
    # b) Simulated Annealing on the best solution (or a few top ones)
    # c) Final polish pass
    
    time_remaining = deadline - time.time()
    
    if time_remaining > 10.0:
        # Run Hill Climbing
        # Use 35% of remaining time for HC, 35% for SA, 15% for polish, 15% buffer
        t_hc = time_remaining * 0.35
        if t_hc > 5.0:
            optimized_hc = _optimize(best_rows, n_rows, k, time_limit=t_hc, seed=42)
            sc_hc = max(max(r) for r in optimized_hc)
            if sc_hc < best_scope:
                best_scope = sc_hc
                best_rows = optimized_hc
        
        # Run SA on the current best (which might be improved by HC)
        t_sa = deadline - time.time() - 20.0 # Leave 20s for polish + buffer
        if t_sa > 5.0:
            optimized_sa = _optimize_simulated_annealing(best_rows, n_rows, k, time_limit=t_sa, seed=43)
            sc_sa = max(max(r) for r in optimized_sa)
            if sc_sa < best_scope:
                best_scope = sc_sa
                best_rows = optimized_sa
        
        # Final polish pass
        t_polish = deadline - time.time() - 5.0 # Leave 5s buffer
        if t_polish > 1.0:
            polished = _final_polish(best_rows, n_rows, k, time_limit=t_polish)
            sc_polish = max(max(r) for r in polished)
            if sc_polish < best_scope:
                best_scope = sc_polish
                best_rows = polished
                
    return best_rows
# EVOLVE-BLOCK-END
