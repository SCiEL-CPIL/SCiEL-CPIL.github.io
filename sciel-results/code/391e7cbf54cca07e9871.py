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

def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 increasing integers."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_rows = None
    best_scope = float('inf')
    
    # 1. Generate diverse initial solutions using interleaved greedy
    # We reserve time for optimization.
    # Let's try 5-10 seeds for greedy.
    max_greedy_seeds = 10
    
    for seed in range(max_greedy_seeds):
        if time.time() > deadline - 60.0: # Leave 60s for optimization
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
    
    time_remaining = deadline - time.time()
    
    if time_remaining > 10.0:
        # Run Hill Climbing
        # Use 50% of remaining time for HC, 50% for SA?
        # Or run HC, then SA on the result?
        # SA can escape local minima found by HC.
        
        # Let's run HC first.
        t_hc = time_remaining * 0.5
        if t_hc > 5.0:
            optimized_hc = _optimize(best_rows, n_rows, k, time_limit=t_hc, seed=42)
            sc_hc = max(max(r) for r in optimized_hc)
            if sc_hc < best_scope:
                best_scope = sc_hc
                best_rows = optimized_hc
        
        # Run SA on the current best (which might be improved by HC)
        t_sa = deadline - time.time() - 5.0 # Leave 5s buffer
        if t_sa > 5.0:
            optimized_sa = _optimize_simulated_annealing(best_rows, n_rows, k, time_limit=t_sa, seed=43)
            sc_sa = max(max(r) for r in optimized_sa)
            if sc_sa < best_scope:
                best_scope = sc_sa
                best_rows = optimized_sa
                
    return best_rows
# EVOLVE-BLOCK-END
