"""Rewritten program for difference triangle set (7,5) with structural change:
constraint-based backtracking with forward checking, plus local search refinement."""
import time
import random
import numpy as np

def _build_constraint_table(n_rows, k, M):
    """Precompute for each row r, position pos, and value v, the set of differences it would use."""
    # diff_conflict[r][pos][v] = list of (other_row, other_pos, other_val) that conflict
    # This is too large. Instead, we'll use a different approach.
    pass

def _backtracking_construct(n_rows, k, M, time_limit=30.0, seed=0):
    """
    Constraint-based backtracking: build rows one element at a time,
    using forward checking to prune invalid choices early.
    """
    random.seed(seed)
    start_time = time.time()
    
    # We'll build rows sequentially. For each row, we build it element by element.
    # used_diffs: set of all differences used so far
    used_diffs = set()
    rows = []
    
    def backtrack(row_idx, pos, current_row):
        """
        Build row row_idx, element at position pos.
        current_row has elements 0..pos-1 already set.
        Returns True if a valid completion is found, False otherwise.
        """
        nonlocal used_diffs, rows
        
        if time.time() - start_time > time_limit:
            return False
        
        if pos > k:
            return True
        
        if pos == 0:
            # First element is always 0
            return backtrack(row_idx, 1, [0])
        
        # Determine range for current element
        low = current_row[-1] + 1
        high = M
        
        # Try values in a smart order: start from low, but also try some random values
        # to escape local patterns
        candidates = list(range(low, high + 1))
        
        # Add some random candidates for diversity
        for _ in range(5):
            candidates.append(random.randint(low, high))
        
        # Sort: prefer smaller values (to keep scope small), but mix in random
        candidates.sort()
        
        # Shuffle the larger portion for diversity
        if len(candidates) > 10:
            # Keep first 10 sorted, shuffle the rest
            rest = candidates[10:]
            random.shuffle(rest)
            candidates = candidates[:10] + rest
        
        for val in candidates:
            if time.time() - start_time > time_limit:
                return False
            
            # Check if this value creates any conflicting differences
            new_diffs = set()
            valid = True
            for i in range(pos):
                d = val - current_row[i]
                if d <= 0 or d in used_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            # Forward check: can the remaining positions be filled?
            # For each remaining position, check if there's at least one valid value
            feasible = True
            for future_pos in range(pos + 1, k + 1):
                # The value at future_pos must be > val (or > current_row[-1] if future_pos is next)
                # Minimum possible value at future_pos
                if future_pos == pos + 1:
                    min_next = val + 1
                else:
                    # We don't know the exact value, but it must be > previous
                    # For a rough check, just ensure there's enough room
                    min_next = val + 1  # Conservative lower bound
                
                # Maximum possible value
                max_next = M
                
                if min_next > max_next:
                    feasible = False
                    break
                
                # Check if there exists at least one value in [min_next, max_next] 
                # that doesn't conflict with used_diffs or new_diffs
                # This is expensive, so we do a simplified check
                # Just check that the range is non-empty
                # A better check: for the immediate next position, verify at least one valid value
                if future_pos == pos + 1:
                    found_valid = False
                    for test_val in range(min_next, min(min_next + 20, max_next + 1)):
                        test_diffs = set()
                        ok = True
                        for i in range(pos + 1):
                            d = test_val - current_row[i] if i < pos else test_val - val
                            # Actually, for i < pos, current_row[i] is known
                            # For i == pos, the value is val
                            if i < pos:
                                d = test_val - current_row[i]
                            else:
                                d = test_val - val
                            if d <= 0 or d in used_diffs or d in new_diffs or d in test_diffs:
                                ok = False
                                break
                            test_diffs.add(d)
                        if ok:
                            found_valid = True
                            break
                    if not found_valid:
                        feasible = False
                        break
            
            if not feasible:
                continue
            
            # Accept this value
            new_row = current_row + [val]
            old_used = used_diffs.copy()
            used_diffs.update(new_diffs)
            
            if backtrack(row_idx, pos + 1, new_row):
                return True
            
            # Backtrack
            used_diffs = old_used
        
        return False
    
    # Build each row
    for r in range(n_rows):
        if time.time() - start_time > time_limit:
            # Return partial result
            break
        
        success = backtrack(r, 0, [])
        if not success:
            # Try with a larger M or different seed
            return None
    
    return rows

def _local_search_reduce(rows, n_rows, k, time_limit=20.0, seed=42):
    """
    Local search: repeatedly try to reduce individual elements to minimize scope.
    Uses a difference-count array for O(1) conflict checking.
    """
    random.seed(seed)
    start_time = time.time()
    
    current_rows = [list(r) for r in rows]
    
    # Build difference count array
    max_val = max(max(r) for r in current_rows)
    diff_size = max_val + 50
    diff_counts = [0] * (diff_size + 1)
    
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d <= diff_size:
                    diff_counts[d] += 1
    
    def get_scope():
        return max(max(r) for r in current_rows)
    
    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        
        # Try each row, each position (from right to left)
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
                found = False
                for new_val in range(lower_bound, old_val):
                    # Check if new_val is valid
                    # Compute old and new differences
                    old_diffs = []
                    new_diffs = []
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        old_diffs.append(abs(old_val - row[i]))
                        new_diffs.append(abs(new_val - row[i]))
                    
                    # Temporarily remove old differences
                    for d in old_diffs:
                        if d <= diff_size:
                            diff_counts[d] -= 1
                    
                    # Check new differences
                    valid = True
                    for d in new_diffs:
                        if d > diff_size:
                            valid = False
                            break
                        if diff_counts[d] > 0:
                            valid = False
                            break
                    
                    if valid:
                        # Apply move
                        row[pos] = new_val
                        for d in new_diffs:
                            if d <= diff_size:
                                diff_counts[d] += 1
                        improved = True
                        found = True
                        break
                    else:
                        # Restore old differences
                        for d in old_diffs:
                            if d <= diff_size:
                                diff_counts[d] += 1
                
                if found:
                    # Update best if scope improved
                    cur_scope = get_scope()
                    if cur_scope < best_scope:
                        best_scope = cur_scope
                        best_rows = [list(r) for r in current_rows]
    
    return best_rows

def _simulated_annealing(rows, n_rows, k, time_limit=30.0, seed=43):
    """
    Simulated annealing to escape local minima.
    """
    random.seed(seed)
    start_time = time.time()
    
    current_rows = [list(r) for r in rows]
    
    max_val = max(max(r) for r in current_rows)
    diff_size = max_val + 50
    diff_counts = [0] * (diff_size + 1)
    
    for row in current_rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d <= diff_size:
                    diff_counts[d] += 1
    
    def get_scope():
        return max(max(r) for r in current_rows)
    
    best_rows = [list(r) for r in current_rows]
    best_scope = get_scope()
    current_scope = best_scope
    
    # Annealing parameters
    T = 5.0
    T_min = 0.1
    alpha = 0.995
    
    while (time.time() - start_time) < time_limit and T > T_min:
        r = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        row = current_rows[r]
        old_val = row[pos]
        
        # Generate random perturbation
        delta = random.randint(-5, 5)
        new_val = old_val + delta
        
        # Ensure valid range
        lower_bound = row[pos - 1] + 1
        upper_bound = row[pos + 1] - 1 if pos < len(row) - 1 else old_val + 10
        
        if new_val < lower_bound:
            new_val = lower_bound
        if new_val > upper_bound:
            new_val = upper_bound
        
        if new_val == old_val:
            T *= alpha
            continue
        
        # Compute old and new differences
        old_diffs = []
        new_diffs = []
        for i in range(len(row)):
            if i == pos:
                continue
            old_diffs.append(abs(old_val - row[i]))
            new_diffs.append(abs(new_val - row[i]))
        
        # Temporarily remove old
        for d in old_diffs:
            if d <= diff_size:
                diff_counts[d] -= 1
        
        # Check validity
        valid = True
        for d in new_diffs:
            if d > diff_size:
                valid = False
                break
            if diff_counts[d] > 0:
                valid = False
                break
        
        if not valid:
            for d in old_diffs:
                if d <= diff_size:
                    diff_counts[d] += 1
            T *= alpha
            continue
        
        # Apply move
        row[pos] = new_val
        for d in new_diffs:
            if d <= diff_size:
                diff_counts[d] += 1
        
        new_scope = get_scope()
        
        if new_scope < current_scope:
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [list(r) for r in current_rows]
        else:
            delta_e = new_scope - current_scope
            if random.random() < pow(2.71828, -delta_e / T):
                current_scope = new_scope
            else:
                # Revert
                row[pos] = old_val
                for d in new_diffs:
                    if d <= diff_size:
                        diff_counts[d] -= 1
                for d in old_diffs:
                    if d <= diff_size:
                        diff_counts[d] += 1
        
        T *= alpha
    
    return best_rows

def _greedy_construct(n_rows, k, seed=0):
    """Simple greedy construction as a fallback."""
    random.seed(seed)
    rows = [[0] for _ in range(n_rows)]
    used = set()
    
    for step in range(1, k + 1):
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

def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 increasing integers."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Try backtracking construction with various M values and seeds
    # The counting bound suggests M >= 105 for (7,5)
    M_candidates = [110, 112, 115, 120, 125, 130]
    
    for M in M_candidates:
        if time.time() > deadline - 60.0:
            break
        
        for seed in range(3):
            if time.time() > deadline - 60.0:
                break
            
            rows = _backtracking_construct(n_rows, k, M, time_limit=15.0, seed=seed)
            if rows is not None and len(rows) == n_rows and all(len(r) == k + 1 for r in rows):
                sc = max(max(r) for r in rows)
                if sc < best_scope:
                    best_scope = sc
                    best_rows = [list(r) for r in rows]
    
    # Phase 2: If backtracking didn't find a good solution, use greedy as fallback
    if best_rows is None or best_scope > 150:
        for seed in range(20):
            if time.time() > deadline - 50.0:
                break
            
            rows = _greedy_construct(n_rows, k, seed=seed)
            sc = max(max(r) for r in rows)
            if sc < best_scope:
                best_scope = sc
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Last resort: simple sequential greedy
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
    
    # Phase 3: Local search refinement
    time_remaining = deadline - time.time()
    
    if time_remaining > 15.0:
        t_ls = min(25.0, time_remaining * 0.4)
        if t_ls > 5.0:
            optimized = _local_search_reduce(best_rows, n_rows, k, time_limit=t_ls, seed=42)
            sc = max(max(r) for r in optimized)
            if sc < best_scope:
                best_scope = sc
                best_rows = optimized
    
    # Phase 4: Simulated annealing
    time_remaining = deadline - time.time()
    if time_remaining > 15.0:
        t_sa = min(30.0, time_remaining * 0.5)
        if t_sa > 5.0:
            optimized = _simulated_annealing(best_rows, n_rows, k, time_limit=t_sa, seed=43)
            sc = max(max(r) for r in optimized)
            if sc < best_scope:
                best_scope = sc
                best_rows = optimized
    
    # Phase 5: Final local search polish
    time_remaining = deadline - time.time()
    if time_remaining > 5.0:
        t_final = min(10.0, time_remaining - 5.0)
        if t_final > 1.0:
            polished = _local_search_reduce(best_rows, n_rows, k, time_limit=t_final, seed=44)
            sc = max(max(r) for r in polished)
            if sc < best_scope:
                best_scope = sc
                best_rows = polished
    
    return best_rows
