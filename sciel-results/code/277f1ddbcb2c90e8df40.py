# EVOLVE-BLOCK-START
import sys
import time
import random

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a randomized CSP approach with bitset-based difference constraint propagation.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound: need n_rows * k*(k+1)/2 distinct positive differences
    total_diffs_needed = n_rows * k * (k + 1) // 2
    
    # Lower bound: We need 'total_diffs_needed' distinct differences.
    # The differences are integers from 1 to M.
    # So M must be at least total_diffs_needed.
    lb = total_diffs_needed
    
    # Greedy baseline
    def greedy_construct(n_rows, k):
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
    
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    start_time = time.time()
    time_limit = 115.0
    
    # Bitset-based solver using Python integers as bitsets
    # Bit i is set if difference i is used
    
    def solve_csp(n_rows, k, max_entry, timeout, seed=0):
        """
        CSP solver using bitset difference tracking and randomized backtracking
        with dynamic row ordering heuristic.
        Returns list of rows or None.
        """
        deadline = time.time() + timeout
        used_bits = 0  # bitset of used differences
        rows = [None] * n_rows
        rng = random.Random(seed)
        
        # Track which rows are completed
        completed = [False] * n_rows
        # Track the last element of each row (for efficiency)
        row_last = [0] * n_rows
        # Track the count of elements in each row
        row_len = [1] * n_rows  # all start with [0]
        
        def get_row_diffs(row):
            """Compute all differences in a row as a bitset."""
            bits = 0
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    bits |= (1 << (row[j] - row[i]))
            return bits
        
        def build_row(row_idx, used_bits, deadline):
            """
            Build a single row starting from its current state.
            Returns True if the row is completed successfully.
            """
            if time.time() > deadline:
                return False
            
            current_row = [0] if row_len[row_idx] == 1 else None
            # We need to store the current row state. Let's use a list.
            # For simplicity, let's rebuild from scratch each time we backtrack to a row start.
            # But that's inefficient. Let's use a different approach.
            
            # Actually, let's just build the row element by element.
            # We'll maintain the current row as a list.
            
            # This function is called when we want to extend row row_idx.
            # The row currently has row_len[row_idx] elements.
            
            if row_len[row_idx] == k + 1:
                completed[row_idx] = True
                return True
            
            # Get the current row (we need to store it somewhere)
            # For now, let's assume we have a way to get it.
            # This is getting complex. Let's simplify.
            
            return False
        
        # Simplified approach: build rows one at a time, but choose which row to build next
        # based on a heuristic (tightest remaining capacity)
        
        def build_full_row(start_row, used_bits, deadline):
            """
            Build a complete row starting from [0].
            Returns the completed row or None.
            """
            if time.time() > deadline:
                return None
            
            current_row = [0]
            
            def extend_row(current_row, used_bits, deadline):
                if time.time() > deadline:
                    return None
                
                if len(current_row) == k + 1:
                    return current_row
                
                last = current_row[-1]
                remaining = k + 1 - len(current_row)
                
                min_x = last + 1
                max_x = max_entry - (remaining - 1)
                
                if min_x > max_x:
                    return None
                
                # Generate candidates with early pruning
                candidates = []
                for x in range(min_x, max_x + 1):
                    # Check if x conflicts with used bits
                    conflict = False
                    new_diff_bits = 0
                    for a in current_row:
                        d = x - a
                        if (used_bits >> d) & 1:
                            conflict = True
                            break
                        new_diff_bits |= (1 << d)
                    
                    if conflict:
                        continue
                    
                    # Heuristic score: prefer x that leaves more room for future elements
                    # and uses less "dense" differences
                    # Simple heuristic: prefer smaller x (compact rows)
                    # But also consider how many new small differences this creates
                    candidates.append((x, new_diff_bits))
                
                if not candidates:
                    return None
                
                # Shuffle for randomness
                rng.shuffle(candidates)
                
                for x, new_diff_bits in candidates:
                    if time.time() > deadline:
                        return None
                    
                    current_row.append(x)
                    new_used = used_bits | new_diff_bits
                    
                    result = extend_row(current_row, new_used, deadline)
                    if result is not None:
                        return result
                    
                    current_row.pop()
                
                return None
            
            return extend_row(current_row, used_bits, deadline)
        
        # Main loop: build rows one by one, choosing the "hardest" row first
        # Heuristic: rows with fewer available candidates are more constrained
        # For simplicity, we build in order but with randomization
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            # Try to build this row
            # Use multiple attempts with different random seeds if needed
            success = False
            for attempt in range(5):  # Try up to 5 times with different randomization
                if time.time() > deadline:
                    return None
                
                # Reset random seed for this attempt
                attempt_rng = random.Random(seed * 1000 + row_idx * 100 + attempt)
                
                # Build the row
                row = build_full_row(row_idx, used_bits, deadline)
                if row is not None:
                    # Update used_bits
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            used_bits |= (1 << (row[j] - row[i]))
                    rows[row_idx] = row
                    success = True
                    break
            
            if not success:
                return None
        
        return rows

    # Try to find better solution than greedy
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Strategy: try max_entry values from lb upward
    search_upper_bound = greedy_max
    
    for max_entry in range(lb, search_upper_bound + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2  # leave 2 sec safety
        
        if remaining_time < 0.5:
            break
        
        # Allocate time: more time for values closer to lb
        distance_from_lb = max_entry - lb
        if distance_from_lb == 0:
            budget = min(15.0, remaining_time)
        elif distance_from_lb <= 5:
            budget = min(8.0, remaining_time)
        elif distance_from_lb <= 15:
            budget = min(4.0, remaining_time)
        else:
            budget = min(2.0, remaining_time)
        
        if budget < 0.1:
            continue
        
        # Try multiple restarts with different seeds
        found = False
        num_restarts = max(1, int(budget / 0.5))
        
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            
            # Give each restart a portion of the budget
            restart_budget = budget / (num_restarts - seed) if num_restarts > 1 else budget
            restart_budget = min(restart_budget, 2.0)  # Cap per-restart time
            
            result = solve_csp(n_rows, k, max_entry, restart_budget, seed=seed)
            if result is not None:
                best_rows = result
                best_max = max_entry
                found = True
                break
        
        if found:
            break
    
    # Validate
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows

# EVOLVE-BLOCK-END
