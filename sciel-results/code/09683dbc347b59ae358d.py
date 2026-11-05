import random
import time
import math

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    Uses simulated annealing local search: start from a valid greedy construction,
    then iteratively try to reduce the scope by modifying individual rows.
    """
    start_time = time.time()
    time_limit = 145.0
    
    def get_row_diffs(row):
        """Return the set of positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        all_diffs = set()
        for row in rows:
            row_diffs = get_row_diffs(row)
            if row_diffs & all_diffs:
                return False
            all_diffs |= row_diffs
        return True
    
    def scope_of(rows):
        """Return the largest entry across all rows."""
        return max(row[-1] for row in rows)
    
    def greedy_construct(max_val):
        """
        Greedily construct n_rows rows with max entry <= max_val.
        Returns None if impossible.
        """
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                found = False
                for x in range(row[-1] + 1, max_val + 1):
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        row.append(x)
                        used_diffs |= new_diffs
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
        return rows
    
    # Phase 1: Find initial valid solution via greedy with increasing scope
    initial_rows = None
    for scope in range(105, 200):
        initial_rows = greedy_construct(scope)
        if initial_rows is not None:
            break
    
    if initial_rows is None:
        # Fallback: very large scope
        used = set()
        rows = []
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
    
    best_rows = [list(r) for r in initial_rows]
    best_scope = scope_of(best_rows)
    
    # Phase 2: Simulated annealing to reduce scope
    def try_modify_row(rows, row_idx, target_scope):
        """
        Try to replace row_idx with a new row that fits within target_scope
        and doesn't conflict with other rows' differences.
        Returns new row or None.
        """
        other_diffs = set()
        for i, row in enumerate(rows):
            if i == row_idx:
                continue
            other_diffs |= get_row_diffs(row)
        
        # Try to find a valid row starting at 0 with max entry <= target_scope
        row = [0]
        for _ in range(k):
            found = False
            # Try candidates in random order for diversity
            candidates = list(range(row[-1] + 1, target_scope + 1))
            random.shuffle(candidates)
            for x in candidates:
                new_diffs = {x - a for a in row}
                if not (new_diffs & other_diffs) and not (new_diffs & get_row_diffs(row)):
                    row.append(x)
                    other_diffs |= new_diffs
                    found = True
                    break
            if not found:
                return None
        return row
    
    # Phase 3: Local search with restarts
    rng = random.Random(42)
    
    for restart in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Start from current best
        current_rows = [list(r) for r in best_rows]
        current_scope = best_scope
        
        # Simulated annealing schedule
        temp_start = 5.0
        temp_end = 0.1
        n_steps = 5000
        
        for step in range(n_steps):
            if time.time() - start_time > time_limit:
                break
            
            temp = temp_start * (temp_end / temp_start) ** (step / n_steps)
            
            # Randomly pick a row to modify
            row_idx = rng.randint(0, n_rows - 1)
            
            # Try to reduce this row's max entry
            current_row_max = current_rows[row_idx][-1]
            
            # Target: slightly smaller than current
            target = current_row_max - rng.randint(1, 3)
            if target < k:  # minimum possible is k (0,1,2,...,k)
                target = k
            
            new_row = try_modify_row(current_rows, row_idx, target)
            
            if new_row is not None:
                # Check validity
                old_row = current_rows[row_idx]
                current_rows[row_idx] = new_row
                
                if is_valid(current_rows):
                    new_scope = scope_of(current_rows)
                    if new_scope < current_scope:
                        current_scope = new_scope
                        # Accept
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [list(r) for r in current_rows]
                            if best_scope <= 105:
                                return best_rows
                    else:
                        # Accept with probability based on temperature
                        if rng.random() < 0.1:  # Low probability to accept worse solutions
                            # Keep the new row even if scope didn't improve (for exploration)
                            pass
                else:
                    # Revert
                    current_rows[row_idx] = old_row
            else:
                # Try a different strategy: modify a random position in the row
                if k >= 1:
                    pos = rng.randint(1, k)
                    old_val = current_rows[row_idx][pos]
                    
                    # Try nearby values
                    for delta in [-2, -1, 1, 2, -3, 3]:
                        new_val = old_val + delta
                        if new_val <= current_rows[row_idx][pos - 1]:
                            continue
                        if pos < k and new_val >= current_rows[row_idx][pos + 1]:
                            continue
                        if new_val > current_scope:
                            continue
                        
                        test_row = list(current_rows[row_idx])
                        test_row[pos] = new_val
                        
                        # Check if row is still strictly increasing
                        if any(test_row[i] >= test_row[i+1] for i in range(len(test_row)-1)):
                            continue
                        
                        # Check validity
                        current_rows[row_idx] = test_row
                        if is_valid(current_rows):
                            new_scope = scope_of(current_rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best_rows = [list(r) for r in current_rows]
                                if best_scope <= 105:
                                    return best_rows
                            break
                        else:
                            current_rows[row_idx] = old_val
                    else:
                        pass
        
        # Occasionally restart from a fresh greedy solution
        if restart % 10 == 0 and restart > 0:
            fresh_scope = best_scope + rng.randint(0, 5)
            fresh = greedy_construct(fresh_scope)
            if fresh is not None:
                current_rows = fresh
    
    return best_rows
