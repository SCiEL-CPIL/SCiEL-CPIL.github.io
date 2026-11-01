"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0
    with all intra-row differences distinct across all rows, minimizing scope."""
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    best_result = None
    best_scope = float('inf')
    
    def get_scope(rows):
        return max(max(row) for row in rows)
    
    def is_valid(rows):
        """Check if all differences are distinct across all rows."""
        used = set()
        for row in rows:
            diffs = set()
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used or d in diffs:
                        return False
                    diffs.add(d)
            used |= diffs
        return True
    
    def greedy_construct(n_rows, k, randomize=False, seed=None):
        """Greedy construction with optional randomization."""
        if seed is not None:
            random.seed(seed)
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
        
        if randomize:
            # Try to improve by rebuilding some rows with different choices
            for attempt in range(20):
                if time.time() > deadline - 10:
                    break
                # Pick a row to rebuild (bias toward max-value rows)
                max_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
                if random.random() < 0.6:
                    idx = max_idx
                else:
                    idx = random.randint(0, len(rows) - 1)
                
                # Remove this row's differences from used
                old_row = rows[idx]
                old_diffs = set()
                for i in range(len(old_row)):
                    for j in range(i+1, len(old_row)):
                        old_diffs.add(old_row[j] - old_row[i])
                used -= old_diffs
                
                # Try to rebuild with smaller max
                old_max = old_row[-1]
                target_max = old_max - random.randint(1, min(5, old_max // 4 + 1))
                
                new_row = _build_row(k, used, target_max)
                if new_row is not None and new_row[-1] < old_max:
                    rows[idx] = new_row
                    used |= {new_row[j] - new_row[i] for i in range(len(new_row)) for j in range(i+1, len(new_row))}
        
        return rows
    
    def _build_row(k, used, max_val):
        """Try to build a row of k+1 elements starting at 0 with max <= max_val."""
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            if x > max_val:
                return None
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
        return row
    
    def local_search_minimize(rows, n_rows, k, time_limit):
        """Local search to minimize the scope."""
        if not rows:
            return rows
        
        current_scope = get_scope(rows)
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Find the row with the largest max value
            max_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
            old_row = rows[max_idx]
            old_max = old_row[-1]
            
            # Try to reduce the max value of this row
            for delta in range(1, min(10, old_max // 3 + 1)):
                if time.time() > time_limit:
                    break
                target_max = old_max - delta
                
                # Remove old row's differences
                old_diffs = set()
                for i in range(len(old_row)):
                    for j in range(i+1, len(old_row)):
                        old_diffs.add(old_row[j] - old_row[i])
                used -= old_diffs
                
                # Try to build a new row with smaller max
                new_row = _build_row(k, used, target_max)
                if new_row is not None:
                    # Add new differences
                    new_diffs = set()
                    for i in range(len(new_row)):
                        for j in range(i+1, len(new_row)):
                            new_diffs.add(new_row[j] - new_row[i])
                    used |= new_diffs
                    rows[max_idx] = new_row
                    improved = True
                    break
                else:
                    # Restore old differences
                    used |= old_diffs
            
            # Also try swapping elements in rows to reduce conflicts
            if time.time() > time_limit:
                break
            
            # Try local moves: adjust one element in a row
            for _ in range(5):
                if time.time() > time_limit:
                    break
                idx = random.randint(0, len(rows) - 1)
                row = rows[idx]
                
                # Try to reduce a non-first element
                for pos in range(len(row) - 1, 0, -1):
                    if time.time() > time_limit:
                        break
                    old_val = row[pos]
                    # Try smaller values
                    for new_val in range(old_val - 1, max(0, old_val - 5), -1):
                        if new_val <= row[pos-1]:
                            break
                        
                        # Check if this creates conflicts
                        new_diffs = set()
                        for i in range(len(row)):
                            if i == pos:
                                continue
                            if i < pos:
                                new_diffs.add(new_val - row[i])
                            else:
                                new_diffs.add(row[i] - new_val)
                        
                        # Also need to check differences involving position pos
                        # Actually, we need to recompute all diffs for this row
                        test_row = row.copy()
                        test_row[pos] = new_val
                        test_diffs = set()
                        for i in range(len(test_row)):
                            for j in range(i+1, len(test_row)):
                                test_diffs.add(test_row[j] - test_row[i])
                        
                        # Check against used (excluding this row's current diffs)
                        row_diffs = set()
                        for i in range(len(row)):
                            for j in range(i+1, len(row)):
                                row_diffs.add(row[j] - row[i])
                        other_used = used - row_diffs
                        
                        if not (test_diffs & other_used):
                            # Accept the move
                            used = other_used | test_diffs
                            row[pos] = new_val
                            improved = True
                            break
                    if improved:
                        break
                
                if improved:
                    break
        
        return rows
    
    # Phase 1: Multiple greedy constructions with different seeds
    num_greedy = 10
    for i in range(num_greedy):
        if time.time() > deadline - 30:
            break
        rows = greedy_construct(n_rows, k, randomize=True, seed=i*1000 + 42)
        if is_valid(rows):
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_result = [row[:] for row in rows]
    
    # If no valid result yet, use pure greedy
    if best_result is None:
        rows = greedy_construct(n_rows, k, randomize=False)
        if is_valid(rows):
            best_scope = get_scope(rows)
            best_result = [row[:] for row in rows]
        else:
            # Fallback: return whatever we have
            best_result = rows
    
    # Phase 2: Local search to minimize scope
    remaining_time = deadline - time.time() - 5
    if remaining_time > 5 and best_result is not None:
        rows_copy = [row[:] for row in best_result]
        rows_copy = local_search_minimize(rows_copy, n_rows, k, time.time() + remaining_time)
        if is_valid(rows_copy):
            scope = get_scope(rows_copy)
            if scope < best_scope:
                best_scope = scope
                best_result = [row[:] for row in rows_copy]
    
    # Phase 3: More randomized restarts if time remains
    remaining_time = deadline - time.time() - 10
    if remaining_time > 10 and best_result is not None:
        for i in range(20):
            if time.time() > deadline - 10:
                break
            rows = greedy_construct(n_rows, k, randomize=True, seed=10000 + i*777)
            if is_valid(rows):
                scope = get_scope(rows)
                if scope < best_scope:
                    best_scope = scope
                    best_result = [row[:] for row in rows]
                    # Quick local search
                    if time.time() < deadline - 15:
                        rows_ls = [row[:] for row in best_result]
                        rows_ls = local_search_minimize(rows_ls, n_rows, k, time.time() + min(5, deadline - time.time() - 15))
                        if is_valid(rows_ls):
                            ls_scope = get_scope(rows_ls)
                            if ls_scope < best_scope:
                                best_scope = ls_scope
                                best_result = [row[:] for row in rows_ls]
    
    # Final deadline check
    if time.time() > deadline - 2:
        pass  # Return whatever we have
    
    return best_result if best_result is not None else greedy_construct(n_rows, k)

# EVOLVE-BLOCK-END
