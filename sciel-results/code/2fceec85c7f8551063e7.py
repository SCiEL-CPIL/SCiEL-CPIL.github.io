import random
import time

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110.0  # Safety margin

    def elapsed():
        return time.time() - start_time

    def time_ok():
        return elapsed() < time_limit

    def row_diffs(row):
        n = len(row)
        return {row[j] - row[i] for i in range(n) for j in range(i + 1, n)}

    def greedy_row(used_diffs, k, rng, bias=1.0):
        """
        Greedily construct a row of length k+1 starting at 0.
        'bias' controls how aggressively we skip candidates. 
        Returns the row or None if stuck (shouldn't happen if scope is large enough).
        """
        row = [0]
        for pos in range(1, k + 1):
            x = row[-1] + 1
            # We need to find an x such that diffs {x - a_i} are not in used_diffs
            # And x itself doesn't cause issues (it's just a value)
            # We also need to ensure x is not too large? No, we just want a valid one.
            # To keep it small, we start from min and go up.
            # To add diversity, we might skip some initial candidates.
            
            # Limit search to avoid infinite loop if no solution (though there should be one)
            # A safe upper bound for a single row in a valid set of 7 rows with M~112 is M.
            # But we don't know M yet. Let's try up to 200.
            found = False
            # Random skip: with probability 'bias', skip the first few valid candidates
            skip_count = 0
            if rng.random() < 0.2:
                skip_count = rng.randint(1, 5)
            
            current_skip = 0
            while x < 200:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    if current_skip < skip_count:
                        current_skip += 1
                        x += 1
                        continue
                    row.append(x)
                    used_diffs.update(new_diffs)
                    found = True
                    break
                x += 1
            
            if not found:
                # Rollback if we added partial row? No, we only add on success.
                # But we modified used_diffs? No, we only update on success.
                # Wait, the code above updates used_diffs inside the loop.
                # If we fail, we should not have updated.
                # Let's re-structure: check first, then update.
                pass
            if not found:
                return None
        return row

    def compress_row(row, used_diffs, k, rng):
        """
        Try to reduce the entries of 'row' by re-greeding it, 
        but starting from the existing row's structure or just re-greeding from scratch 
        with the current used_diffs (excluding this row's diffs).
        Returns a new row if it is valid and has a smaller max entry.
        """
        # Remove current row's diffs from used_diffs
        current_diffs = row_diffs(row)
        used_diffs.difference_update(current_diffs)
        
        # Re-greedy the row
        # We want to keep it similar? No, just find the tightest one.
        # To allow escape from local minima, we can add some randomness.
        new_row = greedy_row(used_diffs, k, rng, bias=0.5)
        
        if new_row is None:
            # Restore diffs if we failed? We didn't add them back yet.
            used_diffs.update(current_diffs)
            return None
        
        new_diffs = row_diffs(new_row)
        
        # Check if the new row is strictly better (smaller max) or equal
        if max(new_row) <= max(row):
            # Add new diffs back
            used_diffs.update(new_diffs)
            return new_row
        else:
            # Restore old diffs
            used_diffs.update(current_diffs)
            return None

    def validate(rows):
        if rows is None:
            return False
        if len(rows) != n_rows:
            return False
        all_diffs = set()
        for r in rows:
            if len(r) != k + 1:
                return False
            if r[0] != 0:
                return False
            for i in range(1, len(r)):
                if r[i] <= r[i - 1]:
                    return False
            diffs = row_diffs(r)
            if len(diffs) != k * (k + 1) // 2:
                return False
            if all_diffs & diffs:
                return False
            all_diffs.update(diffs)
        return True

    best_rows = None
    best_max = float('inf')
    
    # Initialize with a simple greedy solution
    # We need a global used_diffs set
    # Let's do multiple restarts of the whole construction
    
    for restart in range(100):
        if not time_ok():
            break
            
        rng = random.Random(restart * 12345 + 67890)
        used_diffs = set()
        rows = []
        success = True
        
        # Randomize row construction order? 
        # The rows are indistinguishable in the final output, so order doesn't matter for validity.
        # But the "greediness" depends on what diffs are already taken.
        # Taking smaller rows first might be better? Or larger?
        # Let's just build them in order.
        
        for r_idx in range(n_rows):
            # Add some randomness to the greedy choice
            # The greedy_row function uses rng.
            new_row = greedy_row(used_diffs, k, rng)
            if new_row is None:
                success = False
                break
            rows.append(new_row)
            
        if success and validate(rows):
            current_max = max(max(r) for r in rows)
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(r) for r in rows]
                
            # Local Search Phase
            if time_ok():
                # Try to improve the current best_rows
                # We will work on a copy
                work_rows = [list(r) for r in best_rows]
                work_used = set()
                for r in work_rows:
                    work_used.update(row_diffs(r))
                
                improved = True
                iterations = 0
                while improved and iterations < 20 and time_ok():
                    improved = False
                    iterations += 1
                    # Shuffle row order for compression
                    row_order = list(range(n_rows))
                    rng.shuffle(row_order)
                    
                    for i in row_order:
                        if not time_ok():
                            break
                        new_row = compress_row(work_rows[i], work_used, k, rng)
                        if new_row is not None:
                            # Check if it actually improved the global max
                            old_max = max(max(r) for r in work_rows)
                            work_rows[i] = new_row
                            new_max = max(max(r) for r in work_rows)
                            if new_max < old_max:
                                improved = True
                                if new_max < best_max:
                                    best_max = new_max
                                    best_rows = [list(r) for r in work_rows]
                                    # Update work_used is already done in compress_row
                            else:
                                # If it didn't improve global max, maybe keep it for diversity?
                                # Or revert? 
                                # Since compress_row only returns if max(new) <= max(old), 
                                # and we only update best if global max drops, 
                                # we can keep the change in work_rows to allow further changes.
                                pass

    if best_rows is None:
        # Fallback: Simple Greedy
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(1, k + 1):
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        row.append(x)
                        used.update(new_diffs)
                        break
                    x += 1
            rows.append(row)
        best_rows = rows

    return best_rows
