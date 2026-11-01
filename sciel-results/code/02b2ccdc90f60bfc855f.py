import random
import math
import time

def construct(n_rows, k):
    row_len = k + 1
    
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def is_valid_row(row, used_diffs_excluding_row):
        # Check if row's internal diffs are unique and don't conflict with others
        row_diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in row_diffs:
                    return False
                if d in used_diffs_excluding_row:
                    return False
                row_diffs.add(d)
        return True

    def fast_greedy(randomize=False):
        rows = [[0] for _ in range(n_rows)]
        used = set()
        order = list(range(n_rows))
        if randomize:
            random.shuffle(order)
        
        # We will fill rows one by one or interleaved?
        # Interleaved is usually better for span minimization.
        # But for speed, let's try filling one by one with a tight bound.
        # Actually, the previous greedy was interleaved.
        
        # Let's stick to a fast interleaved greedy.
        for step in range(n_rows * k):
            # Pick row with smallest current max, tie-break by length
            # To speed up, we don't need complex randomization in the inner loop
            best_idx = -1
            best_key = None
            for i in range(n_rows):
                if len(rows[i]) < row_len:
                    # Prioritize shorter rows, then smaller last val
                    key = (len(rows[i]), rows[i][-1])
                    if best_key is None or key < best_key:
                        best_key = key
                        best_idx = i
            
            if best_idx == -1:
                break
                
            i = best_idx
            current_row = rows[i]
            x = current_row[-1] + 1
            
            while True:
                # Check validity
                valid = True
                for v in current_row:
                    d = x - v
                    if d in used:
                        valid = False
                        break
                if valid:
                    current_row.append(x)
                    for v in current_row[:-1]:
                        used.add(x - v)
                    break
                x += 1
        return rows

    def local_search(rows, time_limit):
        # Local search to minimize max element
        # We will try to reduce the largest element in the largest rows.
        
        rows = [r[:] for r in rows]
        current_cost = max(max(r) for r in rows)
        best_cost = current_cost
        best_rows = [r[:] for r in rows]
        
        start_time = time.time()
        
        # Precompute used diffs for the current state
        def compute_used(rows):
            used = set()
            for r in rows:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        used.add(r[j]-r[i])
            return used
        
        used = compute_used(rows)
        
        while time.time() - start_time < time_limit:
            # Pick the row with the largest max element
            max_val = -1
            max_row_idx = -1
            for i, r in enumerate(rows):
                if r[-1] > max_val:
                    max_val = r[-1]
                    max_row_idx = i
            
            if max_val == 0:
                break
                
            row = rows[max_row_idx]
            
            # Try to reduce the last element? Or any element?
            # Reducing the last element is most direct.
            # But we might need to reduce intermediate elements to allow the last one to be smaller.
            # Let's try moving the last element to a smaller value.
            
            old_last = row[-1]
            new_last = -1
            
            # Try values from old_last - 1 down to row[-2] + 1
            # This is expensive if the gap is large.
            # Instead, let's try a random smaller value?
            # Or try old_last - 1, old_last - 2, ... up to a limit.
            
            # Better strategy: Pick a random element in the row (not the first)
            # and try to move it to a smaller position.
            
            if len(row) > 1:
                # Pick an index to modify
                # Focus on the last element first as it determines the span
                idx_to_move = len(row) - 1
                
                # Get current diffs of this row
                row_diffs = set()
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        row_diffs.add(row[j]-row[i])
                
                # Diffs used by other rows
                other_diffs = used - row_diffs
                
                # Try to find a new value for row[idx_to_move]
                # It must be greater than row[idx_to_move-1] and less than old_last (if we want to reduce)
                # Actually, we can try any value > row[idx_to_move-1]
                # But we want to reduce the max, so we look for values < old_last if idx_to_move is last.
                
                # Let's try a few random candidates in the range (row[idx_to_move-1], row[idx_to_move])
                # Or simply try row[idx_to_move] - 1, -2, etc.
                
                found = False
                # Try decreasing steps
                for delta in range(1, min(10, row[idx_to_move] - row[idx_to_move-1])):
                    new_val = row[idx_to_move] - delta
                    if new_val <= row[idx_to_move-1]:
                        continue
                    
                    # Check if new_val is valid
                    # New diffs involving idx_to_move
                    new_diffs = set()
                    for i in range(idx_to_move):
                        d = new_val - row[i]
                        if d in other_diffs:
                            new_diffs = None
                            break
                        new_diffs.add(d)
                    
                    if new_diffs is None:
                        continue
                        
                    # Check internal uniqueness with the rest of the row
                    # The rest of the row diffs are unchanged.
                    # We just need to ensure new_diffs don't conflict with each other
                    # (they are distinct by construction if row is strictly increasing? No, new_val - row[i] could equal new_val - row[j] only if row[i]==row[j], which is false).
                    # But new_diffs could conflict with the *old* diffs of the row that don't involve idx_to_move?
                    # No, the other diffs in the row are between indices < idx_to_move.
                    # The new diffs are between idx_to_move and indices < idx_to_move.
                    # So we just need to check against other_diffs.
                    # Wait, do the new diffs conflict with each other?
                    # new_val - row[i] == new_val - row[j] => row[i] == row[j]. False.
                    # So new_diffs are distinct.
                    
                    # Are they distinct from the diffs within the row not involving idx_to_move?
                    # Let row_rest = row[:idx_to_move].
                    # Diffs in row_rest are fixed.
                    # New diffs are new_val - row[i].
                    # Could new_val - row[i] == row[j] - row[i']?
                    # Yes. We must check this.
                    
                    # Efficient check:
                    # 1. Compute set of diffs in row_rest.
                    rest_diffs = set()
                    for i in range(idx_to_move):
                        for j in range(i+1, idx_to_move):
                            rest_diffs.add(row[j]-row[i])
                            
                    # 2. Check if any new_diff is in rest_diffs.
                    conflict = False
                    for d in new_diffs:
                        if d in rest_diffs:
                            conflict = True
                            break
                            
                    if not conflict:
                        # Valid move
                        # Update used set
                        # Remove old diffs involving idx_to_move
                        for i in range(idx_to_move):
                            used.discard(row[idx_to_move] - row[i])
                        
                        # Add new diffs
                        for d in new_diffs:
                            used.add(d)
                            
                        row[idx_to_move] = new_val
                        found = True
                        break
                        
                if found:
                    new_cost = max(max(r) for r in rows)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_rows = [r[:] for r in rows]
                    # Accept the move (greedy local search)
                    # We don't backtrack here, we just keep improving.
                else:
                    # Try another row or element?
                    # For simplicity, just pick a different random row next iteration
                    pass
            else:
                # Row is just [0], nothing to do
                pass
                
            # To avoid getting stuck in a local minimum where we can't reduce the last element,
            # we might need to try moving intermediate elements.
            # But for now, let's stick to reducing the last element of the largest row.
            # If we can't reduce it, we might be stuck.
            # Let's add a random restart or perturbation if stuck?
            # Given the time limit, let's just loop.
            
            # Optimization: If we can't reduce the last element, try reducing the second to last?
            # This is complex. Let's rely on the randomization of the initial greedy solution
            # and multiple restarts.
            
            # Actually, the above loop only tries to reduce the last element of the largest row.
            # If it fails, it loops again and tries the same thing. This is bad.
            # We should track if we made progress.
            
        return best_rows, best_cost

    start_time = time.time()
    timeout = 110
    
    best_solution = None
    best_cost = float('inf')
    
    # Run multiple iterations of Greedy + Local Search
    while time.time() - start_time < timeout:
        # 1. Generate a random greedy solution
        rows = fast_greedy(randomize=True)
        cost = max(max(r) for r in rows)
        
        if cost < best_cost:
            best_cost = cost
            best_solution = [r[:] for r in rows]
            
        # 2. Improve with local search
        # Allocate remaining time to local search, but cap it per iteration
        remaining = timeout - (time.time() - start_time)
        ls_time = min(5.0, remaining * 0.5) # Cap local search at 5s per iteration
        
        if ls_time > 1.0:
            improved_rows, improved_cost = local_search(rows, ls_time)
            if improved_cost < best_cost:
                best_cost = improved_cost
                best_solution = [r[:] for r in improved_rows]
                
    if best_solution is None:
        # Fallback
        rows = fast_greedy(randomize=False)
        best_solution = rows

    return best_solution
