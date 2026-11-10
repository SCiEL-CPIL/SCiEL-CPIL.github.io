import time
import random

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 115  # 5s buffer
    
    # 1. Generate an initial valid solution using a simple greedy algorithm
    def greedy_construct(n, k, max_limit=None):
        used = set()
        rows = []
        for _ in range(n):
            row = [0]
            last = 0
            while len(row) < k + 1:
                found = False
                # Try to pick the smallest available number
                x = last + 1
                while True:
                    if max_limit is not None and x > max_limit:
                        break
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        last = x
                        found = True
                        break
                    x += 1
                if not found:
                    return None
            rows.append(row)
        return rows

    # Initialize
    initial = greedy_construct(n_rows, k)
    if initial is None:
        # Fallback: very large numbers
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
        
    current_rows = initial
    # Calculate initial used diffs and max
    def get_used_and_max(rows):
        used = set()
        max_val = 0
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    used.add(d)
            if r[-1] > max_val:
                max_val = r[-1]
        return used, max_val

    current_used, current_max = get_used_and_max(current_rows)
    
    # Local Search Loop
    # We try to reduce the global maximum by replacing the row containing it.
    # If we can't reduce the max, we try to replace rows with the same max to change the difference set.
    
    max_iterations = 5000
    for _ in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Find the row with the largest element
        max_row_idx = -1
        max_val = 0
        for i, r in enumerate(current_rows):
            if r[-1] > max_val:
                max_val = r[-1]
                max_row_idx = i
        
        # Try to replace row max_row_idx with a better one
        # Target: max < max_val if possible, otherwise max == max_val with different structure
        
        # Remove current row's diffs
        row_to_replace = current_rows[max_row_idx]
        diffs_to_remove = set()
        for i in range(len(row_to_replace)):
            for j in range(i+1, len(row_to_replace)):
                diffs_to_remove.add(row_to_replace[j] - row_to_replace[i])
        
        available_used = current_used - diffs_to_remove
        
        # Try to construct a new row with max < max_val
        # Use a randomized backtracking/greedy
        new_row = None
        
        # Try limit = max_val - 1
        # We can use a simple recursive backtracking with pruning
        # Since k is small (6 in the example, but general k), recursion depth is k+1
        
        def backtrack_row(k_remaining, last_val, current_row, available_diffs, limit):
            if k_remaining == 0:
                return current_row[:]
            
            start_x = last_val + 1
            # Randomize order to escape local minima? 
            # For now, try from smallest to largest to keep it compact, but add some randomness
            # Actually, trying smallest first is good for minimizing max.
            # To add randomness, we can shuffle the range or pick random candidates.
            
            # Let's try a window of candidates
            candidates = list(range(start_x, limit + 1))
            random.shuffle(candidates)
            
            for x in candidates:
                new_diffs = {x - a for a in current_row}
                if new_diffs & available_diffs:
                    continue
                if x > limit:
                    continue
                # Check if we can complete the rest?
                # Simple heuristic: if we pick x, the remaining k_remaining-1 elements must be > x.
                # The max possible value for the rest is limit.
                # We need to ensure enough space.
                
                current_row.append(x)
                new_available = available_diffs - new_diffs
                result = backtrack_row(k_remaining - 1, x, current_row, new_available, limit)
                if result is not None:
                    return result
                current_row.pop()
            return None

        # Attempt 1: Limit = max_val - 1
        if max_val > 1:
            new_row = backtrack_row(k, 0, [], available_used, max_val - 1)
            if new_row:
                current_rows[max_row_idx] = new_row
                # Update used set
                current_used -= diffs_to_remove
                for i in range(len(new_row)):
                    for j in range(i+1, len(new_row)):
                        current_used.add(new_row[j] - new_row[i])
                current_max = max(r[-1] for r in current_rows)
                continue

        # Attempt 2: Limit = max_val (try to find a different row with same max)
        # This might unlock further reductions
        # To avoid infinite loop of swapping same max rows, we can limit attempts or use a "tabu" list
        # For simplicity, try a few random restarts for this limit
        for attempt in range(3):
            new_row = backtrack_row(k, 0, [], available_used, max_val)
            if new_row and new_row != row_to_replace:
                current_rows[max_row_idx] = new_row
                current_used -= diffs_to_remove
                for i in range(len(new_row)):
                    for j in range(i+1, len(new_row)):
                        current_used.add(new_row[j] - new_row[i])
                # current_max remains same, but structure changed
                break

    return current_rows
