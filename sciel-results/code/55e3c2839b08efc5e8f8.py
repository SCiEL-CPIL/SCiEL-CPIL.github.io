import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0

    def get_diffs_bits(row):
        bits = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                bits |= (1 << (row[j] - row[i]))
        return bits

    def find_row_backtrack(used_bits, k, limit, randomize=True):
        row = [0]
        
        def backtrack(depth, last_val, curr_bits):
            if depth == k:
                return True
            
            candidates = list(range(last_val + 1, limit + 1))
            if randomize:
                random.shuffle(candidates)
            
            for x in candidates:
                new_bits = 0
                valid = True
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if (used_bits & bit) or (curr_bits & bit):
                        valid = False
                        break
                    new_bits |= bit
                
                if valid:
                    row.append(x)
                    if backtrack(depth + 1, x, curr_bits | new_bits):
                        return True
                    row.pop()
            
            return False
        
        if backtrack(0, 0, 0):
            return row
        return None

    # Initial Greedy
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        curr_bits = 0
        for _ in range(k):
            x = row[-1] + 1
            while True:
                new_bits = 0
                valid = True
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if (used & bit) or (curr_bits & bit):
                        valid = False
                        break
                    new_bits |= bit
                if valid:
                    row.append(x)
                    curr_bits |= new_bits
                    break
                x += 1
        rows.append(row)
        used |= curr_bits
        
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    current_max = best_max

    T = 1.0
    cooling = 0.995

    while time.time() < deadline:
        r_idx = random.randint(0, n_rows - 1)
        old_row = rows[r_idx]
        old_bits = get_diffs_bits(old_row)
        
        used = used & ~old_bits
        
        max_other = 0
        for i in range(n_rows):
            if i != r_idx:
                if max(rows[i]) > max_other:
                    max_other = max(rows[i])
        
        # Try to find a better row
        # We want to minimize max(new_row, max_other)
        # So we want max(new_row) to be small.
        
        found = False
        # Try a few times with randomization
        for attempt in range(10):
            # Try to fit in a limit that might reduce the global max
            # If max_other is the bottleneck, we can't reduce global max by reducing this row.
            # But we can still try to find a valid row.
            
            # Let's try to find a row with max <= current_max - 1
            # If that fails, try max <= current_max
            # We bias towards lower limits
            
            if random.random() < 0.5:
                limit = current_max - 1
                if limit < max_other:
                    limit = max_other
            else:
                limit = current_max
            
            if limit < 0:
                limit = 0
                
            new_row = find_row_backtrack(used, k, limit, randomize=True)
            if new_row:
                new_max = max(new_row)
                new_global = max(new_max, max_other)
                
                if new_global < current_max:
                    rows[r_idx] = new_row
                    used |= get_diffs_bits(new_row)
                    current_max = new_global
                    if current_max < best_max:
                        best_max = current_max
                        best_rows = [r[:] for r in rows]
                    found = True
                    break
                elif new_global == current_max:
                    # Accept with probability to explore
                    if random.random() < 0.3:
                        rows[r_idx] = new_row
                        used |= get_diffs_bits(new_row)
                        found = True
                        break
        
        if not found:
            # Restore old row
            rows[r_idx] = old_row
            used |= old_bits

        T *= cooling
        if T < 0.1:
            T = 0.1

    return best_rows
