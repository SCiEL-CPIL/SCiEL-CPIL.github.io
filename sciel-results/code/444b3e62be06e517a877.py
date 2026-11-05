# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0

    def greedy_construct():
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
        return rows, max(max(r) for r in rows)

    current_rows, current_scope = greedy_construct()
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope

    def calc_scope(rows):
        return max(max(r) for r in rows)

    def is_valid_move(row_idx, pos, new_val, rows):
        # Check if setting rows[row_idx][pos] = new_val results in a valid configuration
        # We assume other rows are unchanged and valid.
        # We need to check:
        # 1. Strictly increasing in the modified row.
        # 2. No duplicate differences within the modified row.
        # 3. No duplicate differences between modified row and other rows.
        
        # Temporarily apply
        old_val = rows[row_idx][pos]
        rows[row_idx][pos] = new_val
        
        # 1. Strictly increasing
        r = rows[row_idx]
        for i in range(len(r)-1):
            if r[i] >= r[i+1]:
                rows[row_idx][pos] = old_val
                return False
        
        # 2. & 3. Differences
        new_diffs = set()
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in new_diffs:
                    rows[row_idx][pos] = old_val
                    return False
                new_diffs.add(d)
        
        for idx, other_row in enumerate(rows):
            if idx == row_idx:
                continue
            for i in range(len(other_row)):
                for j in range(i+1, len(other_row)):
                    d = other_row[j] - other_row[i]
                    if d in new_diffs:
                        rows[row_idx][pos] = old_val
                        return False
                        
        # Valid
        return True

    # Local Search: Gap Reduction
    # Pick a random row, a random gap, try to shrink it.
    # To avoid getting stuck, we can also try to "shuffle" if no moves are possible for a while.
    
    stuck_counter = 0
    
    while time.time() < deadline:
        # Pick a row
        # Bias towards rows with larger scope
        scopes = [max(r) for r in current_rows]
        weights = [1000 + s for s in scopes]
        total_w = sum(weights)
        r_rand = random.random() * total_w
        acc = 0
        row_idx = 0
        for i, w in enumerate(weights):
            acc += w
            if r_rand <= acc:
                row_idx = i
                break
        
        # Pick a gap
        # Gaps are between index j and j+1 for j in 0..k-1
        # Reducing gap j means decreasing row[j+1] by 1
        # This is only possible if row[j+1] - row[j] > 1
        # And row[j+1] - 1 > row[j]
        
        possible_gaps = []
        for j in range(k):
            if current_rows[row_idx][j+1] - current_rows[row_idx][j] > 1:
                possible_gaps.append(j)
                
        if possible_gaps:
            j = random.choice(possible_gaps)
            new_val = current_rows[row_idx][j+1] - 1
            pos = j+1
            
            if is_valid_move(row_idx, pos, new_val, current_rows):
                # Accept
                # Note: is_valid_move already applied the change and reverted if invalid?
                # No, my is_valid_move above reverts on invalid. If valid, it leaves it applied.
                # Wait, if valid, it returns True, and the change is in `current_rows`.
                # If invalid, it reverts and returns False.
                
                new_scope = calc_scope(current_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
                stuck_counter = 0
            else:
                stuck_counter += 1
        else:
            # No gaps to reduce in this row. Try a different row or random perturbation.
            stuck_counter += 1
            
        if stuck_counter > 1000:
            # Try a random perturbation to escape local minimum
            # Pick a random row, random pos, random new value in range
            row_idx = random.randint(0, n_rows-1)
            pos = random.randint(1, k)
            low = current_rows[row_idx][pos-1] + 1
            high = current_rows[row_idx][pos] + 10 # Allow some increase
            
            if low <= high:
                new_val = random.randint(low, high)
                if new_val != current_rows[row_idx][pos]:
                    if is_valid_move(row_idx, pos, new_val, current_rows):
                        new_scope = calc_scope(current_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [r[:] for r in current_rows]
            stuck_counter = 0

    return best_rows
# EVOLVE-BLOCK-END
