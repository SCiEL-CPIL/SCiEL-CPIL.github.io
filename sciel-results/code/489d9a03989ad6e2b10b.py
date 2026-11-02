import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 145  # 5s safety margin
    rng = random.Random(42)
    
    # Helper: check if a row is valid (strictly increasing, starts at 0)
    def is_valid_row(row):
        if len(row) != k + 1 or row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        return True
    
    # Helper: get all differences from a row
    def row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs
    
    # Helper: check if a set of rows has all distinct differences
    def all_diffs_distinct(rows):
        all_diffs = set()
        for row in rows:
            d = row_diffs(row)
            if all_diffs & d:
                return False
            all_diffs |= d
        return True
    
    # Helper: get max value
    def get_max_val(rows):
        return max(max(row) for row in rows)
    
    # Initialize with greedy construction
    def greedy_init():
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
    
    # Randomized initialization
    def random_init():
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += rng.randint(1, 3)
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows
    
    # Start with greedy
    best_rows = greedy_init()
    best_max = get_max_val(best_rows)
    
    # Try to improve with simulated annealing
    current_rows = [row[:] for row in best_rows]
    current_max = best_max
    
    T = 10.0  # Initial temperature
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline and T > T_min:
        # Pick a random row and a random position to modify
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, k)  # Don't modify position 0 (must be 0)
        
        # Try to change current_rows[row_idx][pos] to a nearby value
        old_val = current_rows[row_idx][pos]
        new_val = old_val + rng.choice([-1, 1]) * rng.randint(1, 2)
        
        if new_val <= 0:
            T *= cooling
            continue
        
        # Check if the modified row would still be strictly increasing
        if pos > 0 and new_val <= current_rows[row_idx][pos - 1]:
            T *= cooling
            continue
        if pos < k and new_val >= current_rows[row_idx][pos + 1]:
            T *= cooling
            continue
        
        # Temporarily modify and check validity
        current_rows[row_idx][pos] = new_val
        new_diffs = row_diffs(current_rows[row_idx])
        
        # Check for conflicts with other rows
        valid = True
        for i in range(n_rows):
            if i == row_idx:
                continue
            if new_diffs & row_diffs(current_rows[i]):
                valid = False
                break
        
        if valid:
            new_max = get_max_val(current_rows)
            delta = new_max - current_max
            
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                # Accept
                current_max = new_max
                if new_max < best_max:
                    best_max = new_max
                    best_rows = [row[:] for row in current_rows]
            else:
                # Reject: revert
                current_rows[row_idx][pos] = old_val
        else:
            # Revert
            current_rows[row_idx][pos] = old_val
        
        T *= cooling
    
    # Final local improvement: try to reduce each entry
    for _ in range(100000):
        if time.time() > deadline:
            break
        improved = False
        for row_idx in range(n_rows):
            for pos in range(1, k + 1):
                old_val = best_rows[row_idx][pos]
                # Try to decrease
                for new_val in range(old_val - 1, 0, -1):
                    if pos > 0 and new_val <= best_rows[row_idx][pos - 1]:
                        break
                    if pos < k and new_val >= best_rows[row_idx][pos + 1]:
                        break
                    best_rows[row_idx][pos] = new_val
                    new_diffs = row_diffs(best_rows[row_idx])
                    valid = True
                    for i in range(n_rows):
                        if i == row_idx:
                            continue
                        if new_diffs & row_diffs(best_rows[i]):
                            valid = False
                            break
                    if valid:
                        improved = True
                        break
                    else:
                        best_rows[row_idx][pos] = old_val
                        break
        if not improved:
            break
    
    return best_rows
# EVOLVE-BLOCK-END
