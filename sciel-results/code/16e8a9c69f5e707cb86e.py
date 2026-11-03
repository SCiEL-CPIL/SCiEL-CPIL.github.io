import time
import random
import math

def construct(n_rows, k):
    """
    Finds a set of n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Helper to generate a greedy solution
    def greedy_solution():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used.update(new_diffs)
            rows.append(row)
        return rows

    # Get initial solution
    best_rows = greedy_solution()
    best_score = max(max(row) for row in best_rows)
    current_rows = [row[:] for row in best_rows]
    
    # Initial global used diffs
    current_used_diffs = set()
    for row in current_rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                current_used_diffs.add(row[j] - row[i])
                
    current_score = max(max(row) for row in current_rows)
    
    # Simulated Annealing Parameters
    T = 10.0  # Initial temperature
    cooling_rate = 0.9995
    min_T = 0.01
    
    # Helper to check if a row has distinct internal differences
    def has_distinct_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.append(row[j] - row[i])
        return len(diffs) == len(set(diffs))

    while T > min_T and time.time() < deadline:
        # Pick a random row and position
        i = random.randint(0, n_rows - 1)
        j = random.randint(1, k) # Positions 1 to k (index 0 is always 0)
        
        row = current_rows[i]
        old_val = row[j]
        
        # Determine valid range for new_val
        low_bound = row[j-1] + 1
        high_bound = row[j+1] - 1 if j < k else old_val + 50 # If last element, allow growth
        
        if low_bound > high_bound:
            if j == k:
                high_bound = old_val + 50
            else:
                T *= cooling_rate
                continue

        range_size = high_bound - low_bound + 1
        if range_size <= 0:
            continue
            
        # Randomly pick a new value in the range
        # Bias towards smaller values to reduce score
        if random.random() < 0.5 and low_bound < old_val:
            new_val = random.randint(low_bound, old_val - 1)
        else:
            new_val = random.randint(low_bound, high_bound)
            
        if new_val == old_val:
            continue

        # Check validity of the move
        # 1. Check strict increasing
        if j > 0 and new_val <= row[j-1]:
            continue
        if j < k and new_val >= row[j+1]:
            continue
            
        # 2. Check internal distinctness of the new row
        # Construct new row temporarily for check
        new_row = row[:]
        new_row[j] = new_val
        if not has_distinct_diffs(new_row):
            continue
            
        # 3. Check global difference conflicts
        diffs_to_remove = set()
        for l in range(j):
            diffs_to_remove.add(old_val - row[l])
        for m in range(j + 1, k + 1):
            diffs_to_remove.add(row[m] - old_val)
            
        diffs_to_add = set()
        for l in range(j):
            diffs_to_add.add(new_val - row[l])
        for m in range(j + 1, k + 1):
            diffs_to_add.add(row[m] - new_val)
            
        conflict = False
        for d in diffs_to_add:
            if d in diffs_to_remove:
                continue
            if d in current_used_diffs:
                conflict = True
                break
        
        if conflict:
            continue
            
        # Move is valid!
        # Apply move
        row[j] = new_val
        current_used_diffs.difference_update(diffs_to_remove)
        current_used_diffs.update(diffs_to_add)
        
        new_score = max(max(r) for r in current_rows)
        
        # Acceptance criterion
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_rows = [r[:] for r in current_rows]
        else:
            delta = new_score - current_score
            if delta == 0:
                accept = True
            else:
                p = math.exp(-delta / T)
                accept = random.random() < p
                
            if accept:
                current_score = new_score
            else:
                # Revert move
                row[j] = old_val
                current_used_diffs.difference_update(diffs_to_add)
                current_used_diffs.update(diffs_to_remove)
                current_score = max(max(r) for r in current_rows)
        
        # Cool down
        T *= cooling_rate
        
    return best_rows
