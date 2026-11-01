"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Greedy initial solution
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
        return rows
    
    rows = greedy_construct()
    best_rows = [r[:] for r in rows]
    best_max = max(max(r) for r in rows)
    
    # Compute used differences
    def get_used_diffs(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        return used
    
    used = get_used_diffs(rows)
    
    # Simulated annealing
    current_temp = 50.0
    cooling = 0.995
    min_temp = 0.1
    
    random.seed(42)
    
    while current_temp > min_temp and time.time() < deadline:
        # Pick a random row and a random mark (not the first)
        row_idx = random.randint(0, n_rows - 1)
        mark_idx = random.randint(1, k)  # marks 1..k (index 1 to k)
        
        old_val = rows[row_idx][mark_idx]
        
        # Try to move this mark to a smaller value
        # Find a new position that's valid
        # The mark must be > rows[row_idx][mark_idx-1] and < rows[row_idx][mark_idx+1] if mark_idx < k
        lower_bound = rows[row_idx][mark_idx - 1] + 1
        upper_bound = rows[row_idx][mark_idx + 1] - 1 if mark_idx < k else old_val - 1
        
        if lower_bound >= upper_bound:
            continue
        
        # Try a random smaller value
        if old_val <= lower_bound:
            continue
            
        # Choose a new value between lower_bound and min(old_val-1, upper_bound)
        max_new = min(old_val - 1, upper_bound)
        if max_new < lower_bound:
            continue
            
        new_val = random.randint(lower_bound, max_new)
        if new_val == old_val:
            continue
        
        # Check if new value creates valid differences
        row = rows[row_idx]
        # Remove old differences involving this mark
        old_diffs = set()
        for i in range(len(row)):
            if i != mark_idx:
                d = abs(row[mark_idx] - row[i])
                old_diffs.add(d)
        
        # Check new differences
        new_diffs = set()
        for i in range(len(row)):
            if i != mark_idx:
                d = abs(new_val - row[i])
                new_diffs.add(d)
        
        # The new differences must not conflict with used (excluding old_diffs)
        conflict = new_diffs & (used - old_diffs)
        if conflict:
            # Accept with probability based on temperature if it's worse
            # But we only accept if no conflict
            continue
        
        # Apply the move
        row[mark_idx] = new_val
        used = (used - old_diffs) | new_diffs
        
        new_max = max(max(r) for r in rows)
        if new_max < best_max:
            best_max = new_max
            best_rows = [r[:] for r in rows]
        
        current_temp *= cooling
    
    return best_rows
# EVOLVE-BLOCK-END

