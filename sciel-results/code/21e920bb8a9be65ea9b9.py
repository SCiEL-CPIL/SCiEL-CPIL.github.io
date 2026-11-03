"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools
import time

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows of k+1 increasing integers
    starting at 0, such that all positive differences within rows are distinct across
    all rows. Minimize the largest entry.
    
    Uses Simulated Annealing / Local Search on a valid initial solution.
    """
    if n_rows <= 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    # 1. Initialize with a valid greedy solution
    def greedy_init():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows, used

    initial_rows, initial_used = greedy_init()
    best_rows = [list(r) for r in initial_rows]
    best_max_val = max(max(r) for r in best_rows)
    
    # 2. Local Search with Simulated Annealing
    current_rows = [list(r) for r in initial_rows]
    current_used = set(initial_used)
    current_max = best_max_val
    
    start_time = time.time()
    time_limit = 110  # seconds
    
    T = 10.0
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        j_idx = random.randint(1, k)
        old_val = row[j_idx]
        
        min_val = row[j_idx - 1] + 1
        # We don't want to increase the max too much, so limit upper bound
        # If we are trying to minimize, we should prefer smaller values.
        # Let's sample candidates from a range biased towards smaller values.
        
        # Generate candidates
        candidates = set()
        # 1. Downward sweep
        for x in range(old_val - 1, min_val - 1, -1):
            candidates.add(x)
            if len(candidates) > 20: break
            
        # 2. Random samples in valid range
        # Upper bound for sampling: current_max + 5 (allow slight expansion)
        upper_bound = max(old_val + 10, current_max + 5)
        for _ in range(10):
            x = random.randint(min_val, upper_bound)
            candidates.add(x)
            
        candidates.discard(old_val)
        
        if not candidates:
            continue
            
        # Evaluate all candidates
        best_candidate = None
        best_new_max = float('inf')
        best_new_used = None
        best_new_row_val = None
        
        for x in candidates:
            # Check validity
            old_diffs = set()
            for i in range(j_idx):
                old_diffs.add(old_val - row[i])
            for i in range(j_idx + 1, len(row)):
                old_diffs.add(row[i] - old_val)
                
            new_diffs = set()
            for i in range(j_idx):
                new_diffs.add(x - row[i])
            for i in range(j_idx + 1, len(row)):
                new_diffs.add(row[i] - x)
                
            valid = True
            for d in new_diffs:
                if d in old_diffs:
                    continue
                if d in current_used:
                    valid = False
                    break
            
            if valid:
                # Calculate new max
                # Only this row changes
                new_row_max = max(row[:j_idx] + [x] + row[j_idx+1:])
                # Global max is max of this row's max and max of other rows
                # We can precompute max of other rows?
                # For now, just calculate full max if it looks promising
                # Optimization: if new_row_max >= current_max and current_max is from another row, 
                # the global max doesn't change.
                
                # Let's just compute the new global max
                # This is O(n_rows * k). Acceptable for small n.
                new_global_max = current_max
                if new_row_max > new_global_max:
                    new_global_max = new_row_max
                else:
                    # If new_row_max < current_max, we need to check if current_max was from this row
                    # and if so, what is the new max of other rows?
                    # This is getting complicated. 
                    # Simpler: just compute max of all rows.
                    pass
                
                # To be safe and simple, compute max of all rows
                # But we only changed one row.
                # Let's store max of other rows? No, just compute.
                # n_rows=7, k=6. 42 elements. Very fast.
                new_global_max = 0
                for r in range(n_rows):
                    if r == r_idx:
                        m = max(row[:j_idx] + [x] + row[j_idx+1:])
                    else:
                        m = max(current_rows[r])
                    if m > new_global_max:
                        new_global_max = m
                
                if new_global_max < best_new_max:
                    best_new_max = new_global_max
                    best_candidate = x
                    # Store the new used set? We can compute it later if accepted.
                    # Or store the diff sets.
                    # Let's just store the candidate. We will re-verify/apply if selected.
        
        if best_candidate is not None:
            # Decide whether to accept
            delta = best_new_max - current_max
            if delta <= 0 or random.random() < pow(2.71828, -delta / T):
                # Accept
                # Apply move
                old_diffs = set()
                for i in range(j_idx):
                    old_diffs.add(old_val - row[i])
                for i in range(j_idx + 1, len(row)):
                    old_diffs.add(row[i] - old_val)
                    
                new_diffs = set()
                for i in range(j_idx):
                    new_diffs.add(best_candidate - row[i])
                for i in range(j_idx + 1, len(row)):
                    new_diffs.add(row[i] - best_candidate)
                    
                current_used -= old_diffs
                current_used |= new_diffs
                row[j_idx] = best_candidate
                current_max = best_new_max
                
                if current_max < best_max_val:
                    best_max_val = current_max
                    best_rows = [list(r) for r in current_rows]
                
                T *= cooling_rate
            else:
                # Reject, but still cool? Or not?
                T *= cooling_rate
        else:
            T *= cooling_rate
            
    return best_rows

# EVOLVE-BLOCK-END
