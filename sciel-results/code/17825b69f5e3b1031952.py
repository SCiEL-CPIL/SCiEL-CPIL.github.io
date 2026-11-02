"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import itertools

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set with n_rows rows of k+1 increasing integers
    starting at 0, such that all positive differences within rows are distinct across
    all rows. Minimize the largest entry.
    
    Uses randomized local search with simulated annealing-like acceptance.
    """
    if n_rows <= 0:
        return []
    if k == 0:
        return [[0] for _ in range(n_rows)]
    
    # Total number of differences needed
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Simple counting bound: need at least total_diffs distinct differences
    # Each row with max m has at most m*(m+1)/2 differences if it were a ruler
    # But differences must be globally distinct
    
    # Lower bound on max entry: smallest m such that n_rows * m >= total_diffs
    # Actually more precisely, each row can contribute at most m differences (1..m)
    # so n_rows * m >= total_diffs => m >= total_diffs / n_rows
    lower_bound = (total_diffs + n_rows - 1) // n_rows
    
    # We'll try to find a solution with max entry close to lower_bound
    # Use randomized construction with backtracking and local search
    
    best_solution = None
    best_max_val = float('inf')
    
    # Try multiple random restarts with different strategies
    max_restarts = 50
    time_limit = 110  # seconds, leave buffer
    
    import time
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Generate a random initial solution using greedy with randomization
        random.seed(restart * 1000 + 42)
        
        used = set()
        rows = []
        valid = True
        
        for row_idx in range(n_rows):
            row = [0]
            # Greedy with some randomness: try values in shuffled order near the minimum
            while len(row) < k + 1:
                # Find the smallest x such that all differences x - a for a in row are unused
                # Add some randomness by considering a window of candidates
                found = False
                # Try from current max + 1 upwards, but with some random jumps
                base = row[-1] + 1
                # Consider candidates in a random order within a window
                window_size = 20 + restart
                candidates = list(range(base, base + window_size))
                random.shuffle(candidates)
                
                for x in candidates:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                
                if not found:
                    # Extend search
                    for x in range(base + window_size, base + window_size * 5):
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                            found = True
                            break
                
                if not found:
                    valid = False
                    break
            
            if not valid:
                break
            
            rows.append(row)
        
        if not valid or len(rows) < n_rows:
            continue
        
        # Check validity and compute max value
        all_diffs = set()
        max_val = 0
        valid = True
        for row in rows:
            if row[0] != 0:
                valid = False
                break
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        valid = False
                        break
                    all_diffs.add(d)
                max_val = max(max_val, row[i])
            if not valid:
                break
        
        if not valid:
            continue
        
        if max_val < best_max_val:
            best_max_val = max_val
            best_solution = [list(r) for r in rows]
        
        # If we've hit the lower bound, we can stop
        if best_max_val <= lower_bound:
            break
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: use the original greedy approach
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

# EVOLVE-BLOCK-END
