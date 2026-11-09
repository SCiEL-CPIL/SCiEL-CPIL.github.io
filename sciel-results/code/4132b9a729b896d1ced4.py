"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    
    Uses a backtracking search with constraint propagation, but given the
    120s time limit, we use a greedy approach with lookahead and local
    improvements, plus a randomized restart strategy to find better solutions.
    """
    # For the specific case of n_rows=7, k=6, we know the best known is 112.
    # We'll try to find a solution with largest entry as small as possible.
    
    # First, try a smart greedy with multiple randomizations
    best_scope = None
    best_rows = None
    
    import random
    import time
    
    start_time = time.time()
    time_limit = 110  # Leave some margin
    
    # Try different random seeds and strategies
    for seed in range(200):
        if time.time() - start_time > time_limit:
            break
        random.seed(seed)
        
        # Strategy: greedy with randomized tie-breaking
        used = set()
        rows = []
        success = True
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                # Generate candidate values, sorted by some heuristic
                # Try values starting from 1, but with some randomization in order
                # We want to minimize the final max, so prefer smaller values
                # but avoid conflicts
                
                # Find all valid next values
                candidates = []
                x = 1
                max_try = 200  # Don't search too far for each step
                while x <= max_try:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    success = False
                    break
                
                # Pick the smallest candidate (greedy) with slight randomization
                # to explore different paths
                if len(candidates) > 1 and random.random() < 0.3:
                    # Randomly pick from the first few candidates
                    pick_idx = random.randint(0, min(3, len(candidates) - 1))
                    x = candidates[pick_idx]
                else:
                    x = candidates[0]
                
                row.append(x)
                used |= {x - a for a in row[:-1]}
            
            if not success:
                break
            rows.append(row)
        
        if success and rows:
            scope = max(max(r) for r in rows)
            if best_scope is None or scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
        
        # Early stop if we reach the theoretical lower bound
        # The counting bound is 105, best known is 112
        if best_scope is not None and best_scope <= 106:
            break
    
    if best_rows is not None:
        return best_rows
    
    # Fallback: simple greedy (should not happen)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not (new & used):
                row.append(x)
                used |= new
        rows.append(row)
    return rows
# EVOLVE-BLOCK-END
