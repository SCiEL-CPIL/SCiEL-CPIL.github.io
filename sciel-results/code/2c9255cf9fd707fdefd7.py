"""Difference triangle set (4,7) solver using randomized greedy with restarts."""
import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    
    best_rows = None
    best_scope = float('inf')
    
    # Number of restarts we'll attempt, but bounded by time
    max_restarts = 50000
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Randomized greedy with row-order shuffling
        # Each row is built greedily, but we randomize the order of rows
        # and use biased candidate selection to explore different solution spaces
        
        random.seed(restart * 7919 + 13)  # deterministic per restart for reproducibility
        
        # Shuffle row order: assign each row a random priority
        row_priorities = list(range(n_rows))
        random.shuffle(row_priorities)
        
        # We'll build rows in shuffled order, but track which row index each is
        used = set()
        rows = [None] * n_rows
        
        success = True
        for row_idx in row_priorities:
            row, x = [0], 0
            while len(row) < k + 1:
                # Try next value with some randomization
                # Biased selection: mostly try x+1, but sometimes skip ahead
                if random.random() < 0.1 and x > 0:
                    # Occasionally jump ahead to explore different structures
                    x += random.randint(1, 3)
                else:
                    x += 1
                
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
                else:
                    # Conflict: try a slightly larger value to avoid
                    # This is a simple heuristic; the while loop continues
                    pass
            
            rows[row_idx] = row
            scope = max(max(r) for r in rows)
        
        # Validate and update best
        if all(r is not None for r in rows):
            # Verify validity
            all_diffs = set()
            valid = True
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in all_diffs:
                            valid = False
                            break
                        all_diffs.add(d)
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                scope = max(max(r) for r in rows)
                if scope < best_scope:
                    best_scope = scope
                    best_rows = [list(r) for r in rows]
        
        # Early termination if we hit the counting bound
        # Counting bound: total differences = n_rows * k*(k+1)/2
        # These must fit in [1, scope], so scope >= total_diffs
        total_diffs = n_rows * k * (k + 1) // 2
        if best_scope <= total_diffs:
            break
    
    if best_rows is None:
        # Fallback: use the original deterministic greedy
        used = set()
        rows = []
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
    
    return best_rows
