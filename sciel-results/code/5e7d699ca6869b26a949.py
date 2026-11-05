import random
import time

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are globally distinct.
    Minimize the largest entry.
    """
    best_rows = None
    best_scope = float('inf')
    
    start_time = time.time()
    max_time = 115  # Leave 5s buffer
    
    # Try multiple random restarts with different strategies
    for attempt in range(200):
        if time.time() - start_time > max_time:
            break
        
        # Randomized greedy with varying biases
        used = set()
        rows = []
        success = True
        
        for _ in range(n_rows):
            row = [0]
            # Use a randomized approach: try candidates in slightly shuffled order
            # but biased toward smaller values
            x = 0
            while len(row) < k + 1:
                # Try a window of candidates starting from x+1
                found = False
                # Bias: try smaller values first but with some randomness
                candidates = list(range(x + 1, x + 50))
                random.shuffle(candidates)
                # Sort by value but add small random perturbation to break ties
                candidates.sort(key=lambda v: v + random.random() * 0.1)
                
                for cand in candidates:
                    new_diffs = {cand - a for a in row}
                    if not new_diffs & used:
                        row.append(cand)
                        used |= new_diffs
                        x = cand
                        found = True
                        break
                
                if not found:
                    success = False
                    break
            
            if not success:
                break
            rows.append(row)
        
        if success and len(rows) == n_rows:
            scope = max(max(r) for r in rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
    
    if best_rows is None:
        # Fallback to pure greedy
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
    
    return best_rows
