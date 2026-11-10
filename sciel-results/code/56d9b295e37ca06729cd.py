import time
import random

def construct(n_rows, k):
    start_time = time.time()
    time_limit = 110 # 10s buffer
    
    best_rows = None
    best_max = float('inf')
    
    # We will run restarts until time runs out or we find a very good solution.
    # Let's say we try 20,000 restarts max.
    max_restarts = 20000
    window_size = 15
    
    for i in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        used = set()
        rows = []
        current_max = 0
        success = True
        
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < k + 1:
                candidates = []
                x = last + 1
                # Dynamic limit: if we have a current_max, try not to exceed it by too much?
                # Actually, if we exceed current_max, we might be improving a different row later?
                # No, current_max is the global max so far.
                # If we pick x > current_max, we increase the global max.
                # We want to minimize the final max.
                # So we should prefer x <= current_max if possible.
                # But if no candidate exists <= current_max, we must go higher.
                
                # Strategy: 
                # 1. Look for candidates in [last+1, current_max] (if current_max > last)
                # 2. If found, pick from them.
                # 3. If not, look for candidates in [current_max+1, current_max + window_size]
                
                limit1 = current_max if current_max > last else last
                while x <= limit1 and len(candidates) < window_size:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    # Need to go higher
                    limit2 = current_max + window_size * 2 # Arbitrary extension
                    while x <= limit2 and len(candidates) < window_size:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            candidates.append(x)
                        x += 1
                
                if not candidates:
                    success = False
                    break
                
                chosen = random.choice(candidates)
                row.append(chosen)
                new_diffs = {chosen - a for a in row[:-1]}
                used |= new_diffs
                last = chosen
                if chosen > current_max:
                    current_max = chosen
            
            if not success:
                break
            
            rows.append(row)
        
        if success:
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
                
                # Optional: Local search improvement
                # Try to reduce the max element of the best row
                # This is complex to implement correctly in the loop.
                # Let's rely on more restarts.
                
                if best_max <= 105:
                    break

    if best_rows is None:
        # Fallback to deterministic greedy
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
        
    return best_rows
