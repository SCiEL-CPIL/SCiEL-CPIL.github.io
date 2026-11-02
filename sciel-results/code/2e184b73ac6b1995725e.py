import time
import random
import sys

def construct(n_rows, k):
    """
    Constructs a set of difference triangles (rows of increasing integers)
    such that all pairwise differences are distinct across all rows.
    Minimizes the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_rows = None
    best_score = float('inf')
    
    # Parameters for randomized greedy
    # We try different window sizes to explore the space
    window_sizes = [1, 2, 3, 5, 8]
    
    while time.time() < deadline:
        # Pick a random window size to vary exploration vs exploitation
        window = random.choice(window_sizes)
        
        rows = []
        used_diffs = set()
        success = True
        
        for _ in range(n_rows):
            row = [0]
            # We need to add k more elements
            for _ in range(k):
                last_val = row[-1]
                x = last_val + 1
                candidates = []
                
                # Find 'window' valid candidates
                # Optimization: If we find a candidate, add it.
                # We stop once we have 'window' candidates.
                while len(candidates) < window:
                    # Check if x is valid with respect to current row and global used_diffs
                    valid = True
                    for a in row:
                        if x - a in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1
                    
                    # Safety check: if x gets too large, we might be stuck?
                    # But usually there's always a solution.
                    # If we run out of time, break outer loop.
                    if time.time() > deadline:
                        success = False
                        break
                
                if not success:
                    break
                
                if not candidates:
                    # No valid candidate found? Should not happen if scope is unbounded,
                    # but if we hit a dead end in logic, break.
                    success = False
                    break
                
                # Pick a random candidate from the window
                chosen = random.choice(candidates)
                row.append(chosen)
                
                # Update used_diffs
                for a in row[:-1]: # Exclude the new element itself (diff 0)
                    used_diffs.add(chosen - a)
            
            if not success:
                break
            
            rows.append(row)
        
        if success and rows:
            # Calculate score: max entry in the entire set
            score = max(max(row) for row in rows)
            if score < best_score:
                best_score = score
                best_rows = rows
                # Optional: If we find a very good score, we might want to spend more time polishing?
                # For now, just keep searching.
                
    if best_rows is None:
        # Fallback to deterministic greedy if randomization failed completely (unlikely)
        used_diffs = set()
        best_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                valid = True
                for a in row:
                    if x - a in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        used_diffs.add(x - a)
                    x += 1
                else:
                    x += 1
            best_rows.append(row)
            
    return best_rows
