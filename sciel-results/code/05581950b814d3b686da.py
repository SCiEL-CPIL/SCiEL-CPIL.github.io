import time
import random
from itertools import combinations

def construct(n_rows, k):
    deadline = time.time() + 145  # Safety margin before 150s limit
    best_rows = None
    best_scope = float('inf')
    
    # Pure greedy as baseline
    def pure_greedy():
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
    
    best_rows = pure_greedy()
    best_scope = max(max(r) for r in best_rows)
    
    # Randomized greedy with different strategies
    rng = random.Random(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Strategy: for each row, at each position, consider a range of candidates
        # and pick one using a heuristic (e.g., minimize future conflicts)
        used = set()
        rows = []
        success = True
        
        for row_idx in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try candidates in a somewhat randomized order near the minimum
                found = False
                # Determine the minimum valid x
                min_x = 0
                while True:
                    min_x += 1
                    new_diffs = {min_x - a for a in row}
                    if not new_diffs & used:
                        break
                
                # Consider candidates from min_x to min_x + window
                window = 5 + attempt % 10
                candidates = list(range(min_x, min_x + window + 1))
                
                # Shuffle for randomization but keep min_x likely
                rng.shuffle(candidates)
                # Put min_x first with some probability
                if rng.random() < 0.5:
                    candidates.remove(min_x)
                    candidates.insert(0, min_x)
                
                for cand in candidates:
                    new_diffs = {cand - a for a in row}
                    if not new_diffs & used:
                        row.append(cand)
                        used |= new_diffs
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
    
    return best_rows
