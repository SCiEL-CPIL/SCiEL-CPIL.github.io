import time
import random
from itertools import combinations

def construct(n_rows, k):
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    
    # For the specific case (7,5): 7 rows, 6 elements each
    # Each row has C(6,2)=15 differences
    # Total differences: 7*15 = 105
    
    # Strategy: Use a more sophisticated search with backtracking
    # and constraint propagation to find a solution with smaller max value
    
    # First, try a randomized greedy with multiple restarts
    # then use local search to improve
    
    best_solution = None
    best_max_val = float('inf')
    
    def greedy_construct(used_diffs, n_rows, k, max_val_limit=None, randomize=False):
        """Greedy construction with optional randomization."""
        rows = []
        used = set(used_diffs) if used_diffs else set()
        
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                if randomize:
                    # Try a few random positions to add diversity
                    candidates = []
                    for _ in range(10):
                        trial = x + random.randint(1, 20)
                        if max_val_limit and trial > max_val_limit:
                            continue
                        new_diffs = {trial - a for a in row}
                        if not new_diffs & used:
                            candidates.append(trial)
                    if candidates:
                        x = random.choice(candidates)
                    else:
                        x += 1
                else:
                    x += 1
                
                if max_val_limit and x > max_val_limit:
                    return None
                
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                else:
                    # If can't place here, try next
                    if not randomize:
                        continue
                    else:
                        # In random mode, we already tried random positions
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                        else:
                            return None
            
            rows.append(row)
        
        return rows
    
    # Try standard greedy first
    sol = greedy_construct(set(), n_rows, k)
    if sol:
        best_max_val = max(max(row) for row in sol)
        best_solution = sol
    
    # Try multiple randomized restarts with different seeds
    rng = random.Random(42)
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        used_diffs = set()
        rows = []
        success = True
        
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Randomized greedy: try several candidate positions
                found = False
                for _ in range(5):
                    if time.time() > deadline - 5:
                        success = False
                        break
                    x = rng.randint(1, 150)
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used_diffs:
                        row.append(x)
                        used_diffs |= new_diffs
                        found = True
                        break
                if not found:
                    # Fall back to sequential search from current x
                    x = row[-1] + 1
                    while len(row) < k + 1:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used_diffs:
                            row.append(x)
                            used_diffs |= new_diffs
                            x += 1
                            break
                        x += 1
                    else:
                        success = False
                        break
                else:
                    x = row[-1] + 1
            
            if not success:
                break
            rows.append(row)
        
        if success:
            max_val = max(max(row) for row in rows)
            if max_val < best_max_val:
                best_max_val = max_val
                best_solution = rows
    
    # Try a more structured approach: build rows in a specific order
    # and use backtracking for the last few rows
    if time.time() < deadline - 20:
        # Try to improve by reordering or using a different strategy
        # For now, return the best found
        pass
    
    if best_solution is None:
        # Fallback: simple greedy
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
    
    return best_solution
