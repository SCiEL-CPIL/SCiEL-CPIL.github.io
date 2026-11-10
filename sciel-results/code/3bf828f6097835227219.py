import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def is_valid(rows):
        """Check if all positive differences are distinct across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    def greedy_build(seed, n_rows, k):
        """Greedy construction with randomization for tie-breaking."""
        rng = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try next few candidates, pick one that works
                found = False
                for _ in range(50):  # try up to 50 candidates
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        found = True
                        break
                if not found:
                    # Fallback: keep incrementing
                    while True:
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                            break
            rows.append(row)
        return rows
    
    def local_search(rows, rng, max_iters=1000):
        """Try to reduce the maximum entry via local moves."""
        best = [list(r) for r in rows]
        best_score = max_entry(best)
        
        for it in range(max_iters):
            if elapsed() > deadline - 5:
                break
            
            # Pick a random row and a random position to try to reduce
            row_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)  # positions 1..k (0 is fixed)
            
            current_val = best[row_idx][pos]
            # Try to reduce this value
            for new_val in range(current_val - 1, 0, -1):
                # Check if replacing works
                trial = [list(r) for r in best]
                trial[row_idx][pos] = new_val
                
                # Check increasing
                if pos > 0 and trial[row_idx][pos] <= trial[row_idx][pos-1]:
                    continue
                if pos < k and trial[row_idx][pos] >= trial[row_idx][pos+1]:
                    continue
                
                # Check distinct differences
                used = set()
                valid = True
                for r in trial:
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            d = r[j] - r[i]
                            if d in used:
                                valid = False
                                break
                            used.add(d)
                    if not valid:
                        break
                
                if valid:
                    new_score = max_entry(trial)
                    if new_score < best_score:
                        best = trial
                        best_score = new_score
                    break
        
        return best
    
    # Phase 1: Try multiple greedy constructions with different seeds
    best_rows = None
    best_val = float('inf')
    
    for seed in range(50):
        if elapsed() > deadline - 20:
            break
        rows = greedy_build(seed, n_rows, k)
        val = max_entry(rows)
        if val < best_val:
            best_val = val
            best_rows = rows
    
    if best_rows is None:
        # Fallback to simple greedy
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
    
    # Phase 2: Local search to improve
    rng = random.Random(42)
    best_rows = local_search(best_rows, rng, max_iters=200)
    
    # Phase 3: More local search iterations if time permits
    if elapsed() < deadline - 10:
        best_rows = local_search(best_rows, rng, max_iters=500)
    
    return best_rows
