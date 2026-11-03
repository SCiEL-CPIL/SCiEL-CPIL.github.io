"""Rewritten program for difference triangle set (4,7) with structural changes."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _scope(rows):
    """Return the maximum entry (scope) of the rows."""
    return max(max(row) for row in rows)

def _validate(rows):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True

def _greedy(n_rows, k, rng, mode='nearest'):
    """Greedy construction. mode: 'nearest', 'random', 'spread'."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if mode == 'nearest':
                while True:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                        break
            elif mode == 'random':
                placed = False
                for _ in range(100):
                    trial = x + rng.randint(1, 25)
                    new_diffs = {trial - a for a in row}
                    if not new_diffs & used:
                        x = trial
                        row.append(x)
                        used |= new_diffs
                        placed = True
                        break
                if not placed:
                    while True:
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                            break
            elif mode == 'spread':
                placed = False
                for _ in range(50):
                    trial = x + rng.randint(5, 35)
                    new_diffs = {trial - a for a in row}
                    if not new_diffs & used:
                        x = trial
                        row.append(x)
                        used |= new_diffs
                        placed = True
                        break
                if not placed:
                    while True:
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                            break
        rows.append(row)
    return rows

def _compact_rows(rows, n_rows, k, rng, time_budget=20.0):
    """Try to compact each row by rebuilding with smaller scope."""
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    start = time.time()
    
    # Compute diffs for each row
    row_diffs_list = []
    for r in best:
        rd = set()
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                rd.add(r[j] - r[i])
        row_diffs_list.append(rd)
    
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        for r_idx in range(n_rows):
            if time.time() - start > time_budget:
                break
            
            other_diffs = set()
            for i in range(n_rows):
                if i != r_idx:
                    other_diffs |= row_diffs_list[i]
            
            current_max = best[r_idx][-1]
            
            for new_max in range(current_max - 1, max(len(best[r_idx]) - 1, 0), -1):
                if new_max < len(best[r_idx]):
                    break
                
                # Try to rebuild row with max scope = new_max
                new_row = [0]
                x = 0
                success = True
                local_used = set()
                
                while len(new_row) < len(best[r_idx]):
                    found = False
                    for trial_x in range(x + 1, new_max + 1):
                        nd = {trial_x - a for a in new_row}
                        if not nd & other_diffs and not nd & local_used:
                            new_row.append(trial_x)
                            local_used |= nd
                            x = trial_x
                            found = True
                            break
                    if not found:
                        success = False
                        break
                
                if success:
                    # Verify no conflicts
                    new_rd = set()
                    for i in range(len(new_row)):
                        for j in range(i + 1, len(new_row)):
                            new_rd.add(new_row[j] - new_row[i])
                    if not (new_rd & other_diffs):
                        best[r_idx] = new_row
                        row_diffs_list[r_idx] = new_rd
                        new_scope = _scope(best)
                        if new_scope < best_scope:
                            best_scope = new_scope
                        improved = True
                        break
    
    return best, best_scope

def _anneal(rows, n_rows, k, rng, time_budget=60.0):
    """Simulated annealing to minimize scope."""
    current = [list(r) for r in rows]
    best = [list(r) for r in current]
    best_scope = _scope(current)
    start = time.time()
    
    T = 8.0
    cooling = 0.9995
    min_T = 0.05
    
    # Precompute all diffs
    all_diffs = set()
    for row in current:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                all_diffs.add(row[j] - row[i])
    
    while T > min_T and (time.time() - start) < time_budget:
        r_idx = rng.randint(0, n_rows - 1)
        row = current[r_idx]
        
        if len(row) < 2:
            T *= cooling
            continue
        
        pos = rng.randint(1, len(row) - 1)
        old_val = row[pos]
        
        # Try to find a better value for this position
        best_new_val = None
        best_delta = 0
        
        # Generate candidate values
        candidates = set()
        for d in range(-15, 16):
            candidates.add(old_val + d)
        # Add some random larger moves
        for _ in range(5):
            candidates.add(old_val + rng.randint(-30, 30))
        
        for new_val in sorted(candidates, key=lambda v: abs(v - old_val)):
            if new_val <= 0:
                continue
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
            
            # Compute new row diffs
            new_row = row.copy()
            new_row[pos] = new_val
            new_row_diffs = set()
            for i in range(len(new_row)):
                for j in range(i + 1, len(new_row)):
                    new_row_diffs.add(new_row[j] - new_row[i])
            
            # Check against other rows
            old_row_diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    old_row_diffs.add(row[j] - row[i])
            
            other_diffs = all_diffs - old_row_diffs
            
            if new_row_diffs & other_diffs:
                continue
            
            new_scope = max(max(new_row), max(max(r) for i, r in enumerate(current) if i != r_idx))
            delta = new_scope - best_scope
            
            if delta < best_delta:
                best_delta = delta
                best_new_val = new_val
        
        if best_new_val is not None:
            row[pos] = best_new_val
            # Recompute all_diffs periodically
            if rng.random() < 0.05:
                all_diffs = set()
                for r in current:
                    for i in range(len(r)):
                        for j in range(i + 1, len(r)):
                            all_diffs.add(r[j] - r[i])
            
            current_scope = _scope(current)
            if current_scope < best_scope:
                best_scope = current_scope
                best = [list(r) for r in current]
        
        T *= cooling
    
    return best, best_scope

def construct(n_rows, k):
    """Construct difference triangle set with minimal scope."""
    deadline = time.time() + 145.0
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multi-start greedy
    modes = ['nearest', 'random', 'spread', 'nearest', 'random', 'nearest', 'spread', 'random']
    for mode in modes:
        if time.time() > deadline - 40:
            break
        rows = _greedy(n_rows, k, rng, mode=mode)
        if _validate(rows):
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
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
    
    # Phase 2: Local compaction
    time_left = deadline - time.time()
    if time_left > 10:
        compact_budget = min(20.0, time_left * 0.25)
        best_rows, best_scope = _compact_rows(best_rows, n_rows, k, rng, time_budget=compact_budget)
    
    # Phase 3: Simulated annealing
    time_left = deadline - time.time()
    if time_left > 15:
        sa_budget = min(70.0, time_left - 5)
        best_rows, best_scope = _anneal(best_rows, n_rows, k, rng, time_budget=sa_budget)
    
    # Phase 4: Final compaction
    time_left = deadline - time.time()
    if time_left > 8:
        best_rows, best_scope = _compact_rows(best_rows, n_rows, k, rng, time_budget=time_left - 5)
    
    return best_rows

# EVOLVE-BLOCK-END
