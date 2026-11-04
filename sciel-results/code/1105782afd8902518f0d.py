import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # 20s safety margin
    
    # For the specific task: n_rows=7, k=5, need 7*15=105 distinct differences
    # Each row has k+1=6 entries starting at 0, strictly increasing
    # Differences: for row [a0=0, a1, ..., ak], differences are aj-ai for i<j
    
    best_result = None
    best_m = float('inf')
    
    def time_left():
        return deadline - time.time()
    
    def validate(rows):
        """Check all differences are distinct across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def get_max_entry(rows):
        return max(max(row) for row in rows)
    
    def try_greedy_construction(seed=None):
        """Greedy construction with randomized tie-breaking."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                # Check if adding x creates any conflict
                new_diffs = set()
                conflict = False
                for a in row:
                    d = x - a
                    if d in used:
                        conflict = True
                        break
                    new_diffs.add(d)
                if not conflict:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        
        return rows
    
    def try_construct_with_priority(n_rows, k, seed=None):
        """Construct rows prioritizing coverage of scarce differences."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = set()
                conflict = False
                for a in row:
                    d = x - a
                    if d in used:
                        conflict = True
                        break
                    new_diffs.add(d)
                if not conflict:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        
        return rows
    
    def local_search_improve(rows, seed=None, time_limit=None):
        """Improve rows by reducing max entry via local search."""
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
        
        if time_limit is None:
            time_limit = time_left() - 5.0  # leave 5s margin
        
        if time_limit <= 0:
            return rows
        
        rows = [list(r) for r in rows]
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        
        m = get_max_entry(rows)
        best_rows = [list(r) for r in rows]
        best_m = m
        
        ts = start_time
        while time.time() - ts < time_limit:
            # Try to reduce the maximum entry
            # Find the row with the max entry
            max_row_idx = max(range(len(rows)), key=lambda i: max(rows[i]))
            max_val = max(rows[max_row_idx])
            
            # Try replacing the largest entry with a smaller value
            row = rows[max_row_idx]
            # The largest entry is row[-1]
            old_val = row[-1]
            
            # Compute differences involving old_val
            old_diffs = set()
            for a in row[:-1]:
                old_diffs.add(old_val - a)
            
            # Try smaller values
            found = False
            for new_val in range(old_val - 1, 0, -1):
                if new_val <= row[-2]:
                    continue
                # Check if new_val creates conflicts
                new_diffs = set()
                ok = True
                for a in row[:-1]:
                    d = new_val - a
                    # d is fine if it was in old_diffs (replacing) or not in used
                    if d in old_diffs:
                        new_diffs.add(d)
                    elif d in used:
                        ok = False
                        break
                    else:
                        new_diffs.add(d)
                if ok:
                    # Accept: update used set
                    used -= old_diffs
                    used |= new_diffs
                    row[-1] = new_val
                    found = True
                    break
            
            if not found:
                # Try a random perturbation: swap two entries in a random row
                if len(rows) > 1:
                    r1 = rng.randint(0, len(rows)-1)
                    r2 = rng.randint(0, len(rows)-1)
                    if r1 != r2:
                        # Try to reduce a random entry in r1
                        row1 = rows[r1]
                        idx = rng.randint(1, len(row1)-1)
                        old = row1[idx]
                        # Try smaller
                        for new in range(old-1, row1[idx-1], -1):
                            new_diffs = set()
                            ok = True
                            # Remove old diffs
                            old_diffs = set()
                            for j in range(len(row1)):
                                if j != idx:
                                    if row1[j] < old:
                                        old_diffs.add(old - row1[j])
                                    elif row1[j] > old:
                                        old_diffs.add(row1[j] - old)
                            
                            for j in range(len(row1)):
                                if j == idx:
                                    continue
                                if row1[j] < new:
                                    d = new - row1[j]
                                else:
                                    d = row1[j] - new
                                if d in old_diffs:
                                    new_diffs.add(d)
                                elif d in used:
                                    ok = False
                                    break
                                else:
                                    new_diffs.add(d)
                            if ok:
                                used -= old_diffs
                                used |= new_diffs
                                row1[idx] = new
                                break
                else:
                    break
            
            # Check current m
            cur_m = get_max_entry(rows)
            if cur_m < best_m:
                best_m = cur_m
                best_rows = [list(r) for r in rows]
        
        return best_rows
    
    # Phase 1: Generate initial solutions with different seeds
    initial_rows = try_greedy_construction(seed=42)
    if validate(initial_rows):
        m = get_max_entry(initial_rows)
        best_m = m
        best_result = [list(r) for r in initial_rows]
    
    # Try a few more greedy variants
    for seed in range(10):
        if time_left() < 10:
            break
        rows = try_greedy_construction(seed=seed)
        if validate(rows):
            m = get_max_entry(rows)
            if m < best_m:
                best_m = m
                best_result = [list(r) for r in rows]
    
    # Phase 2: Local search improvement
    if best_result is not None and time_left() > 5:
        improved = local_search_improve(best_result, seed=42)
        if validate(improved):
            m = get_max_entry(improved)
            if m < best_m:
                best_m = m
                best_result = [list(r) for r in improved]
    
    # Phase 3: More local search with different starting points
    for seed in range(20):
        if time_left() < 10:
            break
        rows = try_greedy_construction(seed=100 + seed)
        if not validate(rows):
            continue
        m = get_max_entry(rows)
        if m < best_m:
            best_m = m
            best_result = [list(r) for r in rows]
        
        if time_left() > 5:
            improved = local_search_improve(rows, seed=100 + seed, time_limit=min(5, time_left()-5))
            if validate(improved):
                m2 = get_max_entry(improved)
                if m2 < best_m:
                    best_m = m2
                    best_result = [list(r) for r in improved]
    
    if best_result is None:
        # Fallback: return greedy result even if not optimal
        return initial_rows
    
    return best_result
