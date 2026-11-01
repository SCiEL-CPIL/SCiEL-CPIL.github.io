import time
import random
import math
from collections import Counter

_DEADLINE = 145.0

def _row_diffs(row):
    d = []
    for i in range(len(row)):
        ai = row[i]
        for j in range(i + 1, len(row)):
            d.append(row[j] - ai)
    return d

def _build_counter(rows):
    c = Counter()
    for row in rows:
        for d in _row_diffs(row):
            c[d] += 1
    return c

def _conflicts_from_counter(c):
    return sum(v - 1 for v in c.values() if v > 1)

def _scope(rows):
    return max(r[-1] for r in rows)

def _energy(rows, counter):
    conf = _conflicts_from_counter(counter)
    if conf > 0:
        return 1000 * conf + _scope(rows)
    else:
        return _scope(rows)

def _greedy_row_randomized(used, k, rng, rand_factor=0.1):
    row = [0]
    x = 0
    while len(row) < k + 1:
        valid_x = None
        candidates = []
        start_x = x + 1
        for trial_x in range(start_x, start_x + 2000):
            new = {trial_x - a for a in row}
            if not (new & used):
                candidates.append(trial_x)
                if len(candidates) > 10:
                    break
        if not candidates:
            trial_x = start_x
            while True:
                new = {trial_x - a for a in row}
                if not (new & used):
                    candidates.append(trial_x)
                    break
                trial_x += 1
        
        if rand_factor > 0 and len(candidates) > 1:
            x = rng.choice(candidates[:3])
        else:
            x = candidates[0]
            
        row.append(x)
        used |= {x - a for a in row[:-1]}
    return row

def _greedy_all_randomized(n_rows, k, rng, rand_factor=0.2):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row_randomized(used, k, rng, rand_factor)
        rows.append(row)
        used |= set(_row_diffs(row))
    return rows

def _sa_optimize(rows, k, deadline, rng):
    """
    Simulated annealing to minimize the maximum entry.
    Uses a different approach: coordinate moves on multiple positions, 
    and use a temperature schedule to escape local optima.
    """
    t0 = time.time()
    cur_rows = [list(r) for r in rows]
    cur_counter = _build_counter(cur_rows)
    
    best_rows = [list(r) for r in cur_rows]
    best_scope = _scope(cur_rows)
    best_conf = _conflicts_from_counter(cur_counter)
    
    n_rows = len(cur_rows)
    row_len = k + 1
    
    # Temperature schedule
    T = 50.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline:
        # Pick a random row and position
        r_idx = rng.randrange(n_rows)
        pos = rng.randrange(1, row_len)
        row = cur_rows[r_idx]
        old_val = row[pos]
        
        lo = row[pos - 1] + 1
        hi = (row[pos + 1] - 1) if pos + 1 < row_len else old_val + 20
        
        if hi <= lo:
            continue
        
        # Generate new value
        if rng.random() < 0.3:
            # Try to reduce
            new_val = rng.randint(lo, min(hi, old_val - 1)) if old_val > lo else lo
        else:
            new_val = rng.randint(lo, hi)
        
        if new_val == old_val:
            continue
        
        # Compute old and new differences
        old_d = []
        new_d = []
        for j in range(row_len):
            if j != pos:
                if j < pos:
                    old_d.append(old_val - row[j])
                    new_d.append(new_val - row[j])
                else:
                    old_d.append(row[j] - old_val)
                    new_d.append(row[j] - new_val)
        
        old_d_counts = Counter(old_d)
        new_d_counts = Counter(new_d)
        
        # Check validity and compute energy change
        valid = True
        delta_conf = 0
        for d, cnt in new_d_counts.items():
            cur_cnt = cur_counter.get(d, 0)
            rem_cnt = old_d_counts.get(d, 0)
            net_cnt = cur_cnt - rem_cnt + cnt
            if net_cnt > 1:
                # This would create a conflict
                delta_conf += net_cnt - 1
                # But we need to check if it was already conflicted
                # Actually, let's just check if it's valid
                if net_cnt > 1:
                    valid = False
                    break
        
        if not valid:
            # Check if we should accept based on energy
            new_scope = max(_scope(cur_rows), max(new_val, row[-1] if pos < row_len-1 else 0))
            # This is approximate; let's just reject for simplicity in SA
            continue
        
        # Apply the move
        for d in old_d:
            cur_counter[d] -= 1
            if cur_counter[d] == 0:
                del cur_counter[d]
        for d in new_d:
            cur_counter[d] = cur_counter.get(d, 0) + 1
        row[pos] = new_val
        
        new_scope = _scope(cur_rows)
        new_conf = _conflicts_from_counter(cur_counter)
        
        # Accept if better or with probability based on temperature
        if new_conf == 0 and new_scope < best_scope:
            best_scope = new_scope
            best_rows = [list(r) for r in cur_rows]
        
        # Cooling
        T = max(T_min, T * cooling)
    
    return best_rows

def _coordinated_descent(rows, k, deadline, rng):
    """
    Coordinated descent: try to reduce the maximum by moving multiple positions
    in the row with the largest last element, considering their joint effect.
    """
    t0 = time.time()
    cur_rows = [list(r) for r in rows]
    cur_counter = _build_counter(cur_rows)
    
    best_rows = [list(r) for r in cur_rows]
    best_scope = _scope(cur_rows)
    
    n_rows = len(cur_rows)
    row_len = k + 1
    
    while time.time() < deadline:
        # Find row with max last element
        max_idx = max(range(n_rows), key=lambda i: cur_rows[i][-1])
        max_row = cur_rows[max_idx]
        
        # Try to reduce the last few elements together
        # Focus on positions near the end
        for start_pos in range(max(1, row_len - 3), row_len):
            # Try to reduce elements from start_pos to end
            for pos in range(start_pos, row_len):
                old_val = max_row[pos]
                lo = max_row[pos - 1] + 1
                hi = (max_row[pos + 1] - 1) if pos + 1 < row_len else old_val - 1
                
                if hi < lo:
                    continue
                
                # Try a few candidate values
                candidates = []
                if hi - lo <= 20:
                    candidates = list(range(hi, lo - 1, -1))
                else:
                    for _ in range(8):
                        candidates.append(rng.randint(lo, hi))
                    candidates.append(hi)
                    candidates.sort(reverse=True)
                
                for new_val in candidates:
                    if new_val >= old_val:
                        continue
                    
                    # Check validity
                    old_d = []
                    new_d = []
                    for j in range(row_len):
                        if j != pos:
                            if j < pos:
                                old_d.append(old_val - max_row[j])
                                new_d.append(new_val - max_row[j])
                            else:
                                old_d.append(max_row[j] - old_val)
                                new_d.append(max_row[j] - new_val)
                    
                    old_d_counts = Counter(old_d)
                    new_d_counts = Counter(new_d)
                    
                    valid = True
                    for d, cnt in new_d_counts.items():
                        cur_cnt = cur_counter.get(d, 0)
                        rem_cnt = old_d_counts.get(d, 0)
                        net_cnt = cur_cnt - rem_cnt + cnt
                        if net_cnt > 1:
                            valid = False
                            break
                    
                    if valid:
                        # Apply
                        for d in old_d:
                            cur_counter[d] -= 1
                            if cur_counter[d] == 0:
                                del cur_counter[d]
                        for d in new_d:
                            cur_counter[d] = cur_counter.get(d, 0) + 1
                        max_row[pos] = new_val
                        
                        new_scope = _scope(cur_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in cur_rows]
                        break
        
        # Occasionally try a random move to escape local optima
        if rng.random() < 0.1:
            r_idx = rng.randrange(n_rows)
            pos = rng.randrange(1, row_len)
            row = cur_rows[r_idx]
            old_val = row[pos]
            lo = row[pos - 1] + 1
            hi = (row[pos + 1] - 1) if pos + 1 < row_len else old_val + 15
            
            if hi > lo:
                new_val = rng.randint(lo, hi)
                if new_val != old_val:
                    old_d = []
                    new_d = []
                    for j in range(row_len):
                        if j != pos:
                            if j < pos:
                                old_d.append(old_val - row[j])
                                new_d.append(new_val - row[j])
                            else:
                                old_d.append(row[j] - old_val)
                                new_d.append(row[j] - new_val)
                    
                    old_d_counts = Counter(old_d)
                    new_d_counts = Counter(new_d)
                    
                    valid = True
                    for d, cnt in new_d_counts.items():
                        cur_cnt = cur_counter.get(d, 0)
                        rem_cnt = old_d_counts.get(d, 0)
                        net_cnt = cur_cnt - rem_cnt + cnt
                        if net_cnt > 1:
                            valid = False
                            break
                    
                    if valid:
                        for d in old_d:
                            cur_counter[d] -= 1
                            if cur_counter[d] == 0:
                                del cur_counter[d]
                        for d in new_d:
                            cur_counter[d] = cur_counter.get(d, 0) + 1
                        row[pos] = new_val
                        
                        new_scope = _scope(cur_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in cur_rows]
    
    return best_rows

def construct(n_rows, k):
    rng = random.Random(12345)
    
    # Generate multiple random greedy starts
    best_start = None
    best_start_energy = float('inf')
    
    for i in range(30):
        rf = 0.05 * (i + 1)
        rows = _greedy_all_randomized(n_rows, k, rng, rf)
        counter = _build_counter(rows)
        conf = _conflicts_from_counter(counter)
        scope = _scope(rows)
        energy = 1000 * conf + scope
        
        if energy < best_start_energy:
            best_start_energy = energy
            best_start = [list(r) for r in rows]
            
    if not best_start:
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
        best_start = rows

    # Use coordinated descent instead of targeted descent
    t0 = time.time()
    optimized_rows = _coordinated_descent(best_start, k, _DEADLINE, rng)
    
    # Final check
    if not _valid(optimized_rows):
        if _valid(best_start):
            return best_start
        else:
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

    return optimized_rows

def _valid(rows):
    seen = set()
    for row in rows:
        for d in _row_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True
