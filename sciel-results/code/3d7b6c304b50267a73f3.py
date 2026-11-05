# EVOLVE-BLOCK-START
import sys
import time
import random
import math

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    
    Uses a structural change: Simulated Annealing with element-level moves
    instead of full-row rebuilds. This allows finer-grained exploration
    that can escape local minima where full-row rebuilds fail.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin before 120s limit
    
    def time_up():
        return time.time() > deadline

    # Helper to compute all positive differences in a row
    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def get_max(rows):
        return max(max(r) for r in rows)

    # Initial construction using greedy
    def greedy_construct(seed):
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            
            for _ in range(k):
                last = row[-1]
                min_x = last + 1
                x = min_x
                
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and not (new_diffs & local_used):
                        break
                    x += 1
                
                row.append(x)
                local_used.update(new_diffs)
            
            rows.append(row)
            used.update(local_used)
        
        return rows, used

    # Try several seeds for initial solution
    best_rows = None
    best_max = float('inf')
    
    for seed in range(10):
        if time_up():
            break
        rows, used = greedy_construct(seed)
        m = get_max(rows)
        if m < best_max:
            best_max = m
            best_rows = [r[:] for r in rows]
    
    if best_rows is None:
        # Fallback
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            local_used = set()
            for _ in range(k):
                last = row[-1]
                x = last + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used) and not (new_diffs & local_used):
                        break
                    x += 1
                row.append(x)
                local_used.update(new_diffs)
            rows.append(row)
            used.update(local_used)
        best_rows = rows
        best_max = get_max(rows)

    # Lower bound check
    lb = max(k, n_rows * k * (k + 1) // 2)
    if best_max == lb:
        return best_rows

    # Simulated Annealing with element-level moves
    rng = random.Random(123)
    current_rows = [r[:] for r in best_rows]
    current_max = best_max
    
    # Build used differences set
    current_used = set()
    for row in current_rows:
        current_used.update(get_diffs(row))
    
    # Temperature schedule
    T = 10.0
    T_min = 0.1
    cooling = 0.9995
    
    def try_move():
        """
        Try a random element-level move:
        1. Pick a random row
        2. Pick a random position in that row (not first)
        3. Try to change the value at that position by a small delta
        """
        nonlocal current_rows, current_used, current_max
        
        # Pick a random row
        r_idx = rng.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        
        # Pick a random position (not 0, since row[0] must be 0)
        pos = rng.randint(1, len(row) - 1)
        
        old_val = row[pos]
        
        # Generate a new value: old_val + delta where delta is small
        # Ensure strict increasing order is maintained
        min_val = row[pos - 1] + 1
        max_val = row[pos + 1] - 1 if pos < len(row) - 1 else current_max
        
        if min_val > max_val:
            return False
        
        # Try several delta values
        for _ in range(20):
            delta = rng.randint(-5, 5)
            new_val = old_val + delta
            
            if new_val < min_val or new_val > max_val:
                continue
            
            # Check if this move is valid
            # Remove old differences involving this element
            old_diffs = set()
            for i in range(len(row)):
                if i != pos:
                    d = abs(row[pos] - row[i])
                    old_diffs.add(d)
            
            # Temporarily remove old diffs
            current_used.difference_update(old_diffs)
            
            # Check if new value creates any conflicts
            valid = True
            new_diffs = set()
            for i in range(len(row)):
                if i != pos:
                    d = abs(new_val - row[i])
                    if d in current_used:
                        valid = False
                        break
                    new_diffs.add(d)
            
            if not valid:
                # Restore old diffs
                current_used.update(old_diffs)
                continue
            
            # Check if new diffs conflict with each other
            if len(new_diffs) != len(set(new_diffs)):
                current_used.update(old_diffs)
                continue
            
            # Calculate new max
            new_max = max(row) if max(row) != old_val or new_val < old_val else max(new_val, row[0], row[-1])
            # More carefully:
            new_max = 0
            for v in row:
                if i == pos:
                    if v > new_max:
                        new_max = v
                else:
                    if v > new_max:
                        new_max = v
            if new_val > new_max:
                new_max = new_val
            
            # Acceptance criterion
            if new_max <= current_max:
                # Always accept if not worse
                row[pos] = new_val
                current_used.update(new_diffs)
                if new_max < current_max:
                    current_max = new_max
                    best_rows = [r[:] for r in current_rows]
                return True
            else:
                # Accept with probability exp(-(new_max - current_max) / T)
                prob = math.exp(-(new_max - current_max) / T)
                if rng.random() < prob:
                    row[pos] = new_val
                    current_used.update(new_diffs)
                    current_max = new_max
                    if current_max > best_max:
                        # Revert if worse than best
                        row[pos] = old_val
                        current_used.difference_update(new_diffs)
                        current_used.update(old_diffs)
                        current_max = best_max
                    return True
                else:
                    # Reject, restore
                    current_used.update(old_diffs)
                    continue
        
        return False

    # Main SA loop
    while not time_up() and T > T_min:
        try_move()
        T *= cooling
    
    # Final validation
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    if not validate(best_rows, n_rows, k):
        # Fallback
        res = greedy_construct(0)
        if res:
            best_rows, _ = res

    return best_rows

# EVOLVE-BLOCK-END
