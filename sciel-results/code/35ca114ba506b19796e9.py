"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _build_greedy(n_rows, k):
    """Fast greedy: each row takes smallest next mark with no difference collision."""
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

def _valid(rows, n_rows, k):
    """Check all rows are valid and all differences are globally distinct."""
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            if i > 0 and row[i] <= row[i - 1]:
                return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _max_entry(rows):
    return max(max(r) for r in rows)

def _row_diffs(row):
    s = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            s.add(row[j] - row[i])
    return s

def _row_diffs_list(row):
    """Return list of all positive differences in a row (with possible dups across pairs)."""
    out = []
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            out.append(row[j] - row[i])
    return out

def construct(n_rows, k):
    deadline = time.time() + 110  # safety margin before 120s limit
    random.seed(42)

    # Start from greedy
    rows = _build_greedy(n_rows, k)
    best = [r[:] for r in rows]
    best_val = _max_entry(best)

    # Fast validity check for a single row change:
    # Given current global used set (excluding row ri's diffs), check if new row diffs are all free.
    # We maintain: global_diffs = set of all diffs across all rows
    # For row ri, we keep row_diffs[ri] separately.
    
    cur = [r[:] for r in best]
    cur_val = best_val
    
    # Precompute per-row diff sets and global set
    row_diffs = [_row_diffs(r) for r in cur]
    global_diffs = set()
    for s in row_diffs:
        global_diffs |= s
    
    T = 10.0
    T_min = 0.01
    cool = 0.9999
    
    step = 0
    while time.time() < deadline:
        step += 1
        # Pick a row: prefer rows with large max entries
        # Weight by (max_entry - min_max) to focus on reducing the bottleneck
        maxes = [max(r) for r in cur]
        max_max = max(maxes)
        # Choose row: 70% chance pick the row with the current max, else random
        if random.random() < 0.7:
            ri = random.choice([i for i in range(n_rows) if maxes[i] == max_max])
        else:
            ri = random.randrange(n_rows)
        
        # Pick position: prefer later positions (they tend to be larger)
        # 50% chance pick the last position, else random
        if random.random() < 0.5:
            pi = k  # last position
        else:
            pi = random.randrange(1, k + 1)
        
        old_val = cur[ri][pi]
        
        # Try to decrease first (that's what we want)
        # Generate candidate values: try decreasing by various amounts
        candidates = []
        # Try decreasing by 1..min(20, old_val - cur[ri][pi-1] - 1)
        min_allowed = cur[ri][pi - 1] + 1 if pi > 0 else 1
        max_allowed = cur[ri][pi + 1] - 1 if pi < k else old_val
        
        # Generate a list of candidate values to try, sorted by preference (closer to old_val, preferring decreases)
        try_range = min(25, old_val - min_allowed + 1, max_allowed - old_val + 1)
        for d in range(1, try_range + 1):
            if old_val - d >= min_allowed:
                candidates.append(old_val - d)
            if old_val + d <= max_allowed:
                candidates.append(old_val + d)
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val == old_val:
                continue
            if new_val < min_allowed or new_val > max_allowed:
                continue
            
            # Build candidate row
            cand_row = cur[ri][:]
            cand_row[pi] = new_val
            
            # Compute new diffs for this row
            new_diffs = _row_diffs(cand_row)
            
            # Check: new_diffs must not intersect with (global_diffs - old_row_diffs)
            # i.e., the diffs from other rows
            other_diffs = global_diffs - row_diffs[ri]
            if new_diffs & other_diffs:
                continue
            # Also check within new_diffs itself (should be fine since it's a set)
            
            # Compute new max
            new_max = max(cand_row)
            new_val_max = max(maxes[:ri] + [new_max] + maxes[ri+1:]) if n_rows > 1 else new_max
            
            # Acceptance criterion
            if new_val_max < cur_val:
                # Definitely accept
                pass
            elif new_val_max == cur_val and random.random() < 0.5:
                # Accept sideways moves to explore
                pass
            else:
                # SA: accept worse with small probability
                delta_e = new_val_max - cur_val
                if delta_e > 0:
                    p = min(1.0, (1.0 / (1.0 + T)) * 0.05)
                    if random.random() >= p:
                        continue
                else:
                    # Shouldn't happen but accept
                    pass
            
            # Apply the move
            # Update global_diffs
            global_diffs -= row_diffs[ri]
            global_diffs |= new_diffs
            row_diffs[ri] = new_diffs
            cur[ri] = cand_row
            cur_val = new_val_max
            maxes[ri] = new_max
            
            if cur_val < best_val:
                best_val = cur_val
                best = [r[:] for r in cur]
            
            moved = True
            break
        
        T = max(T_min, T * cool)
        
        # Periodic restart: if stuck for too long, try a different strategy
        # (keep it simple for now)
    
    # Ensure best is valid
    if not _valid(best, n_rows, k):
        best = _build_greedy(n_rows, k)
    return best
# EVOLVE-BLOCK-END
