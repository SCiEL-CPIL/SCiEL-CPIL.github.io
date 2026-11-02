"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _row_diff_mask(row):
    """Return bitmask of all pairwise differences in a row. Also returns list."""
    n = len(row)
    mask = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            mask |= (1 << d)
    return mask

def _row_diff_list(row):
    """Return list of all pairwise differences (for Sidon check)."""
    n = len(row)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(row[j] - row[i])
    return out

def _is_sidon(row):
    """Check if all pairwise differences in row are distinct."""
    n = len(row)
    seen = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in seen:
                return False
            seen.add(d)
    return True

def _greedy_random(n_rows, k, rng):
    """Randomized greedy: each row picks smallest valid next value, with random tie-breaking."""
    used_mask = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            # Find smallest x > row[-1] such that all (x - a) for a in row are unused
            x = row[-1] + 1
            while True:
                # Check if x - a collides for any a in row
                ok = True
                for a in row:
                    d = x - a
                    if (used_mask >> d) & 1:
                        ok = False
                        break
                if ok:
                    # Also check internal Sidon: no two (x-a1) == (x-a2) which is automatic since a1!=a2
                    row.append(x)
                    for a in row[:-1]:
                        d = x - a
                        used_mask |= (1 << d)
                    break
                x += 1
                # Safety: don't go too far
                if x > 5000:
                    return None
        rows.append(row)
    return rows

def _greedy_ordered(n_rows, k, order):
    """Greedy with a fixed row order for diversity."""
    used_mask = 0
    rows = [None] * n_rows
    for idx in order:
        row = [0]
        while len(row) < k + 1:
            x = row[-1] + 1
            while True:
                ok = True
                for a in row:
                    d = x - a
                    if (used_mask >> d) & 1:
                        ok = False
                        break
                if ok:
                    row.append(x)
                    for a in row[:-1]:
                        d = x - a
                        used_mask |= (1 << d)
                    break
                x += 1
                if x > 5000:
                    return None
        rows[idx] = row
    return rows

def _compress_row(row, k, used_mask, own_mask):
    """Try to compress a row: move each element (from right to left) to smallest valid position."""
    # We want to minimize the last element primarily
    # Work from position k down to 1
    for pos in range(k, 0, -1):
        # Try to reduce row[pos]
        # The new value must be > row[pos-1] and < row[pos+1] (if pos < k)
        lower = row[pos - 1] + 1
        upper = row[pos]  # try from current downward
        if pos < k:
            upper = min(upper, row[pos + 1] - 1)
        
        best_new = row[pos]
        for new_val in range(lower, upper + 1):
            # Build candidate row
            cand = row[:]
            cand[pos] = new_val
            # Check Sidon
            if not _is_sidon(cand):
                continue
            # Check diff mask conflict
            new_mask = _row_diff_mask(cand)
            # Diffs that were in old row but not in new: those become "free"
            # Diffs in new but not in old: those must not be in (used_mask - own_mask)
            # Actually: the global used_mask includes this row's diffs.
            # After change, global = (used_mask - old_row_mask) | new_row_mask
            # Conflict if new_row_mask & (used_mask - old_row_mask) != 0
            old_mask = _row_diff_mask(row)
            if new_mask & (used_mask - old_mask):
                continue
            # Valid!
            best_new = new_val
            break  # take the smallest valid
        
        row[pos] = best_new
    
    return row

def _local_search_bitset(rows, n_rows, k, deadline):
    """Local search using bitset difference tracking. Targeted compression strategy."""
    cur = [r[:] for r in rows]
    # Build global mask and per-row masks
    row_masks = []
    global_mask = 0
    for r in cur:
        m = _row_diff_mask(r)
        row_masks.append(m)
        global_mask |= m
    
    cur_val = max(max(r) for r in cur)
    best = [r[:] for r in cur]
    best_val = cur_val
    
    rng = random.Random(123)
    
    while time.time() < deadline:
        # Find the row with the maximum last element
        max_row_idx = 0
        max_last = cur[0][-1]
        for i in range(n_rows):
            if cur[i][-1] > max_last:
                max_last = cur[i][-1]
                max_row_idx = i
        
        # Try to compress the max row
        old_mask = row_masks[max_row_idx]
        compressed = _compress_row(cur[max_row_idx][:], k, global_mask, old_mask)
        
        if compressed[-1] < cur[max_row_idx][-1]:
            # Check full validity
            new_mask = _row_diff_mask(compressed)
            if new_mask & (global_mask - old_mask):
                # Conflict, skip
                pass
            else:
                # Update
                global_mask = (global_mask - old_mask) | new_mask
                cur[max_row_idx] = compressed
                row_masks[max_row_idx] = new_mask
                cur_val = max(max(r) for r in cur)
                if cur_val < best_val:
                    best_val = cur_val
                    best = [r[:] for r in cur]
        
        # Also try random single-element moves for diversification
        if rng.random() < 0.3:
            ri = rng.randrange(n_rows)
            pi = rng.randrange(1, k + 1)
            old_val = cur[ri][pi]
            lower = cur[ri][pi - 1] + 1
            upper = cur[ri][pi + 1] - 1 if pi < k else old_val + 10
            # Try a few random values
            for _ in range(5):
                if upper <= lower:
                    break
                new_val = rng.randint(lower, upper)
                if new_val == old_val:
                    continue
                cand = cur[ri][:]
                cand[pi] = new_val
                if not _is_sidon(cand):
                    continue
                new_mask = _row_diff_mask(cand)
                old_m = row_masks[ri]
                if new_mask & (global_mask - old_m):
                    continue
                # Accept if it reduces max or is equal
                cand_max = max(max(r) for r in cur)
                if cand_max <= cur_val:
                    global_mask = (global_mask - old_m) | new_mask
                    cur[ri] = cand
                    row_masks[ri] = new_mask
                    cur_val = max(max(r) for r in cur)
                    if cur_val < best_val:
                        best_val = cur_val
                        best = [r[:] for r in cur]
                    break
        
        # Check if we've hit the theoretical minimum (all diffs 1..n_rows*k*(k+1)/2 used)
        total_diffs = n_rows * k * (k + 1) // 2
        if global_mask == (1 << total_diffs) - 1:
            break
        
        # Stagnation check: if best hasn't improved in a while, do a random restart
        # (simplified: just continue, the compression will naturally explore)
    
    return best, best_val

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

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    random.seed(42)
    
    total_diffs = n_rows * k * (k + 1) // 2
    
    # Multi-start: try different greedy strategies
    best = None
    best_val = float('inf')
    
    # Start 1: standard greedy
    rows1 = _greedy_random(n_rows, k, random.Random(42))
    if rows1 is not None:
        v = max(max(r) for r in rows1)
        if v < best_val:
            best_val = v
            best = rows1
    
    # Start 2: reversed order greedy
    order = list(range(n_rows - 1, -1, -1))
    rows2 = _greedy_ordered(n_rows, k, order)
    if rows2 is not None:
        v = max(max(r) for r in rows2)
        if v < best_val:
            best_val = v
            best = rows2
    
    # Start 3: random order
    order3 = list(range(n_rows))
    random.Random(99).shuffle(order3)
    rows3 = _greedy_ordered(n_rows, k, order3)
    if rows3 is not None:
        v = max(max(r) for r in rows3)
        if v < best_val:
            best_val = v
            best = rows3
    
    # Start 4: another random order
    order4 = list(range(n_rows))
    random.Random(777).shuffle(order4)
    rows4 = _greedy_ordered(n_rows, k, order4)
    if rows4 is not None:
        v = max(max(r) for r in rows4)
        if v < best_val:
            best_val = v
            best = rows4
    
    # Local search on the best starting point
    if time.time() < deadline:
        best, best_val = _local_search_bitset(best, n_rows, k, deadline)
    
    # Final validation
    if not _valid(best, n_rows, k):
        # Fallback to simple greedy
        used = set()
        best = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            best.append(row)
    
    return best
# EVOLVE-BLOCK-END
