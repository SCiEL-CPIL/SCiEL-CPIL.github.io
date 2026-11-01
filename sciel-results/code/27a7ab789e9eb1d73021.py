"""Multi-start randomized greedy with local repair for difference triangle sets."""
import random
import time

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    d = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            d.add(row[j] - row[i])
    return d

def _all_diffs(rows):
    """Return the union of all differences across all rows."""
    d = set()
    for r in rows:
        d |= _row_diffs(r)
    return d

def _is_valid(rows):
    """Check that all differences are distinct across all rows."""
    seen = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                diff = r[j] - r[i]
                if diff in seen:
                    return False
                seen.add(diff)
    return True

def _max_entry(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)

def _greedy_construct(n_rows, k, used, rng=None, perturb_prob=0.0):
    """
    Greedily construct rows. If perturb_prob > 0, with that probability
    skip the first valid candidate and take the second, to diversify.
    """
    if rng is None:
        rng = random.Random()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                # Decide whether to take this candidate or skip to next
                if perturb_prob > 0 and rng.random() < perturb_prob:
                    # Try to find a second valid candidate to skip the first
                    x2 = x
                    found = False
                    for _skip in range(20):  # bounded skip
                        x2 += 1
                        new2 = {x2 - a for a in row}
                        if not new2 & used:
                            # Take the second one
                            row.append(x2)
                            used |= new2
                            x = x2
                            found = True
                            break
                    if not found:
                        # Fall back to first
                        row.append(x)
                        used |= new
                else:
                    row.append(x)
                    used |= new
        rows.append(row)
    return rows

def _try_repair(rows, used, k, max_iter=200, rng=None):
    """
    Try to reduce the maximum entry by replacing the largest element in some row
    with a smaller valid one. Returns (rows, used, improved).
    """
    if rng is None:
        rng = random.Random()
    improved = False
    for _ in range(max_iter):
        # Find the row with the largest last element
        max_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
        row = rows[max_idx]
        largest = row[-1]
        # Try to replace largest with a smaller value
        # The row without its last element:
        row_prefix = row[:-1]
        # Differences involving the last element:
        last_diffs = {largest - a for a in row_prefix}
        # We need to find x < largest such that:
        # 1. x > row_prefix[-1] (strictly increasing)
        # 2. {x - a for a in row_prefix} are all not in (used - last_diffs)
        # 3. x - a for a in row_prefix are all distinct from each other (they are, since a are distinct)
        # 4. x - a for a in row_prefix don't collide with differences within row_prefix
        prefix_diffs = _row_diffs(row_prefix)
        # 5. x - a for a in row_prefix don't collide with other rows' diffs
        other_diffs = used - last_diffs
        # So we need: for all a in row_prefix, (x - a) not in (other_diffs | prefix_diffs)
        forbidden = other_diffs | prefix_diffs
        # Try values from largest-1 down to row_prefix[-1]+1
        for x in range(largest - 1, row_prefix[-1], -1):
            new_diffs = {x - a for a in row_prefix}
            if not new_diffs & forbidden:
                # Valid replacement
                rows[max_idx] = row_prefix + [x]
                used = (used - last_diffs) | new_diffs
                improved = True
                break
        if not improved:
            break
    return rows, used, improved

def construct(n_rows, k):
    """
    Multi-start randomized greedy with local repair.
    Returns list of n_rows rows, each a list of k+1 increasing integers starting at 0.
    """
    deadline = time.time() + 140.0  # safety margin before 150s limit
    rng = random.Random(42)
    
    best_rows = None
    best_max = float('inf')
    
    # Start with deterministic greedy
    used = set()
    rows = _greedy_construct(n_rows, k, used, rng, perturb_prob=0.0)
    if _is_valid(rows):
        best_rows = [list(r) for r in rows]
        best_max = _max_entry(rows)
    
    # Multi-start with randomization
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        # Vary perturbation probability
        p = min(0.3, 0.05 * attempt) if attempt <= 6 else 0.3
        used = set()
        rows = _greedy_construct(n_rows, k, used, rng, perturb_prob=p)
        
        if not _is_valid(rows):
            continue
        
        cur_max = _max_entry(rows)
        
        # Try local repair
        rows, used, _ = _try_repair(rows, used, k, max_iter=100, rng=rng)
        
        if _is_valid(rows):
            cur_max = _max_entry(rows)
            if cur_max < best_max:
                best_max = cur_max
                best_rows = [list(r) for r in rows]
        
        # Early stop if we hit the counting bound
        # Counting bound: n_rows * k*(k+1)/2 distinct differences needed,
        # so max entry >= ceil(sqrt(2 * n_rows * k*(k+1)/2)) roughly,
        # but the simple counting bound given is 105 for (7,5).
        if best_max <= 105:
            break
    
    if best_rows is None:
        # Fallback: deterministic greedy
        used = set()
        best_rows = _greedy_construct(n_rows, k, used, rng, perturb_prob=0.0)
    
    return best_rows

# EVOLVE-BLOCK-END
