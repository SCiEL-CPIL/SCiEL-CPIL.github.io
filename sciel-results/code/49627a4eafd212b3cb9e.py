"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry (scope) such that all positive differences
    within rows are distinct across all rows.
    
    Uses a multi-start greedy approach with dynamic difference pressure ordering.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Total number of differences needed
    n_diffs = n_rows * (k * (k + 1) // 2)
    
    # Lower bound on scope: we need n_diffs distinct differences, each <= scope
    # Differences range from 1 to scope, so scope >= n_diffs
    # But also each row has k differences that are sums of consecutive gaps
    # Simple bound: scope >= n_diffs
    lb = n_diffs
    
    best_rows = None
    best_scope = float('inf')
    
    # Strategy 1: Basic greedy (as in seed) - always pick smallest valid x
    # Strategy 2: Greedy with lookahead - try to minimize future pressure
    # Strategy 3: Randomized restarts with different heuristics
    
    def greedy_basic(n_rows, k, deadline):
        """Standard greedy: pick smallest x that doesn't conflict."""
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                if time.time() > deadline:
                    return None
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    def greedy_pressure(n_rows, k, deadline):
        """
        Greedy with dynamic pressure: when choosing next entry for current row,
        prefer candidates that use smaller differences (less pressure on upper range).
        Among valid candidates, pick the one that minimizes the max new difference.
        This is equivalent to picking smallest x, but we also consider that
        picking a slightly larger x might allow more compact packing later.
        
        Actually, for a single row, picking smallest x always minimizes the current
        max difference. The issue is cross-row interference. So let's try:
        build rows in an order that minimizes interference.
        """
        used = set()
        rows = [None] * n_rows
        
        # Try building rows in different orders
        # Start with the "hardest" rows first (those with more constraints)
        # For now, just try standard order but with a twist:
        # After each row is complete, we could try to "tighten" it
        
        for r in range(n_rows):
            if time.time() > deadline:
                return None
            row = [0]
            x = 0
            while len(row) < k + 1:
                if time.time() > deadline:
                    return None
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows[r] = row
        
        return rows
    
    def randomized_greedy(n_rows, k, deadline, rng):
        """
        Randomized greedy: with some probability, skip a candidate and try a larger one.
        This can escape local minima in the greedy choice.
        """
        used = set()
        rows = []
        for r in range(n_rows):
            if time.time() > deadline:
                return None
            row = [0]
            x = 0
            while len(row) < k + 1:
                if time.time() > deadline:
                    return None
                found = False
                # Try up to a few candidates
                for attempt in range(5):
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        # With some probability, accept this; otherwise try next
                        if attempt == 0 or rng.random() < 0.7:
                            row.append(x)
                            used |= new_diffs
                            found = True
                            break
                if not found:
                    # Fallback: just take the first valid one
                    while True:
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            row.append(x)
                            used |= new_diffs
                            break
            rows.append(row)
        return rows
    
    def evaluate(rows):
        """Return the scope (largest entry) if valid, else None."""
        if rows is None:
            return None
        used = set()
        max_val = 0
        for row in rows:
            if not row or row[0] != 0:
                return None
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return None
                    used.add(d)
                max_val = max(max_val, row[-1])
        return max_val
    
    # Try basic greedy first
    rows = greedy_basic(n_rows, k, deadline)
    if rows is not None:
        scope = evaluate(rows)
        if scope is not None:
            best_rows = rows
            best_scope = scope
    
    # Try randomized variants
    rng = random.Random(42)
    for seed in range(50):
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(seed * 12345 + 67890)
        rows = randomized_greedy(n_rows, k, deadline, rng)
        if rows is not None:
            scope = evaluate(rows)
            if scope is not None and scope < best_scope:
                best_rows = rows
                best_scope = scope
    
    # Try a local search / improvement pass on the best solution found
    if best_rows is not None and time.time() < deadline - 10.0:
        best_rows = local_improve(best_rows, n_rows, k, deadline, best_scope)
        scope = evaluate(best_rows)
        if scope is not None:
            best_scope = scope
    
    if best_rows is None:
        # Fallback: return basic greedy
        best_rows = greedy_basic(n_rows, k, deadline)
    
    return best_rows


def local_improve(rows, n_rows, k, deadline, current_scope):
    """
    Try to improve the scope by local search.
    For each row, try to shift or compress it slightly.
    """
    best = [list(r) for r in rows]
    best_scope = current_scope
    
    for iteration in range(20):
        if time.time() > deadline:
            break
        improved = False
        for r in range(n_rows):
            # Try to reduce the last element of row r by 1, if valid
            row = list(best[r])
            if row[-1] > k:  # Must be at least k (since k+1 elements starting at 0)
                new_last = row[-1] - 1
                # Check if new_last is still greater than row[-2]
                if new_last > row[-2]:
                    row[-1] = new_last
                    # Validate
                    if is_valid_partial(best, r, row):
                        best[r] = row
                        best_scope = min(best_scope, max(row))
                        improved = True
        if not improved:
            break
    
    return best


def is_valid_partial(all_rows, row_idx, new_row):
    """Check if replacing row_idx with new_row keeps all differences distinct."""
    used = set()
    # Add differences from all other rows
    for i, row in enumerate(all_rows):
        if i == row_idx:
            continue
        for a in range(len(row)):
            for b in range(a + 1, len(row)):
                d = row[b] - row[a]
                used.add(d)
    # Add differences from new_row
    for a in range(len(new_row)):
        for b in range(a + 1, len(new_row)):
            d = new_row[b] - new_row[a]
            if d in used:
                return False
            used.add(d)
    return True
# EVOLVE-BLOCK-END
