"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within rows are distinct across all rows.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Total number of differences per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs_needed = n_rows * diffs_per_row

    # Counting bound: smallest m such that m >= total_diffs_needed
    # m must satisfy m*(m+1)/2 >= total_diffs_needed
    m_lower = 0
    while m_lower * (m_lower + 1) // 2 < total_diffs_needed:
        m_lower += 1

    # We'll try to find a solution with scope close to m_lower
    # Strategy: randomized greedy with multiple restarts, row-order shuffling,
    # and biased candidate selection

    best_rows = None
    best_scope = float('inf')

    # Upper bound for search: start from a reasonable greedy solution
    # First, get a quick greedy solution to establish an upper bound
    greedy_rows, greedy_scope = _greedy_construct(n_rows, k)
    best_rows = greedy_rows
    best_scope = greedy_scope

    # Now try to improve with randomized approaches
    rng = random.Random(42)

    max_scope_to_try = best_scope  # we only care about improving

    # Try different target scopes from lower bound up to best_scope - 1
    for target_scope in range(max(m_lower, 1), best_scope):
        if time.time() > deadline - 10.0:
            break

        # Try multiple randomized attempts for this target scope
        attempts = 50
        for attempt in range(attempts):
            if time.time() > deadline - 5.0:
                break

            rows, scope = _randomized_construct(n_rows, k, target_scope, rng, attempt)
            if rows is not None:
                if scope < best_scope:
                    best_scope = scope
                    best_rows = rows
                # If we found a solution at this target scope, we can stop
                # trying higher target scopes (but continue trying lower ones)
                break

        # If we found a solution at this target scope, no need to try higher ones
        # in this outer loop iteration (we'll try lower scopes next)
        # Actually, we're iterating upward, so if we found one at target_scope,
        # we can break since we want the minimum scope
        if rows is not None:
            break

    return best_rows


def _greedy_construct(n_rows, k):
    """Original greedy: each row takes smallest next mark with unused differences."""
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
    scope = max(max(r) for r in rows)
    return rows, scope


def _row_diffs(row):
    """Compute all positive differences in a row."""
    diffs = set()
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs


def _randomized_construct(n_rows, k, target_scope, rng, seed_offset):
    """
    Try to construct n_rows rows with all entries <= target_scope.
    Uses randomized greedy with row-order shuffling and biased selection.
    """
    rng_local = random.Random(seed_offset * 1000 + 7)
    
    # Shuffle row order
    row_order = list(range(n_rows))
    rng_local.shuffle(row_order)
    
    used = set()
    rows = [None] * n_rows
    
    for row_idx in row_order:
        row = _construct_single_row_randomized(k, target_scope, used, rng_local)
        if row is None:
            return None, None
        rows[row_idx] = row
        used |= _row_diffs(row)
    
    scope = max(max(r) for r in rows)
    if scope <= target_scope:
        return rows, scope
    return None, None


def _construct_single_row_randomized(k, max_val, used, rng):
    """
    Construct a single row of k+1 elements starting at 0, all <= max_val,
    with differences not in 'used'.
    """
    row = [0]
    
    # Try to build the row greedily with some randomization
    current_max = 0
    
    for _ in range(k):
        # Try candidates from current_max+1 to max_val
        # Use a biased selection: prefer smaller values but with some randomness
        
        candidates = []
        for x in range(current_max + 1, max_val + 1):
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                candidates.append(x)
        
        if not candidates:
            return None
        
        # Choose with bias toward smaller values
        # Use exponential weighting
        if len(candidates) == 1:
            choice = candidates[0]
        else:
            # Prefer smaller values with some randomness
            # Temperature-based selection
            temp = 0.1 + 0.3 * rng.random()
            weights = [math.exp(-c * temp / max_val) for c in candidates]
            total_w = sum(weights)
            r = rng.random() * total_w
            cumulative = 0
            choice = candidates[-1]
            for c, w in zip(candidates, weights):
                cumulative += w
                if r <= cumulative:
                    choice = c
                    break
        
        row.append(choice)
        current_max = choice
        used |= {choice - a for a in row[:-1]}
    
    return row

# EVOLVE-BLOCK-END
