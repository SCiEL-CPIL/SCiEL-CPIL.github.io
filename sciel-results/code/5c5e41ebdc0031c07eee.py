"""Rewritten program for the difference triangle set task."""
import random
import sys

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all positive differences within each row are distinct across all rows.
    Minimizes the largest entry.
    
    Uses a randomized local search / simulated annealing approach starting
    from a greedy construction, then trying to reduce the maximum value.
    """
    # First, get a valid solution via greedy
    base_rows = _greedy_construct(n_rows, k)
    if base_rows is None:
        return None
    
    # Try to improve via local search
    best_rows = base_rows
    best_max = max(max(r) for r in base_rows)
    
    # Multiple restarts with randomization
    for attempt in range(200):
        # Start from greedy but with some randomization
        if attempt == 0:
            current_rows = [list(r) for r in base_rows]
        else:
            current_rows = _randomized_greedy(n_rows, k, seed=attempt)
            if current_rows is None:
                continue
        
        current_max = max(max(r) for r in current_rows)
        if current_max < best_max:
            best_max = current_max
            best_rows = [list(r) for r in current_rows]
        
        # Local search: try to reduce the maximum element
        if not _local_search_improve(current_rows, n_rows, k, iterations=500):
            continue
        
        new_max = max(max(r) for r in current_rows)
        if new_max < best_max:
            best_max = new_max
            best_rows = [list(r) for r in current_rows]
        
        if best_max <= _counting_bound(n_rows, k):
            break
    
    return best_rows


def _greedy_construct(n_rows, k):
    """Original greedy construction."""
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


def _randomized_greedy(n_rows, k, seed=0):
    """Greedy with randomization in choice of next element."""
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            # Try a few candidates randomly
            candidates = []
            for trial in range(10):
                candidate = x + rng.randint(0, 5)
                new_diffs = {candidate - a for a in row}
                if not new_diffs & used:
                    candidates.append((candidate, new_diffs))
            if candidates:
                # Pick the smallest valid candidate, or sometimes a random one
                if rng.random() < 0.7:
                    candidates.sort(key=lambda t: t[0])
                    chosen, chosen_diffs = candidates[0]
                else:
                    chosen, chosen_diffs = rng.choice(candidates)
                row.append(chosen)
                used |= chosen_diffs
                x = chosen
            else:
                # Fallback: just try next
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                else:
                    return None
        rows.append(row)
    return rows


def _local_search_improve(rows, n_rows, k, iterations=500):
    """
    Try to reduce the maximum entry by modifying rows.
    Returns True if any improvement was made.
    """
    improved = False
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    
    for _ in range(iterations):
        # Pick a random row and try to reduce its last element
        row_idx = random.randint(0, n_rows - 1)
        row = rows[row_idx]
        if len(row) < k + 1:
            continue
        
        current_max = row[-1]
        if current_max <= 0:
            continue
        
        # Try to find a smaller value for the last element
        # Remove the last element's contributions
        old_last = row[-1]
        old_diffs = {old_last - a for a in row[:-1]}
        used -= old_diffs
        
        # Try values from current_max - 1 down to 1
        found = False
        for new_val in range(current_max - 1, 0, -1):
            if new_val <= row[-2]:  # Must be strictly increasing
                break
            new_diffs = {new_val - a for a in row[:-1]}
            if not new_diffs & used:
                row[-1] = new_val
                used |= new_diffs
                improved = True
                found = True
                break
        
        if not found:
            # Restore
            used |= old_diffs
            row[-1] = old_last
    
    return improved


def _counting_bound(n_rows, k):
    """
    Simple counting bound: total distinct differences needed is n_rows * k*(k+1)/2.
    The number of possible differences up to m is m(m+1)/2.
    So we need m(m+1)/2 >= n_rows * k*(k+1)/2
    => m(m+1) >= n_rows * k*(k+1)
    """
    target = n_rows * k * (k + 1)
    m = 1
    while m * (m + 1) < target:
        m += 1
    return m
# EVOLVE-BLOCK-END
