"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _diffs_of_row(row):
    """Return set of positive differences for a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _total_scope(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)


def _all_diffs_used(rows):
    """Return the set of all differences used across all rows."""
    used = set()
    for r in rows:
        used |= _diffs_of_row(r)
    return used


def _greedy_row(used, k, start_val=1):
    """
    Greedily construct a row of k+1 integers starting at 0.
    Each new element is the smallest integer > last that doesn't create
    a difference already in 'used'.
    Returns (row, new_diffs) or None if stuck.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used) and not (new & new_diffs):
            row.append(x)
            new_diffs |= new
    return row, new_diffs


def _greedy_construct(n_rows, k):
    """Pure greedy construction."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row, new_diffs = _greedy_row(used, k)
        if row is None:
            return None
        rows.append(row)
        used |= new_diffs
    return rows


def _randomized_greedy_row(used, k, rng, pool_size=5):
    """
    Randomized greedy: at each step, pick from the top 'pool_size' smallest
    valid candidates with some randomness.
    """
    row = [0]
    new_diffs = set()
    x = 0
    while len(row) < k + 1:
        # Collect valid candidates
        candidates = []
        x += 1
        while len(candidates) < pool_size:
            new = {x - a for a in row}
            if not (new & used) and not (new & new_diffs):
                candidates.append(x)
            x += 1
        if not candidates:
            return None, None
        # Pick randomly from candidates, weighted toward smaller
        # Use a simple approach: pick the first candidate with prob 0.6,
        # otherwise pick uniformly from all candidates
        if rng.random() < 0.6:
            choice = candidates[0]
        else:
            choice = rng.choice(candidates)
        # Add all differences up to choice
        row.append(choice)
        for a in row[:-1]:
            new_diffs.add(choice - a)
        x = choice
    return row, new_diffs


def _randomized_greedy(n_rows, k, rng, pool_size=5):
    """Randomized greedy construction with multiple restarts."""
    best_rows = None
    best_scope = float('inf')
    
    max_restarts = 50
    deadline = time.time() + 100  # Leave margin for safety
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row, new_diffs = _randomized_greedy_row(used, k, rng, pool_size)
            if row is None:
                success = False
                break
            rows.append(row)
            used |= new_diffs
        if success:
            scope = _total_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
    
    return best_rows if best_rows is not None else _greedy_construct(n_rows, k)


def _local_search(rows, k, rng, max_iterations=5000):
    """
    Local search: try to reduce the scope by moving elements.
    We try to shrink the largest element in each row.
    """
    rows = [r[:] for r in rows]
    used = _all_diffs_used(rows)
    scope = _total_scope(rows)
    
    for _ in range(max_iterations):
        # Find the row with the largest last element
        max_idx = max(range(len(rows)), key=lambda i: rows[i][-1])
        row = rows[max_idx]
        if len(row) < 2:
            break
        
        # Try to reduce the last element
        last = row[-1]
        found_better = False
        for new_last in range(last - 1, 0, -1):
            # Check if new_last > row[-2]
            if new_last <= row[-2]:
                break
            # Remove old differences involving last
            old_diffs = {last - a for a in row[:-1]}
            # Check if new differences conflict
            new_diffs = {new_last - a for a in row[:-1]}
            # Temporarily remove old_diffs from used, check new_diffs
            remaining_used = used - old_diffs
            if not (new_diffs & remaining_used):
                # Valid move
                row[-1] = new_last
                used = (used - old_diffs) | new_diffs
                scope = _total_scope(rows)
                found_better = True
                break
        if not found_better:
            # Try moving other elements in the row
            for pos in range(len(row) - 2, 0, -1):
                old_val = row[pos]
                found = False
                for new_val in range(old_val - 1, row[pos - 1], -1):
                    if new_val <= row[pos - 1]:
                        break
                    # Compute differences involving old_val
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    remaining_used = used - old_diffs
                    if not (new_diffs & remaining_used):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                        found = True
                        break
                if found:
                    break
            if not found:
                # Try a random perturbation to escape local optimum
                if rng.random() < 0.1:
                    # Randomly increase an element to allow future decreases
                    pos = rng.randint(1, len(row) - 1)
                    old_val = row[pos]
                    old_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            old_diffs.add(old_val - row[i])
                        else:
                            old_diffs.add(row[i] - old_val)
                    new_val = old_val + rng.randint(1, 5)
                    new_diffs = set()
                    for i in range(len(row)):
                        if i == pos:
                            continue
                        if i < pos:
                            new_diffs.add(new_val - row[i])
                        else:
                            new_diffs.add(row[i] - new_val)
                    remaining_used = used - old_diffs
                    if not (new_diffs & remaining_used):
                        row[pos] = new_val
                        used = (used - old_diffs) | new_diffs
                else:
                    break
    
    return rows


def construct(n_rows, k):
    """
    Construct n_rows difference triangles of k+1 integers each.
    Minimizes the scope (largest entry).
    """
    rng = random.Random(42)
    
    # Start with greedy
    rows = _greedy_construct(n_rows, k)
    if rows is None:
        # Fallback: very simple construction
        rows = []
        used = set()
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
    
    # Try randomized greedy with multiple restarts
    deadline = time.time() + 110  # Leave 10s margin
    best_rows = rows
    best_scope = _total_scope(rows)
    
    # Phase 1: Randomized greedy restarts
    for restart in range(30):
        if time.time() > deadline:
            break
        rand_rows = _randomized_greedy(n_rows, k, rng, pool_size=rng.choice([3, 5, 8]))
        if rand_rows is not None:
            scope = _total_scope(rand_rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rand_rows]
    
    # Phase 2: Local search on best solution
    if time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline:
                break
            ls_rows = _local_search(best_rows, k, rng, max_iterations=2000)
            scope = _total_scope(ls_rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in ls_rows]
    
    return best_rows

# EVOLVE-BLOCK-END
