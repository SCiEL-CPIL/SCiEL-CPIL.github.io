import time
import random
import sys

def construct(n_rows, k):
    """Construct difference triangle set with n_rows rows of k+1 entries each."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    best_rows = None
    best_max = float('inf')
    
    # First, get a baseline greedy solution
    baseline = _greedy(n_rows, k)
    best_rows = baseline
    best_max = max(max(r) for r in baseline)
    
    # Try randomized greedy with different orderings
    for attempt in range(200):
        if time.time() > deadline:
            break
        rows = _randomized_greedy(n_rows, k, seed=attempt)
        m = max(max(r) for r in rows)
        if m < best_max:
            best_max = m
            best_rows = rows
    
    # Local search: try to reduce max by re-optimizing individual rows
    if time.time() < deadline - 10:
        best_rows, best_max = _local_search(best_rows, n_rows, k, best_max, deadline)
    
    return best_rows


def _greedy(n_rows, k):
    """Standard greedy construction."""
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows


def _randomized_greedy(n_rows, k, seed=0):
    """Randomized greedy with some exploration."""
    rng = random.Random(seed)
    used, rows = set(), []
    for _ in range(n_rows):
        row, x = [0], 0
        while len(row) < k + 1:
            # Try several candidates, pick randomly among valid ones near the minimum
            candidates = []
            search_limit = x + 5 + rng.randint(0, 5)
            for trial in range(x + 1, search_limit):
                new = {trial - a for a in row}
                if not new & used:
                    candidates.append(trial)
                    if len(candidates) >= 3:
                        break
            if candidates:
                x = candidates[rng.randint(0, len(candidates) - 1)]
            else:
                # Fallback: keep searching
                while True:
                    x += 1
                    new = {x - a for a in row}
                    if not new & used:
                        break
            row.append(x)
            used |= {x - a for a in row[:-1]}
        rows.append(row)
    return rows


def _local_search(rows, n_rows, k, best_max, deadline):
    """Try to improve by re-optimizing rows in different orders."""
    current = [list(r) for r in rows]
    current_max = max(max(r) for r in current)
    
    # Try different row orderings
    orderings = []
    rng = random.Random(42)
    for _ in range(50):
        order = list(range(n_rows))
        rng.shuffle(order)
        orderings.append(order)
    orderings.append(list(range(n_rows)))  # original order
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        # Rebuild using this order with greedy
        used = set()
        new_rows = [None] * n_rows
        success = True
        for idx in order:
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            new_rows[idx] = row
        
        m = max(max(r) for r in new_rows)
        if m < current_max:
            current = new_rows
            current_max = m
    
    return current, current_max
