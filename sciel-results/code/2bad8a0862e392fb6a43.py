"""Difference triangle set (4,7) — simulated annealing with multi-scale moves."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def _all_diffs(row):
    """Return set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _max_entry(rows):
    return max(max(r) for r in rows)


def _is_valid(rows):
    """Check if all differences across all rows are distinct."""
    seen = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _count_conflicts(rows):
    """Count number of duplicate differences (conflicts)."""
    seen = set()
    conflicts = 0
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in seen:
                    conflicts += 1
                else:
                    seen.add(d)
    return conflicts


def _diff_set_for_row(row, used_others):
    """Compute the set of differences in a row, checking for conflicts with used_others."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in used_others:
                return None, -1
            if d in diffs:
                return None, -1
            diffs.add(d)
    return diffs, 0


def _greedy_build(n_rows, k, rng, bias=1.0):
    """Greedy build with optional randomness."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                if bias >= 1.0 or rng.random() < bias:
                    row.append(x)
                    used |= new
                else:
                    if rng.random() > 0.3:
                        row.append(x)
                        used |= new
        rows.append(row)
    return rows


def _rebuild_row(row_idx, rows, k, used_others, rng, greedy=True):
    """Rebuild a single row given the differences used by other rows."""
    if greedy:
        new_row = [0]
        x = 0
        while len(new_row) < k + 1:
            x += 1
            new = {x - a for a in new_row}
            if not new & used_others:
                new_row.append(x)
        return new_row if len(new_row) == k + 1 else None
    else:
        # Randomized rebuild
        new_row = [0]
        x = 0
        attempts = 0
        while len(new_row) < k + 1 and attempts < 5000:
            x += 1
            attempts += 1
            new = {x - a for a in new_row}
            if not new & used_others:
                if rng.random() < 0.8:
                    new_row.append(x)
                else:
                    if rng.random() < 0.5:
                        new_row.append(x)
        return new_row if len(new_row) == k + 1 else None


def _local_search(rows, deadline, rng):
    """
    Local search with simulated annealing: try to reduce the maximum entry
    by adjusting individual row elements while maintaining validity.
    """
    best = [r[:] for r in rows]
    best_score = _max_entry(best)
    current_score = best_score
    
    # Precompute used diffs for each row
    def compute_used_others(rows, row_idx):
        used = set()
        for i, r in enumerate(rows):
            if i != row_idx:
                used |= _all_diffs(r)
        return used
    
    # Temperature schedule
    t0 = time.time()
    total_time = deadline - t0
    T_start = 10.0
    T_end = 0.01
    
    while time.time() < deadline:
        elapsed = time.time() - t0
        T = T_start * (T_end / T_start) ** (elapsed / total_time)
        
        # Pick a random row to modify
        row_idx = rng.randrange(len(best))
        used_others = compute_used_others(best, row_idx)
        
        # Try to rebuild this row
        new_row = _rebuild_row(row_idx, best, len(best[row_idx]) - 1, used_others, rng, greedy=rng.random() < 0.5)
        
        if new_row is not None and len(new_row) == len(best[row_idx]):
            trial_rows = [r[:] for r in best]
            trial_rows[row_idx] = new_row
            
            # Check validity
            if _is_valid(trial_rows):
                sc = _max_entry(trial_rows)
                if sc < current_score:
                    current_score = sc
                    best = trial_rows
                    if sc < best_score:
                        best_score = sc
                else:
                    # Accept with probability based on temperature
                    delta = sc - current_score
                    if rng.random() < min(1.0, (-delta) / T):
                        best = trial_rows
                        current_score = sc
    
    return best


def _element_move_search(rows, deadline, rng):
    """
    Fine-grained search: move individual elements within a row to reduce scope.
    """
    best = [r[:] for r in rows]
    best_score = _max_entry(best)
    current_score = best_score
    
    def compute_used_others(rows, row_idx):
        used = set()
        for i, r in enumerate(rows):
            if i != row_idx:
                used |= _all_diffs(r)
        return used
    
    t0 = time.time()
    total_time = deadline - t0
    T_start = 5.0
    T_end = 0.01
    
    while time.time() < deadline:
        elapsed = time.time() - t0
        T = T_start * (T_end / T_start) ** (elapsed / total_time)
        
        # Pick a random row
        row_idx = rng.randrange(len(best))
        row = best[row_idx]
        used_others = compute_used_others(best, row_idx)
        
        # Try to move a random element
        if len(row) <= 1:
            continue
        
        pos = rng.randrange(1, len(row))  # Don't move the 0
        old_val = row[pos]
        
        # Try shifting by a small amount
        shift = rng.choice([-3, -2, -1, 1, 2, 3])
        new_val = old_val + shift
        
        if new_val <= 0 or new_val >= row[pos + 1] if pos + 1 < len(row) else new_val > old_val:
            continue
        
        # Check if the new value creates valid increasing sequence
        if pos > 0 and new_val <= row[pos - 1]:
            continue
        if pos + 1 < len(row) and new_val >= row[pos + 1]:
            continue
        
        trial_row = row[:]
        trial_row[pos] = new_val
        
        # Check validity of the modified row against used_others
        diffs, conflicts = _diff_set_for_row(trial_row, used_others)
        if diffs is None:
            continue
        
        # Check full validity
        trial_rows = [r[:] for r in best]
        trial_rows[row_idx] = trial_row
        if _is_valid(trial_rows):
            sc = _max_entry(trial_rows)
            if sc < current_score:
                current_score = sc
                best = trial_rows
                if sc < best_score:
                    best_score = sc
            else:
                delta = sc - current_score
                if rng.random() < min(1.0, (-delta) / T):
                    best = trial_rows
                    current_score = sc
    
    return best


def construct(n_rows, k):
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    rng = random.Random(42)
    
    # Start with a good greedy baseline
    best = _greedy_build(n_rows, k, random.Random(0), bias=1.0)
    best_score = _max_entry(best)
    
    # Try multiple greedy variants
    for seed in range(1, 30):
        if time.time() > deadline:
            break
        trial = _greedy_build(n_rows, k, random.Random(seed), bias=rng.choice([1.0, 0.9, 0.8, 0.7]))
        sc = _max_entry(trial)
        if sc < best_score:
            best_score = sc
            best = trial
    
    # Local search to improve (row-level rebuilds)
    if time.time() < deadline - 20:
        best = _local_search(best, min(deadline - 10, time.time() + 40), rng)
        best_score = _max_entry(best)
    
    # Element move search (fine-grained)
    if time.time() < deadline - 10:
        best = _element_move_search(best, min(deadline - 5, time.time() + 30), rng)
        best_score = _max_entry(best)
    
    # If time remains, try more restarts with local search
    while time.time() < deadline - 10:
        trial = _greedy_build(n_rows, k, rng, bias=rng.choice([1.0, 0.9, 0.8, 0.7, 0.6]))
        sc = _max_entry(trial)
        if sc < best_score:
            best_score = sc
            best = trial
            if time.time() < deadline - 15:
                best = _local_search(best, min(deadline - 10, time.time() + 15), rng)
                best_score = _max_entry(best)
                if time.time() < deadline - 5:
                    best = _element_move_search(best, min(deadline - 3, time.time() + 10), rng)
                    best_score = _max_entry(best)
    
    return best

# EVOLVE-BLOCK-END
