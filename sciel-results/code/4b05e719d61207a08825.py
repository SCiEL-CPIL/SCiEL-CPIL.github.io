"""Difference triangle set (4,7) — backtracking with MRV and local search."""
import time
import random
import sys
from collections import defaultdict

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
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


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


def _backtrack_construct(n_rows, k, deadline, rng, max_scope=None):
    """
    Backtracking search with MRV heuristic.
    """
    used_diffs = set()
    rows = []
    
    def solve(idx):
        if time.time() > deadline:
            return None
        if idx == n_rows:
            return rows[:]
        
        best_row = None
        best_used = None
        best_max = float('inf')
        
        for attempt in range(80):
            trial_row = [0]
            trial_used = set(used_diffs)
            x = 0
            success = True
            while len(trial_row) < k + 1:
                x += 1
                if max_scope is not None and x > max_scope:
                    success = False
                    break
                new = {x - a for a in trial_row}
                if new & trial_used:
                    continue
                if attempt == 0:
                    trial_row.append(x)
                    trial_used |= new
                else:
                    if rng.random() < 0.8:
                        trial_row.append(x)
                        trial_used |= new
            
            if success and len(trial_row) == k + 1:
                if trial_row[-1] < best_max:
                    best_row = trial_row
                    best_used = trial_used
                    best_max = trial_row[-1]
                    if best_max < 140:
                        break
        
        if best_row is None:
            return None
        
        rows.append(best_row)
        result = solve(idx + 1)
        if result is not None:
            return result
        rows.pop()
        return None
    
    return solve(0)


def _local_search(rows, deadline, rng):
    """
    Local search: try to reduce the maximum entry.
    """
    best = rows[:]
    best_score = _max_entry(best)
    current_score = best_score
    
    while time.time() < deadline:
        row_idx = rng.randrange(len(best))
        old_row = best[row_idx][:]
        
        used_others = set()
        for i, r in enumerate(best):
            if i != row_idx:
                used_others |= _all_diffs(r)
        
        new_row = [0]
        x = 0
        while len(new_row) < len(old_row):
            x += 1
            new = {x - a for a in new_row}
            if not new & used_others:
                if rng.random() < 0.7:
                    new_row.append(x)
                else:
                    if rng.random() < 0.5:
                        new_row.append(x)
        
        if len(new_row) == len(old_row):
            trial_rows = best[:]
            trial_rows[row_idx] = new_row
            if _is_valid(trial_rows):
                sc = _max_entry(trial_rows)
                if sc < current_score:
                    current_score = sc
                    best = trial_rows
                    if sc < best_score:
                        best_score = sc
                elif rng.random() < 0.1:
                    best = trial_rows
                    current_score = sc
        
        if rng.random() < 0.2:
            used_others = set()
            for i, r in enumerate(best):
                if i != row_idx:
                    used_others |= _all_diffs(r)
            
            new_row = [0]
            x = 0
            while len(new_row) < len(old_row):
                x += 1
                new = {x - a for a in new_row}
                if not new & used_others:
                    new_row.append(x)
            
            if len(new_row) == len(old_row):
                trial_rows = best[:]
                trial_rows[row_idx] = new_row
                if _is_valid(trial_rows):
                    sc = _max_entry(trial_rows)
                    if sc < current_score:
                        current_score = sc
                        best = trial_rows
                        if sc < best_score:
                            best_score = sc
    
    return best


def _spectral_guided_construct(n_rows, k, deadline, rng):
    """
    Spectral relaxation approach: build a conflict matrix for differences,
    use eigenvector separation to guide row construction.
    """
    # For k=7, each row has 28 differences. Total differences needed: n_rows * 28
    # We want to assign differences to rows such that they don't conflict.
    # 
    # Strategy: 
    # 1. Start with greedy to get a baseline
    # 2. Use the structure of differences to guide which rows to build next
    # 3. Local search to compact
    
    used = set()
    rows = []
    
    for row_idx in range(n_rows):
        # Build row with spectral guidance: prefer differences that are "far" 
        # from already-used differences in some sense
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                # Accept with high probability if this extends the row
                row.append(x)
                used |= new
        
        rows.append(row)
    
    return rows


def construct(n_rows, k):
    deadline = time.time() + 110.0
    rng = random.Random(42)
    
    # Start with greedy baseline
    best = _greedy_build(n_rows, k, random.Random(0), bias=1.0)
    best_score = _max_entry(best)
    
    # Try multiple greedy variants
    for seed in range(1, 25):
        if time.time() > deadline:
            break
        bias = rng.choice([1.0, 0.95, 0.9, 0.85, 0.8])
        trial = _greedy_build(n_rows, k, random.Random(seed), bias=bias)
        sc = _max_entry(trial)
        if sc < best_score:
            best_score = sc
            best = trial
    
    # Try backtracking with tighter bounds
    if time.time() < deadline - 30:
        bt_result = _backtrack_construct(n_rows, k, deadline - 15, rng, max_scope=best_score)
        if bt_result is not None:
            bt_score = _max_entry(bt_result)
            if bt_score < best_score:
                best_score = bt_score
                best = bt_result
    
    # Local search to improve
    best = _local_search(best, min(deadline - 5, time.time() + 40), rng)
    best_score = _max_entry(best)
    
    # If time remains, try more restarts
    while time.time() < deadline - 10:
        trial = _greedy_build(n_rows, k, rng, bias=rng.choice([1.0, 0.9, 0.8, 0.7, 0.6]))
        sc = _max_entry(trial)
        if sc < best_score:
            best_score = sc
            best = trial
            if time.time() < deadline - 15:
                best = _local_search(best, min(deadline - 5, time.time() + 20), rng)
                best_score = _max_entry(best)
        
        # Also try backtracking with tighter bound
        if time.time() < deadline - 25 and best_score > 130:
            bt_result = _backtrack_construct(n_rows, k, deadline - 10, rng, max_scope=best_score)
            if bt_result is not None:
                bt_score = _max_entry(bt_result)
                if bt_score < best_score:
                    best_score = bt_score
                    best = bt_result
                    if time.time() < deadline - 10:
                        best = _local_search(best, min(deadline - 5, time.time() + 10), rng)
                        best_score = _max_entry(best)
    
    return best

# EVOLVE-BLOCK-END
