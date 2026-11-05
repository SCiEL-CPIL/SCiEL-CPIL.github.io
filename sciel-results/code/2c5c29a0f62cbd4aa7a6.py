"""
Difference triangle set (4,7) solver.

Strategy:
1. Fast greedy construction to get an initial valid solution.
2. Local search with rebuild-based moves to escape local minima.
   - Randomly select a row and rebuild it using the current global used-differences set
     (excluding that row's own differences).
   - Track the best (minimum max-entry) solution found.
   - Use a deadline guard to return the best solution before timeout.

Key insight from measured knowledge: rebuild-based local search with random reshuffling
of rows allows broader exploration than simple hill-climbing, escaping local minima.
"""
import time
import random
import sys

def _greedy_row(used, k, start=0):
    """Build a single row greedily given the set of already-used differences."""
    row = [0]
    x = start
    while len(row) < k + 1:
        x += 1
        new_diffs = set(x - a for a in row)
        if not new_diffs & used:
            row.append(x)
            used |= new_diffs
    return row, used

def _compute_used(rows):
    """Compute the set of all positive differences across all rows."""
    used = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                used.add(row[j] - row[i])
    return used

def _max_entry(rows):
    return max(max(row) for row in rows)

def _is_valid(rows):
    """Check that all positive differences are distinct across all rows."""
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

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry, subject to all positive differences
    being distinct across all rows.
    """
    deadline = time.time() + 140  # 140s safety margin before 150s kill
    
    # Phase 1: Initial greedy construction
    used = set()
    rows = []
    for _ in range(n_rows):
        row, used = _greedy_row(used, k)
        rows.append(row)
    
    best_rows = [list(r) for r in rows]
    best_score = _max_entry(best_rows)
    
    # Phase 2: Local search with rebuild moves
    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Select a random row to rebuild
        idx = rng.randrange(n_rows)
        
        # Compute used set excluding row idx
        used_excl = set()
        for i, row in enumerate(rows):
            if i == idx:
                continue
            n = len(row)
            for a in range(n):
                for b in range(a + 1, n):
                    used_excl.add(row[b] - row[a])
        
        # Rebuild row idx greedily with some randomization
        # Try multiple random starting offsets and pick the best
        best_new_row = None
        best_new_max = float('inf')
        
        # Number of attempts per iteration (keep small for speed)
        n_attempts = 3
        for attempt in range(n_attempts):
            temp_used = set(used_excl)
            row = [0]
            x = 0
            success = True
            while len(row) < k + 1:
                # Try next few candidates
                found = False
                for delta in range(1, 8):
                    candidate = x + delta
                    new_diffs = set(candidate - a for a in row)
                    if not new_diffs & temp_used:
                        row.append(candidate)
                        temp_used |= new_diffs
                        x = candidate
                        found = True
                        break
                if not found:
                    success = False
                    break
            if success:
                m = max(row)
                if m < best_new_max:
                    best_new_max = m
                    best_new_row = list(row)
        
        if best_new_row is None:
            no_improve_count += 1
            continue
        
        # Evaluate: replace row idx and check validity + score
        old_row = rows[idx]
        rows[idx] = best_new_row
        
        # Quick check: is the new solution valid?
        # Recompute full used set
        full_used = set()
        valid = True
        for i, row in enumerate(rows):
            n = len(row)
            for a in range(n):
                for b in range(a + 1, n):
                    d = row[b] - row[a]
                    if d in full_used:
                        valid = False
                        break
                    full_used.add(d)
            if not valid:
                break
        
        if valid:
            new_score = _max_entry(rows)
            if new_score < best_score:
                best_score = new_score
                best_rows = [list(r) for r in rows]
                no_improve_count = 0
            else:
                # Accept with some probability (simulated annealing-like)
                # to escape local minima
                if rng.random() < 0.15:
                    no_improve_count += 1
                else:
                    rows[idx] = old_row
                    no_improve_count += 1
        else:
            rows[idx] = old_row
            no_improve_count += 1
    
    return best_rows
