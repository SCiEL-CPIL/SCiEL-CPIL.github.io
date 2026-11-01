"""Difference triangle set (4,7) – CDVR local search with deadline guard."""
import time
import random
import sys

# We will use a simple local search with conflict-directed value removal.
# The idea:
# 1. Start from a greedy construction (same as seed) to get a valid baseline.
# 2. Try to improve by reducing the largest entry (the scope).
# 3. Use a time-bounded local search: repeatedly pick a row-position that causes
#    the most "pressure" (i.e., its value is large and it conflicts with the
#    goal of minimizing the max), then try to replace it with a smaller value
#    that keeps the row strictly increasing and all differences globally unique.

def _greedy_construct(n_rows, k):
    """Greedy ruler construction – returns list of rows and the used-diff set."""
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
    return rows, used


def _all_diffs(rows):
    """Return the set of all positive differences across all rows."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
    return diffs


def _is_valid(rows, k):
    """Check that each row has k+1 strictly increasing entries starting at 0,
    and all positive differences are globally distinct."""
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
    return len(_all_diffs(rows)) == n_diffs_total(k)


def n_diffs_total(k):
    """Total number of positive differences per row is k(k+1)/2, times n_rows? No –
    we need the count for a single row: k(k+1)/2. But validity is about global uniqueness,
    so the total number of differences across all rows must equal n_rows * k*(k+1)//2,
    and they must all be distinct."""
    # This helper is just for clarity; actual check uses set length.
    return None  # placeholder


def _check_valid(rows, k):
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i - 1]:
                return False
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
    return True


def _scope(rows):
    return max(max(row) for row in rows)


def construct(n_rows, k):
    start_time = time.time()
    DEADLINE = 140.0  # leave 10s margin

    # 1. Greedy baseline
    rows, used = _greedy_construct(n_rows, k)
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)

    if not _check_valid(best_rows, k):
        # Should not happen with greedy, but safety
        return best_rows

    # 2. Local search: try to reduce the scope by replacing large values with smaller ones
    #    that keep validity.
    
    # Precompute for each row the "slots" (positions 1..k, since position 0 is always 0)
    # We'll do a simple simulated-annealing-like or hill-climbing search.
    
    # Represent state as list of rows.
    # Move: pick a row r and a position p (1..k), try a new value v such that:
    #   row[r][p-1] < v < row[r][p+1] (if p < k) or v > row[r][p-1] if p == k
    #   and v < current scope (to try to reduce max)
    #   and all new diffs are not in the global used set.
    
    # To make this efficient, we maintain the global diff set incrementally.
    
    # Let's implement a simple random restart local search with a time budget.
    
    import copy
    
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    
    # We'll do multiple rounds of improvement.
    # For each row, try to "tighten" it by reducing its last element.
    
    # A simpler approach: for each row independently, try to find a shorter ruler
    # that uses only differences not in the other rows' diffs.
    
    # Actually, let's do a full local search:
    # State: rows (list of lists)
    # Objective: minimize max entry
    # Move: change one entry in one row to a smaller valid value.
    
    # For efficiency, maintain:
    #   global_diffs: set of all differences
    #   For a candidate change in row r, position p, from old_val to new_val:
    #     Remove old diffs involving position p, add new diffs.
    
    def _diffs_involving(row, p):
        """Return set of diffs that involve position p in row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            if i != p:
                if i < p:
                    diffs.add(row[p] - row[i])
                else:
                    diffs.add(row[i] - row[p])
        return diffs
    
    # Let's do a time-bounded search with random moves.
    
    rng = random.Random(42)
    
    # We'll try to improve by reducing the maximum.
    # Find which row has the max, and try to reduce its last element.
    
    # Simple hill-climbing with random restarts:
    
    def _try_improve(rows, max_time):
        """Try to reduce the scope of rows. Returns improved rows or original."""
        nonlocal best_rows, best_scope
        t0 = time.time()
        rows = [list(r) for r in rows]
        current_scope = _scope(rows)
        
        # Maintain global diffs
        global_diffs = _all_diffs(rows)
        
        improved = False
        
        while time.time() - t0 < max_time:
            # Find the row with the largest last element
            max_row = max(range(n_rows), key=lambda i: rows[i][-1])
            max_val = rows[max_row][-1]
            
            if max_val < best_scope:
                best_scope = max_val
                best_rows = [list(r) for r in rows]
                improved = True
            
            # Try to reduce the last element of max_row
            # We need to find a value v < max_val such that:
            #   rows[max_row][-2] < v
            #   and all diffs involving the new last element are not in global_diffs
            #     (after removing the old last element's diffs)
            
            old_diffs = _diffs_involving(rows[max_row], k)  # position k is the last
            # Remove old_diffs from global_diffs temporarily
            # We need to check: new_diffs = {v - a for a in rows[max_row][:k]}
            # These must not be in (global_diffs - old_diffs)
            
            base_diffs = global_diffs - old_diffs
            
            # Try values from max_val - 1 down to rows[max_row][k-1] + 1
            found = False
            for v in range(max_val - 1, rows[max_row][k - 1], -1):
                new_diffs = {v - a for a in rows[max_row][:k]}
                if not new_diffs & base_diffs:
                    # Valid move
                    rows[max_row][k] = v
                    global_diffs = base_diffs | new_diffs
                    found = True
                    break
            
            if not found:
                # Try a random move: pick a random row and position, try a random smaller value
                r = rng.randint(0, n_rows - 1)
                p = rng.randint(1, k)  # positions 1..k
                old_val = rows[r][p]
                lo = rows[r][p - 1] + 1
                hi = rows[r][p + 1] - 1 if p < k else old_val - 1
                if hi < lo:
                    continue
                # Try a few random values
                tried = 0
                while tried < 50 and time.time() - t0 < max_time:
                    v = rng.randint(lo, hi)
                    if v == old_val:
                        tried += 1
                        continue
                    old_diffs = _diffs_involving(rows[r], p)
                    base_diffs = global_diffs - old_diffs
                    new_diffs = set()
                    n = len(rows[r])
                    for i in range(n):
                        if i != p:
                            if i < p:
                                new_diffs.add(v - rows[r][i])
                            else:
                                new_diffs.add(rows[r][i] - v)
                    if not new_diffs & base_diffs:
                        rows[r][p] = v
                        global_diffs = base_diffs | new_diffs
                        found = True
                        break
                    tried += 1
            
            if not found:
                # Stuck; break and restart
                break
        
        return rows
    
    # Do several improvement rounds
    num_rounds = 5
    time_per_round = 20.0
    for rnd in range(num_rounds):
        if time.time() > DEADLINE:
            break
        rows = _try_improve(best_rows, time_per_round)
        if _scope(rows) < best_scope:
            best_rows = [list(r) for r in rows]
            best_scope = _scope(best_rows)
    
    # Final check
    if _check_valid(best_rows, k):
        return best_rows
    
    # Fallback to original greedy
    return [list(r) for r in rows]
