"""Difference triangle set (4,7): minimize largest entry."""
import time
import random
from itertools import combinations

# Deadline guard
_DEADLINE = 150.0  # seconds wall-clock limit for evaluator
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START) - 5.0  # 5s safety margin


def _row_diffs(row):
    """Return set of all positive differences in a row."""
    n = len(row)
    return {row[j] - row[i] for i in range(n) for j in range(i + 1, n)}


def _all_diffs(rows):
    """Return set of all positive differences across all rows."""
    s = set()
    for r in rows:
        s |= _row_diffs(r)
    return s


def _is_valid(rows):
    """Check that all positive differences are distinct across all rows."""
    seen = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - r[i]
                if d in seen:
                    return False
                seen.add(d)
    return True


def _score(rows):
    """The largest entry across all rows (lower is better)."""
    return max(max(r) for r in rows)


def _greedy_row(used, k, start=1):
    """Build one greedy row of k+1 entries starting at 0, given used diffs."""
    row = [0]
    x = start
    while len(row) < k + 1:
        new = {x - a for a in row}
        if not new & used:
            row.append(x)
            used |= new
        x += 1
    return row


def _greedy_construct(n_rows, k):
    """Greedy construction: each row takes smallest marks avoiding used diffs."""
    used = set()
    rows = []
    for _ in range(n_rows):
        row = _greedy_row(used, k)
        rows.append(row)
    return rows


def _conflict_count(rows):
    """Count number of duplicate differences (0 means valid)."""
    seen = set()
    conflicts = 0
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[j] - r[i]
                if d in seen:
                    conflicts += 1
                seen.add(d)
    return conflicts


def _local_search(rows, n_rows, k, max_iters=5000):
    """Local search: try to reduce the largest entry by moving individual marks."""
    best = [list(r) for r in rows]
    best_score = _score(best)
    best_conflicts = _conflict_count(best)
    
    if best_conflicts == 0:
        return best
    
    # We work on a copy; try to reduce max entry
    # Strategy: for the row with the largest entry, try to pull its last element down
    # by adjusting intermediate marks.
    
    for iteration in range(max_iters):
        if _time_left() < 2.0:
            break
        
        # Find the row with the largest max entry
        max_row_idx = max(range(n_rows), key=lambda i: best[i][-1])
        row = best[max_row_idx]
        
        # Try moving each non-first, non-last element to a smaller value
        improved = False
        for pos in range(1, k):  # skip first (0) and last
            old_val = row[pos]
            # Try decreasing
            for new_val in range(old_val - 1, 0, -1):
                if new_val <= row[pos - 1] or (pos < k and new_val >= row[pos + 1]):
                    continue
                # Check if this creates conflicts within this row
                row[pos] = new_val
                if _conflict_count(best) < best_conflicts or (
                    _conflict_count(best) == best_conflicts and _score(best) < best_score
                ):
                    best_conflicts = _conflict_count(best)
                    best_score = _score(best)
                    improved = True
                    break
                row[pos] = old_val
            if improved:
                break
        
        if not improved:
            # Try increasing some marks to free up space, then decrease others
            # Or try a random perturbation
            break
    
    return best


def _sa_optimize(rows, n_rows, k, time_limit=10.0):
    """Simulated annealing to reduce the largest entry."""
    best = [list(r) for r in rows]
    best_score = _score(best)
    best_conflicts = _conflict_count(best)
    
    if best_conflicts == 0:
        return best
    
    current = [list(r) for r in best]
    current_score = _score(current)
    current_conflicts = _conflict_count(current)
    
    start_temp = 50.0
    end_temp = 0.1
    start_time = time.time()
    
    while time.time() - start_time < time_limit and _time_left() > 3.0:
        temp = start_temp * (end_temp / start_temp) ** ((time.time() - start_time) / time_limit)
        
        # Pick a random row and a random position (not 0)
        row_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        old_val = current[row_idx][pos]
        # Try a random new value
        delta = random.randint(-10, 10)
        new_val = old_val + delta
        
        # Bounds: must stay increasing
        if new_val <= current[row_idx][pos - 1]:
            new_val = current[row_idx][pos - 1] + 1
        if pos < k and new_val >= current[row_idx][pos + 1]:
            new_val = current[row_idx][pos + 1] - 1
        if new_val <= current[row_idx][pos - 1] or (pos < k and new_val >= current[row_idx][pos + 1]):
            continue
        
        current[row_idx][pos] = new_val
        new_score = _score(current)
        new_conflicts = _conflict_count(current)
        
        # Accept if better or with probability exp(-delta_energy/temp)
        delta_energy = (new_conflicts - current_conflicts) + 0.1 * (new_score - current_score)
        
        if new_conflicts == 0 and new_score < best_score:
            best = [list(r) for r in current]
            best_score = new_score
            best_conflicts = 0
            current_conflicts = new_conflicts
            current_score = new_score
        elif delta_energy <= 0 or random.random() < __import__('math').exp(-delta_energy / max(temp, 0.01)):
            current_conflicts = new_conflicts
            current_score = new_score
        else:
            current[row_idx][pos] = old_val
    
    if best_conflicts == 0:
        return best
    return current if _score(current) < best_score else best


def construct(n_rows, k):
    """Construct n_rows rows of k+1 increasing integers starting at 0,
    minimizing the largest entry subject to all positive differences being distinct."""
    global _START
    _START = time.time()
    
    # Phase 1: Greedy construction
    rows = _greedy_construct(n_rows, k)
    if _is_valid(rows):
        best = rows
        best_score = _score(rows)
    else:
        best = rows
        best_score = _score(rows)
    
    # Phase 2: Local search to fix conflicts and reduce max
    if _time_left() > 5.0:
        rows_ls = _local_search(rows, n_rows, k, max_iters=3000)
        if _is_valid(rows_ls) and _score(rows_ls) < best_score:
            best = rows_ls
            best_score = _score(rows_ls)
        elif _is_valid(rows_ls):
            best = rows_ls
            best_score = _score(rows_ls)
    
    # Phase 3: Simulated annealing for further optimization
    if _time_left() > 8.0:
        sa_time = min(_time_left() - 6.0, 20.0)
        if sa_time > 2.0:
            rows_sa = _sa_optimize(best, n_rows, k, time_limit=sa_time)
            if _is_valid(rows_sa) and _score(rows_sa) < best_score:
                best = rows_sa
                best_score = _score(rows_sa)
    
    # Phase 4: Final local polish
    if _time_left() > 3.0:
        rows_final = _local_search(best, n_rows, k, max_iters=2000)
        if _is_valid(rows_final) and _score(rows_final) < best_score:
            best = rows_final
    
    # Ensure validity: if not valid, fall back to greedy
    if not _is_valid(best):
        best = _greedy_construct(n_rows, k)
    
    return best
