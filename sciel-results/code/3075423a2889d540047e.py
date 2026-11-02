"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 120
    rows = _greedy_construct(n_rows, k)
    rows = _local_search(rows, n_rows, k, deadline)
    return rows

def _greedy_construct(n_rows, k):
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

def _get_used_diffs(rows):
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                used.add(row[j] - row[i])
    return used

def _is_valid(rows):
    used = set()
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    return False
                used.add(d)
    return True

def _scope(rows):
    return max(max(row) for row in rows)

def _local_search(rows, n_rows, k, deadline):
    best_rows = [list(r) for r in rows]
    best_scope = _scope(best_rows)
    rng = random.Random(42)
    
    while time.time() < deadline - 5:
        # Try to improve by moving one element in one row
        improved = False
        for _ in range(200):
            if time.time() >= deadline - 5:
                break
            ri = rng.randrange(n_rows)
            row = best_rows[ri]
            # Choose which position to try moving (not position 0 which is fixed at 0)
            if len(row) <= 1:
                continue
            pos = rng.randrange(1, len(row))
            old_val = row[pos]
            # Try shifting this element
            for delta in [-1, 1, -2, 2, -3, 3]:
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                # Check if new value maintains strict increase
                if pos > 0 and new_val <= row[pos - 1]:
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                # Tentatively apply
                row[pos] = new_val
                if _is_valid(best_rows):
                    new_scope = _scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        improved = True
                    else:
                        row[pos] = old_val
                else:
                    row[pos] = old_val
                if improved:
                    break
            if improved:
                break
        if not improved:
            # Try a different strategy: rebuild one row greedily with a bias
            if time.time() >= deadline - 5:
                break
            ri = rng.randrange(n_rows)
            other_rows = [list(r) for j, r in enumerate(best_rows) if j != ri]
            used = _get_used_diffs(other_rows)
            new_row = [0]
            x = 0
            while len(new_row) < k + 1 and time.time() < deadline - 5:
                x += 1
                new = {x - a for a in new_row}
                if not new & used:
                    new_row.append(x)
                    used |= new
            if len(new_row) == k + 1:
                candidate = [list(r) for r in best_rows]
                candidate[ri] = new_row
                new_scope = _scope(candidate)
                if new_scope <= best_scope:
                    best_rows = candidate
                    best_scope = new_scope
    
    return best_rows
# EVOLVE-BLOCK-END
