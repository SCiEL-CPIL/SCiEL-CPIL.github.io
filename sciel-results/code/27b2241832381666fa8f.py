"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 110  # safety margin before 120s limit
    best_rows = None
    best_scope = float('inf')
    
    # First try deterministic greedy
    rows = _greedy_construct(n_rows, k)
    scope = max(max(r) for r in rows)
    best_rows = rows
    best_scope = scope
    
    # Try randomized restarts to find better solutions
    for restart in range(200):
        if time.time() > deadline:
            break
        random.seed(restart * 7919 + 13)
        rows = _randomized_construct(n_rows, k, deadline)
        if rows is None:
            continue
        if not _is_valid(rows, k):
            continue
        scope = max(max(r) for r in rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
            # If we hit the theoretical lower bound, stop
            if best_scope <= 105:
                break
    
    return best_rows

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

def _randomized_construct(n_rows, k, deadline):
    used, rows = set(), []
    for row_idx in range(n_rows):
        if time.time() > deadline:
            return None
        row, x = [0], 0
        while len(row) < k + 1:
            if time.time() > deadline:
                return None
            # Try candidates in random order with bias toward smaller values
            candidates = list(range(x + 1, x + 50))
            random.shuffle(candidates)
            # Sort by value but keep some randomness
            candidates.sort(key=lambda c: c + random.random() * 10)
            found = False
            for cand in candidates:
                new = {cand - a for a in row}
                if not new & used:
                    row.append(cand)
                    used |= new
                    x = cand
                    found = True
                    break
            if not found:
                # Extend search range
                for cand in range(x + 50, x + 200):
                    new = {cand - a for a in row}
                    if not new & used:
                        row.append(cand)
                        used |= new
                        x = cand
                        found = True
                        break
            if not found:
                return None
        rows.append(row)
    return rows

def _is_valid(rows, k):
    used = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diff = row[j] - row[i]
                if diff <= 0:
                    return False
                if diff in used:
                    return False
                used.add(diff)
    return True
# EVOLVE-BLOCK-END
