"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _greedy_set(n_rows, k, seed=0):
    """Greedy construction."""
    rng = random.Random(seed)
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def _score(rows):
    """Return the largest entry across rows (the scope)."""
    return max(r[-1] for r in rows)

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0, minimizing the
    largest entry, subject to all positive differences within rows being distinct
    across all rows.
    """
    deadline = time.time() + 110  # safety margin before 120s limit
    best_rows = None
    best_scope = float('inf')

    # 1. Get initial upper bound via Greedy
    for seed in range(20):
        if time.time() > deadline:
            break
        rows = _greedy_set(n_rows, k, seed)
        scope = _score(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
            
    # 2. Try to improve using Randomized Backtracking
    def _backtrack_solve_random(n_rows, k, best_scope, deadline, seed):
        limit = best_scope - 1
        used_diffs = set()
        rng = random.Random(seed)
        
        def dfs(r, marks, row_diffs, acc_rows):
            if time.time() > deadline:
                return None
            
            if len(marks) == k + 1:
                if r == n_rows - 1:
                    return acc_rows + [marks]
                else:
                    used_diffs.update(row_diffs)
                    res = dfs(r + 1, [0], set(), acc_rows + [marks])
                    if res is not None:
                        return res
                    used_diffs.difference_update(row_diffs)
                    return None
            else:
                last_mark = marks[-1]
                candidates = list(range(last_mark + 1, limit + 1))
                if not candidates:
                    return None
                
                # Shuffle candidates to diversify search
                rng.shuffle(candidates)
                
                for x in candidates:
                    new_diffs = {x - a for a in marks}
                    if new_diffs & (used_diffs | row_diffs):
                        continue
                    
                    row_diffs.update(new_diffs)
                    marks.append(x)
                    
                    res = dfs(r, marks, row_diffs, acc_rows)
                    if res is not None:
                        return res
                    
                    marks.pop()
                    row_diffs.difference_update(new_diffs)
                
                return None

        res = dfs(0, [0], set(), [])
        if res:
            return res, _score(res)
        return None, None

    # Run randomized backtracking for a few seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        rows, scope = _backtrack_solve_random(n_rows, k, best_scope, deadline, seed)
        if rows is not None and scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]

    if best_rows is None:
        best_rows = _greedy_set(n_rows, k, 0)

    return best_rows

# EVOLVE-BLOCK-END
