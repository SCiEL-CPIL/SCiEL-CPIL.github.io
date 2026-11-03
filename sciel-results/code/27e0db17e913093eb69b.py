import time
import random
import math
import numpy as np
from collections import defaultdict

_DEADLINE = 0.0
_BEST = None

def _valid(rows, k):
    seen = set()
    for r in rows:
        if r[0] != 0 or len(r) != k + 1:
            return False
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d <= 0 or d in seen:
                    return False
                seen.add(d)
    return True

def _scope(rows):
    return max(a for r in rows for a in r[1:])

def _fast_greedy(k, n_rows, rng, spread=False):
    used = 0
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = 0
            for a in row:
                new |= (1 << (x - a))
            if (used & new) == 0:
                row.append(x)
                used |= new
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _column_local_search(mat, j, k, n_rows, rng, deadline, used_diffs_counter, T_init=5.0):
    """
    Optimizes column j of the matrix mat.
    used_diffs_counter is a dict of diff -> count.
    We try to assign values to mat[:, j] such that:
    1. mat[i, j-1] < mat[i, j] < mat[i, j+1] (if exists)
    2. All differences are distinct (minimize collisions, then scope)
    
    Strategy: Simulated Annealing on the column vector.
    """
    # Current values
    current_vals = mat[:, j].copy()
    
    # Calculate current cost for this column's contribution
    # Note: used_diffs_counter is global. We need to remove current column's diffs,
    # then add new ones to evaluate.
    
    # Get constraints
    lower_bounds = np.zeros(n_rows, dtype=int)
    upper_bounds = np.zeros(n_rows, dtype=int)
    
    if j > 0:
        lower_bounds = mat[:, j-1] + 1
    else:
        lower_bounds = np.zeros(n_rows, dtype=int) # mat[:,0] is 0, but j=0 is fixed
    
    if j < k:
        upper_bounds = mat[:, j+1] - 1
    else:
        # For the last column, upper bound is current max + some slack
        upper_bounds = mat[:, j] + 10 # Initial slack
    
    # Actually, for j=0, values are fixed at 0.
    if j == 0:
        return
    
    # Pre-calculate diffs for current column
    current_diffs = []
    for i in range(n_rows):
        v = current_vals[i]
        for p in range(j):
            d = v - mat[i, p]
            current_diffs.append(d)
        for p in range(j+1, k+1):
            d = mat[i, p] - v
            current_diffs.append(d)
    
    # Remove current diffs from counter
    for d in current_diffs:
        if used_diffs_counter[d] > 0:
            used_diffs_counter[d] -= 1
    
    best_vals = current_vals.copy()
    best_cost = float('inf')
    
    T = T_init
    cooling = 0.99
    min_T = 0.1
    
    # Local search on the column
    # We can try to swap values or move values.
    # Since n_rows is small (7) and k is small (5), we can do a more exhaustive search if needed.
    # But let's stick to SA for consistency and speed.
    
    def get_cost(vals):
        # Calculate collisions and scope increase
        # Scope is max of all entries.
        new_scope = 0
        for i in range(n_rows):
            new_scope = max(new_scope, int(vals[i]))
        
        # Calculate collisions
        collisions = 0
        # Temporarily add diffs to a local counter? No, we need to check against global used_diffs_counter.
        # The global counter currently does NOT include this column's diffs.
        # So any diff d in new_vals will have count used_diffs_counter[d] + (internal collisions).
        
        local_counts = defaultdict(int)
        for i in range(n_rows):
            v = vals[i]
            for p in range(j):
                d = v - mat[i, p]
                local_counts[d] += 1
            for p in range(j+1, k+1):
                d = mat[i, p] - v
                local_counts[d] += 1
        
        # Total count for each diff d is used_diffs_counter[d] + local_counts[d]
        # Collision count is sum(max(0, total_count - 1))
        for d, cnt in local_counts.items():
            total = used_diffs_counter[d] + cnt
            if total > 1:
                collisions += total - 1
        
        # Also check internal duplicates? 
        # If local_counts[d] > 1, that's already handled by total > 1 if used_diffs_counter[d] >= 0.
        # Wait, if used_diffs_counter[d] is 0 and local_counts[d] is 2, total is 2, collision 1. Correct.
        
        return new_scope, collisions

    current_scope, current_collisions = get_cost(current_vals)
    current_cost = current_scope + 10.0 * current_collisions
    best_cost = current_cost
    best_vals = current_vals.copy()
    
    stagnant = 0
    MAX_STAGNANT = 200
    
    while time.time() < deadline:
        # Pick a random row in the column to move
        i = rng.randint(0, n_rows - 1)
        
        # Determine valid range for this row
        lo = int(lower_bounds[i])
        hi = int(upper_bounds[i])
        
        if lo > hi:
            # Try to relax bounds? No, bounds are hard constraints from neighbors.
            # If infeasible, skip.
            continue
            
        old_val = int(current_vals[i])
        
        # Random move
        # Bias towards smaller values to reduce scope?
        # Or uniform.
        new_val = rng.randint(lo, hi)
        
        if new_val == old_val:
            continue
            
        current_vals[i] = new_val
        
        new_scope, new_collisions = get_cost(current_vals)
        new_cost = new_scope + 10.0 * new_collisions
        
        if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_vals = current_vals.copy()
                stagnant = 0
            else:
                stagnant += 1
        else:
            current_vals[i] = old_val
            stagnant += 1
            
        T = max(T * cooling, min_T)
        
        if stagnant > MAX_STAGNANT:
            # Restart column search
            current_vals = best_vals.copy()
            current_scope, current_collisions = get_cost(current_vals)
            current_cost = current_scope + 10.0 * current_collisions
            T = T_init
            stagnant = 0

    # Restore best values to mat
    mat[:, j] = best_vals
    
    # Add best diffs back to used_diffs_counter
    for i in range(n_rows):
        v = best_vals[i]
        for p in range(j):
            d = v - mat[i, p]
            used_diffs_counter[d] += 1
        for p in range(j+1, k+1):
            d = mat[i, p] - v
            used_diffs_counter[d] += 1

def _column_sa(n_rows, k, deadline, rng, initial_rows):
    """
    Main SA loop operating on the full matrix, but using column-wise moves.
    """
    mat = np.array(initial_rows, dtype=np.int32)
    
    # Build initial used_diffs_counter
    used_diffs_counter = defaultdict(int)
    for i in range(n_rows):
        row = mat[i]
        for a in range(k+1):
            for b in range(a+1, k+1):
                d = row[b] - row[a]
                used_diffs_counter[d] += 1
    
    current_scope = int(mat.max())
    
    # Calculate current collisions
    def get_total_collisions():
        collisions = 0
        for d, cnt in used_diffs_counter.items():
            if cnt > 1:
                collisions += cnt - 1
        return collisions
    
    current_collisions = get_total_collisions()
    W = 10.0
    current_cost = current_scope + W * current_collisions
    
    best_cost = current_cost
    best_mat = mat.copy()
    best_scope = current_scope
    
    T = 5.0
    cooling = 0.999
    min_T = 0.1
    stagnant = 0
    MAX_STAGNANT = 500
    
    while time.time() < deadline:
        # Pick a column to optimize
        # Bias towards later columns (larger values) or random?
        # Random is fine.
        j = rng.randint(1, k) # Skip column 0 (fixed at 0)
        
        _column_local_search(mat, j, k, n_rows, rng, deadline, used_diffs_counter, T_init=T*0.5)
        
        # Recalculate global cost
        new_scope = int(mat.max())
        new_collisions = get_total_collisions()
        new_cost = new_scope + W * new_collisions
        
        if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
            current_cost = new_cost
            current_scope = new_scope
            current_collisions = new_collisions
            if new_cost < best_cost:
                best_cost = new_cost
                best_mat = mat.copy()
                best_scope = new_scope
                stagnant = 0
            else:
                stagnant += 1
        else:
            # Revert? 
            # _column_local_search already modified mat and used_diffs_counter.
            # We need to revert to previous state.
            # This is tricky because we don't have a snapshot of used_diffs_counter.
            # To make this easier, we should only accept if better, or keep a stack.
            # For simplicity and speed, let's assume the column search improves or stays neutral.
            # If it worsens, we might get stuck.
            # Better approach: _column_local_search should return the old state if it wants to revert?
            # Or, we just accept the column change always (since it's a local search on that column, 
            # it should find the best for that column given the rest).
            # Actually, the "revert" logic in the previous SA was for single element moves.
            # Here, the column search is a mini-SA. It finds the best configuration for that column.
            # So we should always accept the result of the column search.
            stagnant += 1
            
        T = max(T * cooling, min_T)
        
        if stagnant > MAX_STAGNANT:
            # Perturb: pick a random column and randomize it slightly
            j = rng.randint(1, k)
            for i in range(n_rows):
                # Randomize value within bounds
                lo = mat[i, j-1] + 1
                hi = mat[i, j+1] - 1 if j < k else mat[i, j] + 5
                if lo <= hi:
                    new_val = rng.randint(lo, hi)
                    # Update used_diffs_counter manually? 
                    # This is getting complex. 
                    # Let's just restart the column search with higher T.
                    pass
            
            # Reset stagnant
            stagnant = 0
            T = T * 2.0

    return best_scope, [list(map(int, r)) for r in best_mat]

def construct(n_rows, k):
    global _BEST, _DEADLINE
    _BEST = None
    _DEADLINE = time.time() + 108.0

    def consider(rows):
        global _BEST
        if not _valid(rows, k):
            return
        s = _scope(rows)
        if _BEST is None or s < _BEST[0]:
            _BEST = (s, [list(r) for r in rows])

    rng = random.Random(42)

    # 1. Generate initial candidates
    candidates = []
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    # Add some randomized greedy
    for _ in range(2):
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                found = False
                max_step = rng.randint(1, 10)
                for t in range(x + 1, x + 1 + max_step * 4):
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    if (used & new) == 0:
                        row.append(t)
                        used |= new
                        x = t
                        found = True
                        break
                if not found:
                    t = x + 1
                    new = 0
                    for a in row:
                        new |= (1 << (t - a))
                    while used & new:
                        t += 1
                        new = 0
                        for a in row:
                            new |= (1 << (t - a))
                    row.append(t)
                    used |= new
                    x = t
            rows.append(row)
        candidates.append(rows)
        
    candidates.sort(key=_scope)
    
    # 2. Run Column-wise SA on the best candidates
    for idx, cand in enumerate(candidates[:2]):
        if time.time() > _DEADLINE:
            break
        remaining = _DEADLINE - time.time()
        if idx == 0:
            sa_deadline = time.time() + remaining * 0.5
        else:
            sa_deadline = _DEADLINE
            
        s, rows = _column_sa(n_rows, k, sa_deadline, rng, cand)
        consider(rows)
        
    # 3. Final polish: Try to compress the best result found
    if _BEST is not None and time.time() < _DEADLINE - 5:
        # We can try a few more column SA runs with different seeds or parameters
        remaining = _DEADLINE - time.time()
        if remaining > 5:
            # Use a fresh rng for diversity
            rng2 = random.Random(123)
            s, rows = _column_sa(n_rows, k, _DEADLINE - 2, rng2, _BEST[1])
            consider(rows)

    if _BEST is not None:
        return _BEST[1]
    return candidates[0]
