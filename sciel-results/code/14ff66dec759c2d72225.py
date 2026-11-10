"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START

def _diff_set(row):
    """Compute the set of positive differences in a row."""
    s = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ai)
    return s

def _validate(rows):
    """Check if rows are valid: start at 0, strictly increasing, all diffs distinct."""
    used = set()
    for r in rows:
        if not r or r[0] != 0:
            return False
        for a, b in zip(r, r[1:]):
            if b <= a:
                return False
        for d in _diff_set(r):
            if d in used:
                return False
            used.add(d)
    return True

def _score(rows):
    """Return the largest entry across rows (the scope)."""
    return max(r[-1] for r in rows)

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

def _backtrack_row(used, scope, time_limit=None):
    """
    Build one row of size k+1 with max <= scope, all diffs disjoint from used.
    Returns row or None.
    """
    def now():
        return time.time()
    
    row = [0]
    cur_used = set(used)
    target_len = k + 1
    
    def dfs(row, cur_used):
        if time_limit is not None and now() > time_limit:
            return None
        if len(row) == target_len:
            return row[:]
        
        last = row[-1]
        rem = target_len - len(row)
        
        min_next = last + 1
        max_next = scope - (rem - 1)
        
        if min_next > max_next:
            return None
        
        for x in range(min_next, max_next + 1):
            if time_limit is not None and now() > time_limit:
                return None
            
            # Check conflicts
            conflict = False
            for a in row:
                d = x - a
                if d in cur_used:
                    conflict = True
                    break
            if conflict:
                continue
            
            new_row = row + [x]
            new_cur = cur_used.copy()
            for a in row:
                new_cur.add(x - a)
            
            res = dfs(new_row, new_cur)
            if res is not None:
                return res
        return None
    
    return dfs(row, cur_used)

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0, minimizing the
    largest entry, subject to all positive differences within rows being distinct
    across all rows.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    def now():
        return time.time()
    
    # 1. Get initial upper bound via multiple greedy seeds
    best_rows = None
    best_scope = float('inf')
    
    for seed in range(30):
        if now() > deadline - 30:
            break
        rows = _greedy_set(n_rows, k, seed)
        scope = _score(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
    
    if best_rows is None:
        best_rows = _greedy_set(n_rows, k, 0)
        best_scope = _score(best_rows)
    
    # 2. Iterative improvement: try to replace rows with smaller ones
    #    Sort rows by their max descending; try to improve the largest first
    improved = True
    while improved and now() < deadline - 20:
        improved = False
        order = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        
        for i in order:
            if now() > deadline - 20:
                break
            
            cur_max_row = best_rows[i][-1]
            target = cur_max_row - 1
            
            # Compute fixed used diffs from other rows
            other_rows = [r for j, r in enumerate(best_rows) if j != i]
            fixed_used = set()
            for r in other_rows:
                fixed_used |= _diff_set(r)
            
            # Try to find a replacement with strictly smaller max
            new_row = _backtrack_row(fixed_used, target, time_limit=deadline - 20)
            if new_row is not None:
                new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                if _validate(new_rows):
                    mv = _score(new_rows)
                    if mv < best_scope:
                        best_scope = mv
                        best_rows = new_rows
                        improved = True
    
    # 3. Pair-swap refinement: try to swap pairs of rows to find better packings
    if now() < deadline - 15:
        pairs = [(i, j) for i in range(n_rows) for j in range(i + 1, n_rows)]
        random.shuffle(pairs)
        
        for (i, j) in pairs:
            if now() > deadline - 15:
                break
            
            # Get other rows
            other_rows = [r for x, r in enumerate(best_rows) if x not in (i, j)]
            fixed_used = set()
            for r in other_rows:
                fixed_used |= _diff_set(r)
            
            # Try to find two new rows that fit in the scope of the current pair
            scope_i = best_rows[i][-1]
            scope_j = best_rows[j][-1]
            target_scope = max(scope_i, scope_j) - 1
            
            # Try to build row i with target_scope
            row_i_new = _backtrack_row(fixed_used, target_scope, time_limit=deadline - 15)
            if row_i_new is not None:
                # Update fixed_used with row_i_new's diffs
                fixed_used_with_i = fixed_used | _diff_set(row_i_new)
                
                # Try to build row j with target_scope
                row_j_new = _backtrack_row(fixed_used_with_i, target_scope, time_limit=deadline - 15)
                if row_j_new is not None:
                    new_rows = [r if x not in (i, j) else (row_i_new if x == i else row_j_new) 
                                for x, r in enumerate(best_rows)]
                    if _validate(new_rows):
                        mv = _score(new_rows)
                        if mv < best_scope:
                            best_scope = mv
                            best_rows = new_rows
    
    # 4. Randomized backtracking for global improvement
    def _backtrack_solve_random(n_rows, k, best_scope, deadline, seed):
        limit = best_scope - 1
        used_diffs = set()
        rng = random.Random(seed)
        
        def dfs(r, marks, row_diffs, acc_rows):
            if now() > deadline:
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
    for seed in range(20):
        if now() > deadline - 5:
            break
        rows, scope = _backtrack_solve_random(n_rows, k, best_scope, deadline - 5, seed)
        if rows is not None and scope < best_scope:
            best_scope = scope
            best_rows = [r[:] for r in rows]
    
    return best_rows

# EVOLVE-BLOCK-END
