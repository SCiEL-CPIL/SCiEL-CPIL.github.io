"""Difference triangle set (7,5): minimize the largest entry."""
import time
import random

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs


def _valid(rows):
    """Check rows are valid: start at 0, strictly increasing, globally distinct diffs."""
    all_d = set()
    for row in rows:
        if not row or row[0] != 0:
            return False
        for i in range(len(row) - 1):
            if row[i + 1] <= row[i]:
                return False
        rd = _row_diffs(row)
        if rd & all_d:
            return False
        all_d |= rd
    return True


def _scope(rows):
    """Return the largest entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_construct(n_rows, k, randomize=False, seed=None):
    """Greedy construction with optional randomization."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    used = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            if randomize:
                # Try a random step within reasonable bounds
                if row:
                    max_step = max(1, int((k + 1) * 2.0))
                    found = False
                    for _attempt in range(30):
                        step = rng.randint(1, max_step)
                        x = row[-1] + step
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            found = True
                            break
                    if not found:
                        # Fallback to sequential
                        x = row[-1] + 1
                        while True:
                            new_diffs = {x - a for a in row}
                            if not new_diffs & used:
                                break
                            x += 1
                else:
                    x = 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            break
                        x += 1
            else:
                x += 1
                new_diffs = {x - a for a in row}
                if new_diffs & used:
                    continue
            
            row.append(x)
            used |= new_diffs
        rows.append(row)
    
    return rows


def _backtrack_row(used, k, max_val, deadline, random_order=False, rng=None):
    """Backtrack search for a single row of length k+1 starting at 0,
    max entry <= max_val, with all differences disjoint from used."""
    if rng is None:
        rng = random
    row = [0]
    row_diffs = set()
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == k + 1:
            return True
        
        # Generate candidate values
        candidates = list(range(row[-1] + 1, max_val + 1))
        if random_order:
            rng.shuffle(candidates)
        
        for x in candidates:
            new_diffs = {x - a for a in row}
            if new_diffs & used:
                continue
            if new_diffs & row_diffs:
                continue
            
            row.append(x)
            row_diffs.update(new_diffs)
            if backtrack(pos + 1):
                return True
            row_diffs.difference_update(new_diffs)
            row.pop()
        
        return False
    
    if backtrack(1):
        return list(row)
    return None


def _construct_with_backtracking(n_rows, k, deadline, rng=None):
    """Construct rows one by one using backtracking, minimizing max entry."""
    if rng is None:
        rng = random
    used = set()
    rows = []
    
    for r_idx in range(n_rows):
        # Determine what max value we need to search up to
        current_max = max((max(r) for r in rows), default=0)
        search_max = current_max + k + 15
        
        # Try to find a row with minimal max entry
        found = None
        for target_max in range(current_max + 1, search_max + 1):
            if time.time() > deadline - 5.0:
                break
            row = _backtrack_row(used, k, target_max, deadline - 5.0, 
                                  random_order=(r_idx % 2 == 1), rng=rng)
            if row is not None:
                found = row
                break
        
        if found is None:
            # Fallback to greedy
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            found = row
        
        rows.append(found)
        used |= _row_diffs(found)
    
    return rows


def _local_search_improve(rows, k, deadline, rng=None):
    """Local search: try to reduce max entry by modifying individual elements with incremental diff checks."""
    if rng is None:
        rng = random
    best = [list(r) for r in rows]
    best_scope = _scope(best)
    
    # Precompute global used diffs
    used = set()
    for r in best:
        used |= _row_diffs(r)
    
    while time.time() < deadline - 2.0:
        improved = False
        current_scope = _scope(best)
        
        # Find the row with the maximum entry
        max_row_idx = max(range(len(best)), key=lambda i: max(best[i]))
        
        # Try to reduce elements in that row from the end
        row = best[max_row_idx]
        for pos in range(len(row) - 1, 0, -1):
            if time.time() > deadline - 2.0:
                break
            old_val = row[pos]
            lower_bound = row[pos - 1] + 1
            
            # Try values from lower_bound to old_val - 1
            for new_val in range(lower_bound, old_val):
                if time.time() > deadline - 2.0:
                    break
                
                # Compute differences that change
                # Old diffs involving position pos
                old_diffs = set()
                for i in range(pos):
                    old_diffs.add(old_val - row[i])
                
                # New diffs involving position pos
                new_diffs = set()
                for i in range(pos):
                    new_diffs.add(new_val - row[i])
                
                # Check if new diffs conflict with used (excluding old diffs that will be removed)
                remaining_used = used - old_diffs
                if new_diffs & remaining_used:
                    continue
                
                # Check internal consistency: new diffs shouldn't conflict with other diffs in the same row
                # that don't involve pos
                other_diffs = set()
                for i in range(pos):
                    for j in range(i + 1, pos):
                        other_diffs.add(row[j] - row[i])
                
                if new_diffs & other_diffs:
                    continue
                
                # Tentatively apply
                row[pos] = new_val
                # Update used set
                used -= old_diffs
                used |= new_diffs
                
                # Full validity check for safety
                if _valid(best):
                    new_scope = _scope(best)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        improved = True
                        break
                    else:
                        # Revert
                        row[pos] = old_val
                        used -= new_diffs
                        used |= old_diffs
                else:
                    # Revert
                    row[pos] = old_val
                    used -= new_diffs
                    used |= old_diffs
            
            if improved:
                break
        
        if not improved:
            # Try perturbations
            if rng.random() < 0.5:
                ridx = rng.randint(0, len(best) - 1)
                p1 = rng.randint(1, len(best[ridx]) - 2)
                p2 = rng.randint(p1 + 1, len(best[ridx]) - 1)
                saved1, saved2 = best[ridx][p1], best[ridx][p2]
                best[ridx][p1], best[ridx][p2] = saved2, saved1
                if not _valid(best):
                    best[ridx][p1], best[ridx][p2] = saved1, saved2
                else:
                    # Accept perturbation to explore
                    used = set()
                    for r in best:
                        used |= _row_diffs(r)
            else:
                break
    
    return best, best_scope


def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Greedy baseline
    rows = _greedy_construct(n_rows, k)
    best_rows = rows
    best_scope = _scope(rows)
    
    # Phase 2: Backtracking construction (try to do better)
    if time.time() < deadline - 30.0:
        bt_rows = _construct_with_backtracking(n_rows, k, deadline - 30.0)
        bt_scope = _scope(bt_rows)
        if bt_scope < best_scope:
            best_rows = bt_rows
            best_scope = bt_scope
    
    # Phase 3: Local search improvement
    if time.time() < deadline - 15.0:
        ls_rows, ls_scope = _local_search_improve(best_rows, k, deadline - 15.0)
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope
    
    # Phase 4: Try multiple random restarts if time permits
    attempts = 0
    while time.time() < deadline - 10.0 and attempts < 100:
        attempts += 1
        # Random greedy
        rrows = _greedy_construct(n_rows, k, randomize=True, seed=attempts * 137)
        rs = _scope(rrows)
        if rs < best_scope:
            best_rows = rrows
            best_scope = rs
            # Quick local search on the new best
            if time.time() < deadline - 10.0:
                ls2, ls2s = _local_search_improve(best_rows, k, deadline - 10.0)
                if ls2s < best_scope:
                    best_rows = ls2
                    best_scope = ls2s
    
    # Phase 5: Final local search pass
    if time.time() < deadline - 5.0:
        ls_rows, ls_scope = _local_search_improve(best_rows, k, deadline - 2.0)
        if ls_scope < best_scope:
            best_rows = ls_rows
            best_scope = ls_scope
    
    return best_rows

# EVOLVE-BLOCK-END
