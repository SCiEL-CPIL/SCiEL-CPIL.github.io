import time
import sys
import random
import numpy as np

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 110
    
    # Multi-start greedy: try several different random orderings / seeds
    best_solution = None
    best_score = float('inf')
    
    # First, a deterministic greedy for a baseline
    sol = _greedy_construct(n_rows, k, random_state=42)
    if sol:
        s = max(max(r) for r in sol)
        if s < best_score:
            best_score = s
            best_solution = sol
    
    # A few randomized starts to diversify
    n_starts = 6
    for seed in range(n_starts):
        if time.time() > deadline - 60:
            break
        sol = _greedy_construct(n_rows, k, random_state=1000 + seed * 7)
        if sol:
            s = max(max(r) for r in sol)
            if s < best_score:
                best_score = s
                best_solution = sol
    
    if best_solution is None:
        return []
    
    # Simulated annealing with valid moves (single-position changes that preserve validity)
    best_solution = _simulated_annealing(best_solution, n_rows, k, deadline)
    final_score = max(max(r) for r in best_solution)
    if final_score < best_score:
        best_score = final_score
    
    # Final polish: local search (greedy descent)
    if time.time() < deadline - 5:
        polished = _local_search_greedy(best_solution, n_rows, k, deadline)
        if polished:
            ps = max(max(r) for r in polished)
            if ps < best_score:
                best_score = ps
                best_solution = polished
    
    return best_solution


def _compute_all_diffs(rows):
    """Compute the set of all positive differences within each row."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
    return diffs


def _diffs_involving_position(row, pos):
    """Compute diffs that involve element at position pos in this row."""
    n = len(row)
    d = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            d.add(row[pos] - row[i])
        else:
            d.add(row[i] - row[pos])
    return d


def _find_valid_values_for_position(rows, r_idx, pos, upper_limit=400):
    """
    Find all valid values for rows[r_idx][pos] that maintain global diff uniqueness.
    Returns a list of valid values (sorted).
    """
    row = rows[r_idx]
    n = len(row)
    old_val = row[pos]
    
    # Precompute all diffs
    all_diffs = _compute_all_diffs(rows)
    
    # Diffs involving old_val
    diffs_old = _diffs_involving_position(row, pos)
    
    # Forbidden diffs: all diffs not involving this position
    forbidden = all_diffs - diffs_old
    
    lower = row[pos - 1] + 1
    upper = upper_limit
    if pos + 1 < n:
        upper = min(upper, row[pos + 1] - 1)
    
    valid = []
    for v in range(lower, upper + 1):
        if v == old_val:
            continue
        ok = True
        new_diffs = set()
        for i in range(n):
            if i == pos:
                continue
            if i < pos:
                d = v - row[i]
            else:
                d = row[i] - v
            if d in forbidden or d in new_diffs:
                ok = False
                break
            new_diffs.add(d)
        if ok:
            valid.append(v)
    
    return valid


def _greedy_construct(n_rows, k, random_state=0):
    """
    Greedy construction: build each row one at a time, choosing the smallest
    possible next element that doesn't conflict with used diffs.
    Randomization: shuffle row order and add jitter to the search.
    """
    rng = random.Random(random_state)
    used_diffs = set()
    rows = []
    
    # Order in which to build rows (randomize for diversity)
    row_order = list(range(n_rows))
    rng.shuffle(row_order)
    
    rows = [None] * n_rows
    
    for idx in row_order:
        row = [0]
        current_max = 0
        found = True
        while len(row) < k + 1:
            if not found:
                return None
            found = False
            # Search range: try from current_max+1 upward
            search_start = current_max + 1
            search_end = current_max + 600
            for x in range(search_start, search_end):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                # Wider randomized search
                for _retry in range(100):
                    x = current_max + 1 + rng.randint(0, 200)
                    new_diffs = []
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        row.append(x)
                        used_diffs.update(new_diffs)
                        current_max = x
                        found = True
                        break
                if not found:
                    return None
        rows[idx] = row
    
    return rows


def _simulated_annealing(initial_rows, n_rows, k, deadline):
    """
    Simulated annealing over valid single-position moves.
    A move picks (r_idx, pos) and changes the value to another valid value
    (maintaining global diff uniqueness). The "cost" is the max entry.
    """
    rows = [r[:] for r in initial_rows]
    n = k + 1
    current_max = max(max(r) for r in rows)
    best_rows = [r[:] for r in rows]
    best_max = current_max
    
    # Temperature schedule
    t_start = 50.0
    t_end = 0.5
    t_current = t_start
    
    # Precompute for speed
    all_diffs = _compute_all_diffs(rows)
    
    max_steps = 50000
    step = 0
    
    while time.time() < deadline - 15 and step < max_steps:
        step += 1
        
        # Pick random position
        r_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        row = rows[r_idx]
        old_val = row[pos]
        
        # Get valid values for this position
        # To speed up, limit the search range
        lower = row[pos - 1] + 1
        upper = min(350, old_val + 50)  # allow some increase for escape
        if pos + 1 < n:
            upper = min(upper, row[pos + 1] - 1)
        
        if upper < lower:
            continue
        
        # Find valid values in range
        diffs_old = _diffs_involving_position(row, pos)
        forbidden = all_diffs - diffs_old
        
        valid_vals = []
        for v in range(lower, upper + 1):
            if v == old_val:
                continue
            ok = True
            new_diffs = set()
            for i in range(n):
                if i == pos:
                    continue
                if i < pos:
                    d = v - row[i]
                else:
                    d = row[i] - v
                if d in forbidden or d in new_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if ok:
                valid_vals.append(v)
        
        if not valid_vals:
            continue
        
        new_val = random.choice(valid_vals)
        
        # Compute new max
        new_row_max = max(row[:pos] + [new_val] + row[pos+1:])
        # Recompute global max
        new_max = max(new_row_max, max(max(r) for r in rows if r is not row))
        # Actually this is expensive; approximate:
        # The global max can only decrease if we reduce the current max entry,
        # or stay same, or increase.
        
        # Simpler: just track the max of the changed row vs the max of all other rows
        other_max = 0
        for r in rows:
            if r is not row:
                m = max(r)
                if m > other_max:
                    other_max = m
        
        new_global_max = max(new_row_max, other_max)
        delta_cost = new_global_max - current_max
        
        # Accept?
        if delta_cost <= 0:
            # Accept
            # Update all_diffs
            all_diffs.difference_update(diffs_old)
            all_diffs.update(new_diffs)
            row[pos] = new_val
            current_max = new_global_max
            
            if current_max < best_max:
                best_max = current_max
                best_rows = [r[:] for r in rows]
        else:
            # Metropolis acceptance
            if random.random() < np.exp(-delta_cost / max(t_current, 0.01)):
                all_diffs.difference_update(diffs_old)
                all_diffs.update(new_diffs)
                row[pos] = new_val
                current_max = new_global_max
        
        # Cool down
        t_current *= 0.9995
    
    return best_rows


def _local_search_greedy(rows, n_rows, k, deadline):
    """
    Greedy local search: repeatedly try to reduce the max entry by finding
    a valid smaller value for some position. Continue until no improvement
    or time runs out.
    """
    rows = [r[:] for r in rows]
    n = k + 1
    all_diffs = _compute_all_diffs(rows)
    
    while time.time() < deadline - 8:
        progress = False
        # Try positions in order of largest values first
        candidates = []
        for r_idx in range(n_rows):
            for pos in range(1, k + 1):
                candidates.append((rows[r_idx][pos], r_idx, pos))
        candidates.sort(reverse=True)  # try largest values first
        
        for val, r_idx, pos in candidates:
            if time.time() > deadline - 8:
                break
            
            row = rows[r_idx]
            old_val = row[pos]
            
            lower = row[pos - 1] + 1
            upper = old_val - 1
            if pos + 1 < n:
                upper = min(upper, row[pos + 1] - 1)
            
            if upper < lower:
                continue
            
            diffs_old = _diffs_involving_position(row, pos)
            forbidden = all_diffs - diffs_old
            
            # Try smallest valid value
            for new_val in range(lower, upper + 1):
                ok = True
                new_diffs = set()
                for i in range(n):
                    if i == pos:
                        continue
                    if i < pos:
                        d = new_val - row[i]
                    else:
                        d = row[i] - new_val
                    if d in forbidden or d in new_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if ok:
                    # Accept
                    all_diffs.difference_update(diffs_old)
                    all_diffs.update(new_diffs)
                    row[pos] = new_val
                    progress = True
                    break
            
            if progress:
                break  # Restart the pass with updated values
        
        if not progress:
            break
    
    return rows
