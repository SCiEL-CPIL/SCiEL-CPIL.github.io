import time
import random
import sys
import numpy as np

sys.setrecursionlimit(10000)

def _compute_all_diffs(rows):
    """Compute the set of all pairwise differences across all rows."""
    all_diffs = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            for j in range(i + 1, n):
                all_diffs.add(r[j] - r[i])
    return all_diffs

def _diffs_involving_position(row, pos):
    """Compute set of diffs involving row[pos] with all other positions."""
    n = len(row)
    diffs = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            diffs.add(row[pos] - row[i])
        else:
            diffs.add(row[i] - row[pos])
    return diffs

def _check_move_valid(row, pos, new_val, forbidden_diffs):
    """Check if placing new_val at position pos is valid given forbidden_diffs."""
    n = len(row)
    new_diffs = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            d = new_val - row[i]
        else:
            d = row[i] - new_val
        if d in forbidden_diffs or d in new_diffs:
            return False, None
        new_diffs.add(d)
    return True, new_diffs

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all differences between two entries of the same row are distinct over all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 115
    
    # Phase 1: Multi-start greedy initialization
    best_solution = None
    best_score = float('inf')
    
    n_starts = 5
    for s in range(n_starts):
        if time.time() > deadline - 25:
            break
        sol = _greedy_construct(n_rows, k, random_seed=s * 12345 + 42)
        if sol:
            score = max(max(r) for r in sol)
            if score < best_score:
                best_score = score
                best_solution = sol
    
    if best_solution is None:
        best_solution = _greedy_construct(n_rows, k, random_seed=0)
        if not best_solution:
            return []
        best_score = max(max(r) for r in best_solution)
    
    # Phase 2: Iterated Local Search with valid perturbation
    while time.time() < deadline - 5:
        candidate = [r[:] for r in best_solution]
        
        # Compute all_diffs once for perturbation
        all_diffs = _compute_all_diffs(candidate)
        
        # Perform random valid moves (perturbation)
        n_perturb = random.randint(2, 6)
        for _ in range(n_perturb):
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            
            new_val = _find_random_valid_value_fast(candidate, r_idx, pos, all_diffs)
            if new_val is not None and new_val != candidate[r_idx][pos]:
                # Update all_diffs incrementally
                row = candidate[r_idx]
                old_val = row[pos]
                old_diffs = _diffs_involving_position(row, pos)
                all_diffs.difference_update(old_diffs)
                row[pos] = new_val
                new_diffs = _diffs_involving_position(row, pos)
                all_diffs.update(new_diffs)
        
        # Local search to minimize scope
        improved = _local_search_minimize_fast(candidate, n_rows, k, deadline)
        
        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
            elif curr_score == best_score and random.random() < 0.4:
                best_solution = improved
    
    # Final polish
    if time.time() < deadline - 2:
        final = _local_search_minimize_fast([r[:] for r in best_solution], n_rows, k, deadline)
        if final:
            final_score = max(max(r) for r in final)
            if final_score < best_score:
                best_solution = final
                best_score = final_score
    
    return best_solution

def _find_random_valid_value_fast(rows, r_idx, pos, all_diffs):
    """
    Find a random valid value for rows[r_idx][pos] using precomputed all_diffs.
    Returns None if no valid value found.
    """
    row = rows[r_idx]
    old_val = row[pos]
    n = len(row)
    
    diffs_involving_old = _diffs_involving_position(row, pos)
    forbidden_diffs = all_diffs - diffs_involving_old
    
    lower_bound = row[pos - 1] + 1
    upper_bound = 250
    if pos + 1 < n:
        upper_bound = min(upper_bound, row[pos + 1] - 1)
    
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue
        valid, _ = _check_move_valid(row, pos, new_val, forbidden_diffs)
        if valid:
            valid_candidates.append(new_val)
    
    if not valid_candidates:
        return None
    
    return random.choice(valid_candidates)

def _local_search_minimize_fast(rows, n_rows, k, deadline):
    """
    Fast local search: reduce max value by moving elements to smaller positions.
    Uses incremental diff tracking.
    """
    best_rows = [r[:] for r in rows]
    n = k + 1
    
    all_diffs = _compute_all_diffs(best_rows)
    
    while time.time() < deadline - 3:
        progress = False
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k + 1)]
        random.shuffle(indices)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 3:
                break
                
            row = best_rows[r_idx]
            old_val = row[pos]
            
            diffs_involving = _diffs_involving_position(row, pos)
            forbidden_diffs = all_diffs - diffs_involving
            
            lower_bound = row[pos - 1] + 1
            upper_bound = old_val - 1
            if pos + 1 < n:
                upper_bound = min(upper_bound, row[pos + 1] - 1)
            
            if upper_bound >= lower_bound:
                found = False
                for new_val in range(lower_bound, upper_bound + 1):
                    valid, new_diffs = _check_move_valid(row, pos, new_val, forbidden_diffs)
                    if valid:
                        all_diffs.difference_update(diffs_involving)
                        all_diffs.update(new_diffs)
                        best_rows[r_idx][pos] = new_val
                        progress = True
                        found = True
                        break
                
                if found:
                    continue
        
        if not progress:
            break
            
    return best_rows

def _greedy_construct(n_rows, k, random_seed=0):
    """
    Greedy construction: build rows one by one, choosing smallest possible values.
    """
    rng = random.Random(random_seed)
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            base = current_max + 1
            search_range = min(200, 300 - current_max)
            
            for x in range(base, base + search_range):
                valid = True
                new_diffs = []
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
                for _retry in range(100):
                    x = base + rng.randint(0, 100)
                    if x > 250:
                        continue
                    valid = True
                    new_diffs = []
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
        rows.append(row)
    
    return rows

def _local_search_ilS(rows, n_rows, k, deadline):
    return _local_search_minimize_fast(rows, n_rows, k, deadline)

def _repair(rows, n_rows, k):
    return None
