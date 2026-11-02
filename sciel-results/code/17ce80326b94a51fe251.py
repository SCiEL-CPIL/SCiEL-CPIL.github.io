import time
import random
import sys
import numpy as np

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0 such that
    all differences between two entries of the same row are distinct over all rows.
    Minimize the largest entry.
    """
    deadline = time.time() + 112
    
    # Phase 1: Multi-start greedy initialization
    best_solution = None
    best_score = float('inf')
    
    # Try several greedy starts with different randomizations
    n_starts = 8
    for s in range(n_starts):
        if time.time() > deadline - 60:
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
    
    # Phase 2: Initial local search polish
    if time.time() < deadline - 40:
        polished = _local_search_minimize([r[:] for r in best_solution], n_rows, k, deadline)
        if polished:
            ps = max(max(r) for r in polished)
            if ps < best_score:
                best_score = ps
                best_solution = polished
    
    # Phase 3: Iterated Local Search with stronger perturbation
    no_improve_streak = 0
    while time.time() < deadline - 15:
        candidate = [r[:] for r in best_solution]
        
        # Stronger perturbation: 3-8 moves, sometimes including a full row rebuild
        n_perturb = random.randint(3, 8)
        for _ in range(n_perturb):
            if random.random() < 0.2:
                # Rebuild one row from scratch
                r_idx = random.randint(0, n_rows - 1)
                new_row = _rebuild_row(candidate, r_idx, n_rows, k)
                if new_row:
                    candidate[r_idx] = new_row
            else:
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                new_val = _find_random_valid_value(candidate, r_idx, pos, n_rows, k)
                if new_val is not None and new_val != candidate[r_idx][pos]:
                    candidate[r_idx][pos] = new_val
        
        # Local search to minimize scope
        improved = _local_search_minimize(candidate, n_rows, k, deadline)
        
        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
                no_improve_streak = 0
            elif curr_score == best_score and random.random() < 0.4:
                best_solution = improved
                no_improve_streak += 1
            else:
                no_improve_streak += 1
            
            # If stuck for a while, do a more aggressive perturbation
            if no_improve_streak >= 5:
                candidate = [r[:] for r in best_solution]
                for _ in range(random.randint(5, 12)):
                    r_idx = random.randint(0, n_rows - 1)
                    pos = random.randint(1, k)
                    new_val = _find_random_valid_value(candidate, r_idx, pos, n_rows, k)
                    if new_val is not None and new_val != candidate[r_idx][pos]:
                        candidate[r_idx][pos] = new_val
                improved = _local_search_minimize(candidate, n_rows, k, deadline)
                if improved:
                    curr_score = max(max(r) for r in improved)
                    if curr_score < best_score:
                        best_score = curr_score
                        best_solution = improved
                        no_improve_streak = 0
    
    # Phase 4: Final aggressive squeeze
    if time.time() < deadline - 3:
        final = _aggressive_squeeze([r[:] for r in best_solution], n_rows, k, deadline)
        if final:
            final_score = max(max(r) for r in final)
            if final_score < best_score:
                best_solution = final
                best_score = final_score
    
    return best_solution

def _find_random_valid_value(rows, r_idx, pos, n_rows, k):
    """
    Find a random valid value for rows[r_idx][pos] that maintains all-diff-distinct property.
    Returns None if no valid value found in a reasonable range.
    """
    row = rows[r_idx]
    old_val = row[pos]
    n = len(row)
    
    # Compute all diffs in the system
    all_diffs = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                all_diffs.add(r[j] - r[i])
    
    # Diffs involving old_val at position pos
    diffs_involving_old = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)
    
    forbidden_diffs = all_diffs - diffs_involving_old
    
    # Determine search range
    lower_bound = row[pos-1] + 1
    upper_bound = 300
    if pos + 1 < n:
        upper_bound = min(upper_bound, row[pos+1] - 1)
    
    if upper_bound < lower_bound:
        return None
    
    # Collect valid candidates (limit to avoid excessive search)
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue
        
        valid = True
        new_diffs = set()
        for i in range(n):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val
            
            if d in forbidden_diffs or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
        
        if valid:
            valid_candidates.append(new_val)
            if len(valid_candidates) > 200:
                break
    
    if not valid_candidates:
        return None
    
    return random.choice(valid_candidates)

def _rebuild_row(rows, r_idx, n_rows, k):
    """Rebuild a single row from scratch, keeping all other rows fixed."""
    other_diffs = set()
    for i, r in enumerate(rows):
        if i == r_idx:
            continue
        for a in range(len(r)):
            for b in range(a+1, len(r)):
                other_diffs.add(r[b] - r[a])
    
    row = [0]
    for pos in range(1, k+1):
        found = False
        lower = row[-1] + 1
        for x in range(lower, 350):
            valid = True
            new_diffs = set()
            for a in row:
                d = x - a
                if d in other_diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                row.append(x)
                other_diffs.update(new_diffs)
                found = True
                break
        if not found:
            return None
    return row

def _local_search_minimize(rows, n_rows, k, deadline):
    """
    Local search that tries to reduce the maximum value by moving elements to smaller positions.
    Uses incremental diff tracking for speed.
    """
    best_rows = [r[:] for r in rows]
    n = k + 1
    
    # Build initial diff set
    all_diffs = set()
    for r in best_rows:
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.add(r[j] - r[i])
    
    # Track current max
    curr_max = max(max(r) for r in best_rows)
    no_progress_count = 0
    
    while time.time() < deadline - 3:
        progress = False
        
        # Process all positions sorted by value descending
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        indices.sort(key=lambda x: best_rows[x[0]][x[1]], reverse=True)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 3:
                break
                
            row = best_rows[r_idx]
            old_val = row[pos]
            
            # Compute diffs involving (r_idx, pos)
            diffs_involving = set()
            for i in range(n):
                if i == pos:
                    continue
                diffs_involving.add(abs(row[i] - old_val))
            
            forbidden_diffs = all_diffs - diffs_involving
            
            # Try to find smaller valid value
            lower_bound = row[pos-1] + 1
            upper_bound = old_val - 1
            if pos + 1 < n:
                upper_bound = min(upper_bound, row[pos+1] - 1)
            
            if upper_bound >= lower_bound:
                found = False
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        d = abs(row[i] - new_val)
                        
                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        # Update: remove old diffs, add new diffs
                        all_diffs.difference_update(diffs_involving)
                        all_diffs.update(new_diffs)
                        best_rows[r_idx][pos] = new_val
                        progress = True
                        found = True
                        if old_val == curr_max:
                            curr_max = max(max(r) for r in best_rows)
                        break
                
                if found:
                    no_progress_count = 0
        
        if not progress:
            no_progress_count += 1
            if no_progress_count > 3:
                # Shake: randomly perturb several values to escape local minimum
                for _ in range(5):
                    r_idx = random.randint(0, n_rows - 1)
                    pos = random.randint(1, k)
                    row = best_rows[r_idx]
                    old_val = row[pos]
                    
                    diffs_involving = set()
                    for i in range(n):
                        if i == pos: continue
                        diffs_involving.add(abs(row[i] - old_val))
                    forbidden_diffs = all_diffs - diffs_involving
                    
                    lower_bound = row[pos-1] + 1
                    upper_bound = 300
                    if pos + 1 < n:
                        upper_bound = min(upper_bound, row[pos+1] - 1)
                    
                    if upper_bound < lower_bound:
                        continue
                    
                    for _ in range(80):
                        new_val = random.randint(lower_bound, upper_bound)
                        if new_val == old_val: continue
                        valid = True
                        new_diffs = set()
                        for i in range(n):
                            if i == pos: continue
                            d = abs(row[i] - new_val)
                            if d in forbidden_diffs or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            all_diffs.difference_update(diffs_involving)
                            all_diffs.update(new_diffs)
                            best_rows[r_idx][pos] = new_val
                            break
            else:
                break
                
    return best_rows

def _aggressive_squeeze(rows, n_rows, k, deadline):
    """
    Final pass: iteratively try to reduce each element to its minimum valid value.
    Multiple passes to handle cascading improvements.
    """
    best_rows = [r[:] for r in rows]
    n = k + 1
    
    # Build initial diff set
    all_diffs = set()
    for r in best_rows:
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.add(r[j] - r[i])
    
    for _pass in range(3):
        if time.time() > deadline - 2:
            break
        progress = False
        # Process from right to left, high to low
        for r_idx in range(n_rows):
            for pos in range(k, 0, -1):
                if time.time() > deadline - 2:
                    break
                row = best_rows[r_idx]
                old_val = row[pos]
                
                diffs_involving = set()
                for i in range(n):
                    if i == pos:
                        continue
                    diffs_involving.add(abs(row[i] - old_val))
                forbidden_diffs = all_diffs - diffs_involving
                
                lower_bound = row[pos-1] + 1
                upper_bound = old_val - 1
                if pos + 1 < n:
                    upper_bound = min(upper_bound, row[pos+1] - 1)
                
                if upper_bound >= lower_bound:
                    for new_val in range(lower_bound, upper_bound + 1):
                        valid = True
                        new_diffs = set()
                        for i in range(n):
                            if i == pos:
                                continue
                            d = abs(row[i] - new_val)
                            if d in forbidden_diffs or d in new_diffs:
                                valid = False
                                break
                            new_diffs.add(d)
                        if valid:
                            all_diffs.difference_update(diffs_involving)
                            all_diffs.update(new_diffs)
                            best_rows[r_idx][pos] = new_val
                            progress = True
                            break
        if not progress:
            break
    
    return best_rows

def _greedy_construct(n_rows, k, random_seed=0):
    """
    Greedy construction: build rows one by one, choosing smallest possible values.
    Uses randomization to explore different starting points.
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
            search_range = min(300, 400 - current_max)
            
            # First try sequential
            for x in range(base, base + search_range):
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
                # Try randomized search with wider range
                for _retry in range(200):
                    x = base + rng.randint(0, 200)
                    if x > 400:
                        continue
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
        rows.append(row)
    
    return rows

def _local_search_ilS(rows, n_rows, k, deadline):
    # Deprecated - use _local_search_minimize instead
    return _local_search_minimize(rows, n_rows, k, deadline)

def _repair(rows, n_rows, k):
    # Placeholder for compatibility
    return None
