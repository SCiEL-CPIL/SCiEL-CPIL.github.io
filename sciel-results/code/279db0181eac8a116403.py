import time
import random
import sys

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0.
    All differences between entries in the same row must be distinct over all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 150s limit (was 120s, now 150s)
    
    best_rows = None
    best_scope = float('inf')
    
    def get_scope(rows):
        return max(max(r) for r in rows) if rows else 0

    def _row_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def _valid(rows):
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

    def _greedy_construct(seed, randomize=False):
        rng = random.Random(seed)
        used = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                if randomize:
                    found = False
                    for _attempt in range(50):
                        step = rng.randint(1, max(1, int((k + 1) * 2.0)))
                        x = row[-1] + step
                        new_diffs = {x - a for a in row}
                        if not new_diffs & used:
                            found = True
                            break
                    if not found:
                        x = row[-1] + 1
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
        
        return rows, used

    def _compress_rows(rows, deadline):
        """Systematically compress rows by trying to reduce each element."""
        nonlocal best_rows, best_scope
        
        best = [list(r) for r in rows]
        best_scope_local = get_scope(best)
        
        used = set()
        for r in best:
            used |= _row_diffs(r)
        
        k1 = k + 1
        
        while time.time() < deadline:
            improved = False
            
            # Sort positions by current value descending to target largest first
            positions = []
            for ri in range(n_rows):
                for pi in range(1, k1):
                    positions.append((best[ri][pi], ri, pi))
            positions.sort(reverse=True)
            
            for val, ri, pi in positions:
                if time.time() > deadline:
                    break
                
                row = best[ri]
                old_val = row[pi]
                if old_val <= 1:
                    continue
                
                lower = row[pi - 1] + 1
                upper = old_val - 1
                
                if lower > upper:
                    continue
                
                found_new = None
                
                for new_val in range(lower, upper + 1):
                    new_diffs = set()
                    for i in range(pi):
                        new_diffs.add(new_val - row[i])
                    for i in range(pi + 1, k1):
                        new_diffs.add(row[i] - new_val)
                    
                    old_diffs = set()
                    for i in range(pi):
                        old_diffs.add(old_val - row[i])
                    for i in range(pi + 1, k1):
                        old_diffs.add(row[i] - old_val)
                    
                    remaining = used - old_diffs
                    if new_diffs & remaining:
                        continue
                    
                    found_new = new_val
                    break
                
                if found_new is not None:
                    row[pi] = found_new
                    
                    old_diffs = set()
                    for i in range(pi):
                        old_diffs.add(old_val - row[i])
                    for i in range(pi + 1, k1):
                        old_diffs.add(row[i] - old_val)
                    
                    new_diffs = set()
                    for i in range(pi):
                        new_diffs.add(found_new - row[i])
                    for i in range(pi + 1, k1):
                        new_diffs.add(row[i] - found_new)
                    
                    used.difference_update(old_diffs)
                    used.update(new_diffs)
                    
                    new_scope = get_scope(best)
                    if new_scope < best_scope_local:
                        best_scope_local = new_scope
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [r.copy() for r in best]
                    improved = True
            
            if not improved:
                # Try random perturbation to escape local minimum
                if random.random() < 0.5:
                    ri = random.randint(0, n_rows - 1)
                    pi = random.randint(1, k1 - 2)
                    saved = best[ri][pi]
                    delta = random.randint(1, 8)
                    best[ri][pi] = saved + delta
                    if _valid(best):
                        used = set()
                        for r in best:
                            used |= _row_diffs(r)
                    else:
                        best[ri][pi] = saved
                else:
                    break
        
        return best, best_scope_local

    def _local_search(rows, used_set, deadline):
        """Randomized local search with row shuffling to break symmetry."""
        nonlocal best_rows, best_scope
        
        rows = [r.copy() for r in rows]
        used = set(used_set)
        current_scope = get_scope(rows)
        
        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [r.copy() for r in rows]
        
        no_improve_count = 0
        max_no_improve = 3000
        
        while time.time() < deadline:
            # With 30% probability, shuffle row order to break symmetry
            if random.random() < 0.3:
                random.shuffle(rows)
                # Rebuild used set (order doesn't matter for the set)
                used = set()
                for r in rows:
                    used |= _row_diffs(r)
            
            # Pick a random row and element to try to decrease
            if random.random() < 0.4:
                # Target global max
                max_val = -1
                max_r = -1
                max_j = -1
                for r_idx, row in enumerate(rows):
                    for j_idx, val in enumerate(row):
                        if val > max_val:
                            max_val = val
                            max_r = r_idx
                            max_j = j_idx
                if max_j == 0:
                    continue
                r_idx, j_idx = max_r, max_j
                new_val = rows[r_idx][j_idx] - 1
            else:
                r_idx = random.randint(0, n_rows - 1)
                j_idx = random.randint(1, k)
                new_val = rows[r_idx][j_idx] - 1
            
            if new_val < 0:
                continue
            
            row = rows[r_idx]
            if j_idx > 0 and new_val <= row[j_idx - 1]:
                continue
            if j_idx < k and new_val >= row[j_idx + 1]:
                continue
            
            # Check validity
            old_diffs = _row_diffs(row)
            new_row = row.copy()
            new_row[j_idx] = new_val
            new_diffs = _row_diffs(new_row)
            
            removed = old_diffs - new_diffs
            added = new_diffs - old_diffs
            other_diffs = used - removed
            
            if added & other_diffs:
                no_improve_count += 1
                continue
            
            # Apply move
            used.difference_update(old_diffs)
            used.update(new_diffs)
            row[j_idx] = new_val
            
            new_scope = get_scope(rows)
            if new_scope < current_scope:
                current_scope = new_scope
                no_improve_count = 0
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r.copy() for r in rows]
            else:
                no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                break
        
        return best_rows, best_scope

    # Strategy:
    # 1. Generate multiple initial solutions with different seeds and randomization
    # 2. Compress each one
    # 3. Run local search on the best ones
    
    initial_candidates = []
    
    # Generate diverse initial solutions
    for i in range(15):
        randomize = (i % 3 != 0)  # Mix of deterministic and randomized
        rows, used = _greedy_construct(i * 137, randomize)
        scope = get_scope(rows)
        initial_candidates.append((scope, rows, used))
    
    # Sort by scope
    initial_candidates.sort(key=lambda x: x[0])
    
    # Compress the top candidates
    compressed_candidates = []
    for scope, rows, used in initial_candidates[:8]:
        deadline = time.time() + (time_limit * 0.15)
        if time.time() > time_limit * 0.5:
            break
        comp_rows, comp_scope = _compress_rows(rows, deadline)
        if comp_scope < scope:
            compressed_candidates.append((comp_scope, comp_rows))
        else:
            compressed_candidates.append((scope, rows))
    
    # Sort compressed by scope
    compressed_candidates.sort(key=lambda x: x[0])
    
    # Run local search on the best compressed candidates
    for scope, rows in compressed_candidates[:5]:
        if time.time() > time_limit * 0.85:
            break
        deadline = time.time() + (time_limit * 0.1)
        local_search(rows, set(), deadline)
    
    if best_rows is None:
        return initial_candidates[0][1]
    
    return best_rows
