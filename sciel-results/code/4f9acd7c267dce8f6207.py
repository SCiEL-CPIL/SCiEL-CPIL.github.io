import time
import random
import math
from collections import Counter

def construct(n_rows, k):
    deadline = time.time() + 148.0
    rng = random.Random(12345)
    
    # For the specific problem: n_rows=7, k=5
    # Each row has k+1 = 6 elements
    # Total diffs per row: C(6,2) = 15
    # Total diffs across all rows: 7 * 15 = 105
    
    def _row_diffs(row):
        n = len(row)
        diffs = []
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(row[j] - row[i])
        return diffs
    
    def _is_valid(rows, k):
        seen = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                if i > 0 and row[i] <= row[i - 1]:
                    return False
            for d in _row_diffs(row):
                if d in seen:
                    return False
                seen.add(d)
        return True
    
    def _get_scope(rows):
        if not rows:
            return 0
        return max(max(row) for row in rows)
    
    def _diff_counts(rows):
        counts = Counter()
        for r in rows:
            for d in _row_diffs(r):
                counts[d] += 1
        return counts
    
    def _build_row_greedy(used_diffs, k, max_scope, rng, randomize=0.0):
        """Build a single row greedily, avoiding used_diffs."""
        row = [0]
        current = 0
        for _ in range(k):
            found = False
            # If randomize > 0, sometimes skip ahead
            if randomize > 0 and rng.random() < randomize:
                start = current + rng.randint(2, 5)
            else:
                start = current + 1
            
            for val in range(start, max_scope + 1):
                new_diffs = [val - a for a in row]
                if not any(d in used_diffs for d in new_diffs):
                    row.append(val)
                    used_diffs.update(new_diffs)
                    current = val
                    found = True
                    break
            
            if not found:
                return None
        return row
    
    def _remove_row_from_counts(counts, row):
        for d in _row_diffs(row):
            counts[d] -= 1
            if counts[d] == 0:
                del counts[d]
    
    def _add_row_to_counts(counts, row):
        for d in _row_diffs(row):
            counts[d] = counts.get(d, 0) + 1
    
    def _conflict_score(rows, counts):
        """Total number of excess occurrences (collisions)."""
        return sum(max(0, c - 1) for c in counts.values())
    
    def _objective(rows, counts):
        scope = _get_scope(rows)
        conflicts = _conflict_score(rows, counts)
        return scope + 1000 * conflicts
    
    def _sa_optimize(rows, n_rows, k, deadline, rng):
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_scope = _get_scope(rows)
        
        counts = _diff_counts(rows)
        current_cost = _objective(rows, counts)
        best_cost = current_cost
        
        T = 5.0
        cooling = 0.9997
        min_T = 0.02
        no_improve = 0
        max_no_improve = 800
        
        while T > min_T and time.time() < deadline - 15:
            r = rng.random()
            
            if r < 0.35:
                # Rebuild a random row
                i = rng.randint(0, n_rows - 1)
                old_row = list(rows[i])
                _remove_row_from_counts(counts, old_row)
                
                # Try to rebuild with some randomization
                max_scope = best_scope + 3
                new_row = None
                for attempt in range(8):
                    rand_level = 0.0 if attempt == 0 else min(0.3, 0.05 * attempt)
                    candidate = _build_row_greedy(set(), k, max_scope, rng, rand_level)
                    if candidate is not None:
                        # Check conflicts
                        cand_diffs = _row_diffs(candidate)
                        conflicts = sum(1 for d in cand_diffs if counts.get(d, 0) > 0)
                        if conflicts == 0:
                            new_row = candidate
                            break
                        elif new_row is None:
                            new_row = candidate
                
                if new_row is not None:
                    _add_row_to_counts(counts, new_row)
                    rows[i] = new_row
                    new_cost = _objective(rows, counts)
                    
                    if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                        current_cost = new_cost
                        new_scope = _get_scope(rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in rows]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        _remove_row_from_counts(counts, new_row)
                        _add_row_to_counts(counts, old_row)
                        rows[i] = old_row
                        no_improve += 1
                else:
                    _add_row_to_counts(counts, old_row)
                    no_improve += 1
            
            elif r < 0.55:
                # Try to reduce the maximum element
                max_val = best_scope
                target_row_idx = -1
                for ri in range(n_rows):
                    if rows[ri][-1] == max_val:
                        target_row_idx = ri
                        break
                
                if target_row_idx < 0:
                    no_improve += 1
                else:
                    i = target_row_idx
                    old_row = list(rows[i])
                    _remove_row_from_counts(counts, old_row)
                    
                    # Try to rebuild with a tighter scope
                    new_row = _build_row_greedy(set(), k, max_val - 1, rng, 0.1)
                    
                    if new_row is not None:
                        cand_diffs = _row_diffs(new_row)
                        conflicts = sum(1 for d in cand_diffs if counts.get(d, 0) > 0)
                        if conflicts == 0:
                            _add_row_to_counts(counts, new_row)
                            rows[i] = new_row
                            new_cost = _objective(rows, counts)
                            if new_cost < current_cost:
                                current_cost = new_cost
                                new_scope = _get_scope(rows)
                                if new_scope < best_scope:
                                    best_scope = new_scope
                                    best_rows = [list(r) for r in rows]
                                    no_improve = 0
                                else:
                                    no_improve += 1
                            else:
                                _remove_row_from_counts(counts, new_row)
                                _add_row_to_counts(counts, old_row)
                                rows[i] = old_row
                                no_improve += 1
                        else:
                            _add_row_to_counts(counts, old_row)
                            no_improve += 1
                    else:
                        _add_row_to_counts(counts, old_row)
                        no_improve += 1
            
            elif r < 0.80:
                # Single element move: change one position in one row
                r_idx = rng.randint(0, n_rows - 1)
                row = rows[r_idx]
                pos = rng.randint(1, len(row) - 1)
                old_val = row[pos]
                
                # Generate candidate: try nearby values
                candidates = []
                for delta in range(-8, 9):
                    if delta == 0:
                        continue
                    nv = old_val + delta
                    if nv <= 0:
                        continue
                    if pos > 0 and nv <= row[pos - 1]:
                        continue
                    if pos < len(row) - 1 and nv >= row[pos + 1]:
                        continue
                    candidates.append(nv)
                
                if candidates:
                    new_val = rng.choice(candidates)
                    _remove_row_from_counts(counts, row)
                    row[pos] = new_val
                    _add_row_to_counts(counts, row)
                    
                    new_cost = _objective(rows, counts)
                    if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                        current_cost = new_cost
                        new_scope = _get_scope(rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [list(r) for r in rows]
                            no_improve = 0
                        else:
                            no_improve += 1
                    else:
                        _remove_row_from_counts(counts, row)
                        row[pos] = old_val
                        _add_row_to_counts(counts, row)
                        no_improve += 1
                else:
                    no_improve += 1
            
            else:
                # Swap two elements between different rows (structural move)
                if n_rows >= 2:
                    i, j = rng.sample(range(n_rows), 2)
                    row_i = rows[i]
                    row_j = rows[j]
                    pos_i = rng.randint(1, len(row_i) - 1)
                    pos_j = rng.randint(1, len(row_j) - 1)
                    vi = row_i[pos_i]
                    vj = row_j[pos_j]
                    
                    # Try swapping
                    row_i[pos_i] = vj
                    row_j[pos_j] = vi
                    
                    # Check validity of both rows
                    valid = True
                    if row_i[0] != 0 or row_j[0] != 0:
                        valid = False
                    for r in [row_i, row_j]:
                        for idx in range(1, len(r)):
                            if r[idx] <= r[idx - 1]:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        _remove_row_from_counts(counts, rows[i])
                        _remove_row_from_counts(counts, rows[j])
                        _add_row_to_counts(counts, rows[i])
                        _add_row_to_counts(counts, rows[j])
                        
                        new_cost = _objective(rows, counts)
                        if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                            current_cost = new_cost
                            new_scope = _get_scope(rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best_rows = [list(r) for r in rows]
                                no_improve = 0
                            else:
                                no_improve += 1
                        else:
                            # Revert
                            row_i[pos_i] = vi
                            row_j[pos_j] = vj
                            _remove_row_from_counts(counts, rows[i])
                            _remove_row_from_counts(counts, rows[j])
                            _add_row_to_counts(counts, rows[i])
                            _add_row_to_counts(counts, rows[j])
                            no_improve += 1
                    else:
                        # Revert
                        row_i[pos_i] = vi
                        row_j[pos_j] = vj
                        no_improve += 1
                else:
                    no_improve += 1
            
            T *= cooling
            
            if no_improve > max_no_improve:
                break
        
        return best_rows
    
    def _local_polish(rows, k, deadline, rng):
        """Fine-grained local search: try each element with small deltas."""
        rows = [list(r) for r in rows]
        best_rows = [list(r) for r in rows]
        best_scope = _get_scope(rows)
        
        counts = _diff_counts(rows)
        current_cost = _objective(rows, counts)
        
        T = 3.0
        cooling = 0.99985
        min_T = 0.01
        no_improve = 0
        max_no_improve = 400
        
        while T > min_T and time.time() < deadline - 5:
            r_idx = rng.randint(0, len(rows) - 1)
            row = rows[r_idx]
            pos = rng.randint(1, len(row) - 1)
            old_val = row[pos]
            
            improved = False
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_val = old_val + delta
                if new_val <= 0:
                    continue
                if pos > 0 and new_val <= row[pos - 1]:
                    continue
                if pos < len(row) - 1 and new_val >= row[pos + 1]:
                    continue
                
                _remove_row_from_counts(counts, row)
                row[pos] = new_val
                _add_row_to_counts(counts, row)
                
                new_cost = _objective(rows, counts)
                if new_cost < current_cost or rng.random() < math.exp(-(new_cost - current_cost) / max(T, min_T)):
                    current_cost = new_cost
                    new_scope = _get_scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                        no_improve = 0
                    else:
                        no_improve += 1
                    improved = True
                    break
                else:
                    _remove_row_from_counts(counts, row)
                    row[pos] = old_val
                    _add_row_to_counts(counts, row)
            
            if not improved:
                no_improve += 1
            
            T *= cooling
            
            if no_improve > max_no_improve:
                break
        
        return best_rows
    
    # Main construction
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multiple greedy inits with varying randomization
    for attempt in range(12):
        if time.time() > deadline - 120:
            break
        
        rand_prob = 0.0 if attempt == 0 else min(0.25, 0.02 * attempt)
        rows = []
        used_diffs = set()
        success = True
        
        for ri in range(n_rows):
            row = _build_row_greedy(used_diffs, k, 200, rng, rand_prob)
            if row is None:
                success = False
                break
            rows.append(row)
        
        if success and _is_valid(rows, k):
            scope = _get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
    
    if best_rows is None:
        # Fallback: pure greedy
        rows = []
        used_diffs = set()
        for ri in range(n_rows):
            row = _build_row_greedy(used_diffs, k, 300, rng, 0.0)
            rows.append(row)
        return rows
    
    # Phase 2: SA optimization
    rows_sa = _sa_optimize(best_rows, n_rows, k, deadline, rng)
    if _is_valid(rows_sa, k):
        scope_sa = _get_scope(rows_sa)
        if scope_sa < best_scope:
            best_scope = scope_sa
            best_rows = rows_sa
    
    # Phase 3: Local polish
    if time.time() < deadline - 10:
        polished = _local_polish(best_rows, k, deadline, rng)
        if _is_valid(polished, k):
            scope_pol = _get_scope(polished)
            if scope_pol < best_scope:
                best_scope = scope_pol
                best_rows = polished
    
    # Phase 4: Another round of SA from the best
    if time.time() < deadline - 20:
        rows_sa2 = _sa_optimize(best_rows, n_rows, k, deadline, rng)
        if _is_valid(rows_sa2, k):
            scope_sa2 = _get_scope(rows_sa2)
            if scope_sa2 < best_scope:
                best_scope = scope_sa2
                best_rows = rows_sa2
    
    # Phase 5: Final polish
    if time.time() < deadline - 5:
        polished2 = _local_polish(best_rows, k, deadline, rng)
        if _is_valid(polished2, k):
            scope_pol2 = _get_scope(polished2)
            if scope_pol2 < best_scope:
                best_scope = scope_pol2
                best_rows = polished2
    
    return best_rows
