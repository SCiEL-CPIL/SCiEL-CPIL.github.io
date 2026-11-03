import time
import random
import math
from collections import Counter

def _scope(rows):
    return max(a for r in rows for a in r[1:])

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

def _diffs_for_row(row):
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

def _build_diff_counts(rows):
    counts = Counter()
    for r in rows:
        for d in _diffs_for_row(r):
            counts[d] += 1
    return counts

def _fast_greedy(k, n_rows, rng, spread=False):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            conflict = False
            for a in row:
                if (x - a) in used:
                    conflict = True
                    break
            if not conflict:
                row.append(x)
                for a in row[:-1]:
                    used.add(x - a)
                if spread:
                    x += max(1, (k - len(row) + 1) * 2 - 2)
        rows.append(row)
    return rows

def _sa_optimize(rows, k, deadline, rng, max_iters=100000):
    n_rows = len(rows)
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    
    diff_counts = _build_diff_counts(current)
    collisions = sum(max(0, c - 1) for c in diff_counts.values())
    
    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = collisions
    
    T = 100.0
    cooling = 0.9995
    min_T = 0.1
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = current[i][j]
        prev_val = current[i][j-1]
        next_val = current[i][j+1] if j < k else old_val + 1000
        
        move_type = rng.random()
        
        if move_type < 0.4:
            # Compression: try to reduce
            low = prev_val + 1
            high = max(low, old_val - 1)
            if high <= low:
                continue
            new_val = rng.randint(low, high)
        elif move_type < 0.8:
            # Small perturbation
            delta = rng.randint(-5, 5)
            new_val = old_val + delta
        else:
            # Larger perturbation
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val + 20))
        
        if new_val <= prev_val or (j < k and new_val >= next_val):
            continue
        if new_val == old_val:
            continue
        
        # Remove old diffs
        old_diffs = _diffs_for_row(current[i])
        for d in old_diffs:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
        
        # Apply change
        current[i][j] = new_val
        new_diffs = _diffs_for_row(current[i])
        
        # Check row validity (no duplicate diffs within row)
        if len(set(new_diffs)) != len(new_diffs):
            # Revert
            current[i][j] = old_val
            for d in old_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            continue
        
        # Add new diffs
        for d in new_diffs:
            diff_counts[d] = diff_counts.get(d, 0) + 1
        
        new_scope = max(current_scope, current[i][-1])
        new_collisions = sum(max(0, c - 1) for c in diff_counts.values())
        
        # Cost function: prioritize eliminating collisions, then reducing scope
        weight = 10000
        old_cost = current_scope + weight * collisions
        new_cost = new_scope + weight * new_collisions
        
        if new_cost <= old_cost:
            accept = True
        else:
            delta_cost = new_cost - old_cost
            accept_prob = math.exp(-delta_cost / max(T, min_T))
            accept = rng.random() < accept_prob
        
        if accept:
            current_scope = new_scope
            collisions = new_collisions
            if new_scope < best_scope or (new_scope == best_scope and new_collisions < best_collisions):
                best_scope = new_scope
                best_collisions = new_collisions
                best = [list(r) for r in current]
        else:
            # Revert
            for d in new_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            current[i][j] = old_val
            for d in old_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
        
        T = max(T * cooling, min_T)
    
    return best_scope, best

def _diff_set_to_row(diff_set, k, deadline):
    """
    Try to reconstruct a valid row from a set of C(k+1,2) differences.
    Uses backtracking with constraint propagation.
    """
    n = k + 1
    diffs = sorted(diff_set)
    max_diff = diffs[-1]
    
    # The row must end at max_diff
    # row[0] = 0, row[k] = max_diff
    # We need to find row[1], ..., row[k-1]
    
    best = None
    best_scope = float('inf')
    
    # a1 must be in diff_set and small
    a1_candidates = [d for d in diffs if d <= 25]
    
    for a1 in a1_candidates:
        if time.time() > deadline:
            break
        
        def rec(pos, current, remaining, dl):
            nonlocal best, best_scope
            if time.time() > dl:
                return False
            
            if pos == k - 1:
                # Place the last element: must be max_diff
                full_row = current + [max_diff]
                row_diffs = set()
                valid = True
                for i in range(n):
                    for j in range(i + 1, n):
                        d = full_row[j] - full_row[i]
                        if d not in diff_set:
                            valid = False
                            break
                        row_diffs.add(d)
                    if not valid:
                        break
                if valid and row_diffs == diff_set:
                    if max_diff < best_scope:
                        best_scope = max_diff
                        best = list(full_row)
                    return True
                return False
            
            # Place element at position pos
            prev = current[-1]
            upper = max_diff - 1
            
            # Find candidates: values v such that v - current[i] is in remaining for all i
            candidates = set()
            for d in remaining:
                for i in range(len(current)):
                    val = current[i] + d
                    if prev < val < max_diff:
                        candidates.add(val)
            
            candidates = sorted(candidates)
            
            for val in candidates:
                new_diffs = []
                valid = True
                for i in range(len(current)):
                    d = val - current[i]
                    if d not in remaining:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if not valid:
                    continue
                if len(set(new_diffs)) != len(new_diffs):
                    continue
                
                new_remaining = set(remaining)
                for d in new_diffs:
                    new_remaining.discard(d)
                
                if rec(pos + 1, current + [val], new_remaining, dl):
                    return True
            
            return False
        
        if rec(1, [0, a1], set(diff_set) - {a1}, deadline):
            if best is not None:
                break
    
    return best

def construct(n_rows, k):
    deadline = time.time() + 140.0
    rng = random.Random(123)
    
    diffs_per_row = k * (k + 1) // 2
    
    # Phase 1: Generate initial candidates
    candidates = []
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    # Randomized greedy with different seeds
    for seed in range(8):
        r = random.Random(seed * 31 + 7)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try a few random steps
                found = False
                for _ in range(20):
                    x += r.randint(1, 4)
                    conflict = False
                    for a in row:
                        if (x - a) in used:
                            conflict = True
                            break
                    if not conflict:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                        found = True
                        break
                if not found:
                    while True:
                        x += 1
                        conflict = False
                        for a in row:
                            if (x - a) in used:
                                conflict = True
                                break
                        if not conflict:
                            row.append(x)
                            for a in row[:-1]:
                                used.add(x - a)
                            break
            rows.append(row)
        candidates.append(rows)
    
    candidates.sort(key=_scope)
    
    best_result = None
    best_score = float('inf')
    
    # Phase 2: SA optimization on best candidates
    for idx, cand in enumerate(candidates[:4]):
        if time.time() > deadline - 10:
            break
        remaining = deadline - time.time() - 10
        sa_deadline = time.time() + remaining * (0.35 if idx == 0 else 0.2)
        s, rows = _sa_optimize(cand, k, sa_deadline, rng, max_iters=50000)
        
        if _valid(rows, k):
            if s < best_score:
                best_score = s
                best_result = rows
    
    # Phase 3: Try difference set approach for lower scope
    if time.time() < deadline - 40:
        for target_scope in [105, 106, 108, 110, 112]:
            if time.time() > deadline - 15:
                break
            
            success = False
            for attempt in range(30):
                if time.time() > deadline - 15:
                    break
                
                r2 = random.Random(attempt * 7919 + target_scope)
                used = set()
                diff_sets = []
                ok = True
                
                for row_idx in range(n_rows):
                    available = [d for d in range(1, target_scope + 1) if d not in used]
                    if len(available) < diffs_per_row:
                        ok = False
                        break
                    
                    # Weighted sampling: prefer smaller differences
                    weights = [1.0 / d for d in available]
                    total_w = sum(weights)
                    
                    selected = set()
                    for _ in range(diffs_per_row):
                        r_val = r2.random() * total_w
                        cum = 0
                        chosen = None
                        for d in available:
                            if d in selected:
                                continue
                            cum += 1.0 / d
                            if r_val <= cum:
                                chosen = d
                                break
                        if chosen is None:
                            chosen = available[0]
                        selected.add(chosen)
                        total_w -= 1.0 / chosen
                    
                    diff_sets.append(selected)
                    used.update(selected)
                
                if not ok or len(diff_sets) != n_rows:
                    continue
                
                # Reconstruct rows
                rows = []
                all_valid = True
                for ds in diff_sets:
                    if time.time() > deadline - 15:
                        all_valid = False
                        break
                    row = _diff_set_to_row(ds, k, deadline - 10)
                    if row is None:
                        all_valid = False
                        break
                    rows.append(row)
                
                if all_valid and _valid(rows, k):
                    s = _scope(rows)
                    if s < best_score:
                        best_score = s
                        best_result = rows
                        success = True
                        break
            
            if success:
                break
    
    # Phase 4: Final SA polish
    if best_result is not None and time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        s, rows = _sa_optimize(best_result, k, deadline - 5, rng, max_iters=int(remaining * 30000))
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows
    
    # Phase 5: Local polish
    if best_result is not None and time.time() < deadline - 5:
        final_rows = [list(r) for r in best_result]
        final_scope = _scope(final_rows)
        polish_deadline = deadline - 2
        
        for _ in range(200):
            if time.time() > polish_deadline:
                break
            
            reduced = False
            for i, r in enumerate(final_rows):
                for j in range(len(r) - 1, 0, -1):
                    if r[j] == final_scope:
                        new_val = r[j] - 1
                        if new_val > r[j-1]:
                            old_diffs = _diffs_for_row(r)
                            r[j] = new_val
                            new_diffs = _diffs_for_row(r)
                            if len(set(new_diffs)) == len(new_diffs):
                                # Check global validity
                                all_diffs = []
                                for rr in final_rows:
                                    all_diffs.extend(_diffs_for_row(rr))
                                if len(set(all_diffs)) == len(all_diffs):
                                    final_scope -= 1
                                    reduced = True
                                    break
                            r[j] = new_val + 1
                if reduced:
                    break
            
            if not reduced:
                break
        
        if final_scope < best_score:
            best_score = final_scope
            best_result = final_rows
    
    if best_result is not None:
        return best_result
    
    return candidates[0] if candidates else _fast_greedy(k, n_rows, rng)
