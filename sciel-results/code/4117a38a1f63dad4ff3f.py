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

def _diffs_for_row(row):
    diffs = []
    for i in range(len(row)):
        for j in range(i + 1, len(row)):
            diffs.append(row[j] - row[i])
    return diffs

def _build_all_diffs(rows):
    all_diffs = []
    for r in rows:
        all_diffs.extend(_diffs_for_row(r))
    return all_diffs

def _sa_optimize(rows, k, deadline, rng, max_iters=50000):
    """
    Simulated Annealing to minimize the scope.
    """
    n_rows = len(rows)
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    
    # Track all differences for quick collision checking
    all_diffs = _build_all_diffs(current)
    diff_count = Counter(all_diffs)
    collisions = sum(max(0, c - 1) for c in diff_count.values())
    
    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = collisions
    
    T = 50.0
    cooling = 0.999
    min_T = 0.05
    
    # Track which entries are at the current scope (candidates for reduction)
    scope_entries = set()
    for i, r in enumerate(current):
        for j, v in enumerate(r):
            if v == current_scope:
                scope_entries.add((i, j))
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Choose a random row and position to modify
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(1, k)
        
        old_val = current[i][j]
        prev_val = current[i][j-1]
        next_val = current[i][j+1] if j < k else old_val + 100
        
        # Determine move type
        move_type = rng.random()
        
        if move_type < 0.3 and (i, j) in scope_entries:
            # Scope compression: try to reduce a scope entry
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val - 1))
            if new_val >= old_val:
                new_val = old_val - 1
            if new_val <= prev_val:
                continue
        elif move_type < 0.7:
            # Small perturbation
            delta = rng.randint(-3, 3)
            new_val = old_val + delta
        else:
            # Larger perturbation
            new_val = rng.randint(prev_val + 1, max(prev_val + 1, old_val + 15))
        
        if new_val <= prev_val or (j < k and new_val >= next_val):
            continue
        
        # Compute the difference changes
        old_diffs = _diffs_for_row(current[i])
        
        # Update the row temporarily
        current[i][j] = new_val
        new_row = current[i]
        new_diffs = _diffs_for_row(new_row)
        
        # Check validity of new row
        valid_row = len(set(new_diffs)) == len(new_diffs) and all(d > 0 for d in new_diffs)
        
        if not valid_row:
            current[i][j] = old_val
            continue
        
        # Compute scope change
        new_scope = max(_scope(current), max(new_row))
        
        # Compute collision change
        # Remove old diffs from count
        for d in old_diffs:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        
        # Add new diffs
        for d in new_diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
        
        new_collisions = sum(max(0, c - 1) for c in diff_count.values())
        
        # Cost function: heavily weight scope reduction, especially when close to optimum
        # Use a softer cost that becomes more aggressive as scope decreases
        scope_penalty = 5000 if current_scope > 120 else 3000 if current_scope > 110 else 2000
        old_cost = current_scope + scope_penalty * collisions
        new_cost = new_scope + scope_penalty * new_collisions
        
        # Accept if better or with probability based on temperature
        if new_cost < old_cost:
            accept = True
        else:
            delta_cost = new_cost - old_cost
            if delta_cost < 0:
                accept = True
            else:
                accept_prob = math.exp(-delta_cost / max(T, min_T))
                accept = rng.random() < accept_prob
        
        if accept:
            # Update scope entries
            if old_val == current_scope:
                scope_entries.discard((i, j))
            if new_val == new_scope:
                scope_entries.add((i, j))
            # Also update if other entries were at scope
            for idx, r in enumerate(current):
                for jj, v in enumerate(r):
                    if v == new_scope and v != current_scope:
                        scope_entries.add((idx, jj))
                    elif v == current_scope and v != new_scope:
                        scope_entries.discard((idx, jj))
            
            current_scope = new_scope
            collisions = new_collisions
            if new_scope < best_scope or (new_scope == best_scope and new_collisions < best_collisions):
                best_scope = new_scope
                best_collisions = new_collisions
                best = [list(r) for r in current]
        else:
            # Revert
            current[i][j] = old_val
            # Restore diff counts
            for d in new_diffs:
                diff_count[d] -= 1
                if diff_count[d] == 0:
                    del diff_count[d]
            for d in old_diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
        
        T = max(T * cooling, min_T)
    
    return best_scope, best

def construct(n_rows, k):
    deadline = time.time() + 140.0  # Leave 10s margin
    rng = random.Random(123)
    
    # Generate multiple initial candidates
    candidates = []
    
    # Standard greedy
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    
    # Spread greedy
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    # Randomized greedy with different parameters
    for seed in range(5):
        r = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Try random jumps
                max_step = r.randint(1, 10)
                found = False
                for _ in range(max_step):
                    x += r.randint(1, 3)
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
                    # Fallback: find next valid
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
    
    # Sort by scope
    candidates.sort(key=_scope)
    
    # Run SA on the best few candidates
    best_result = None
    best_score = float('inf')
    
    for idx, cand in enumerate(candidates[:3]):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        # Allocate more time to earlier candidates
        sa_deadline = time.time() + remaining * (0.5 if idx == 0 else 0.3)
        s, rows = _sa_optimize(cand, k, sa_deadline, rng, max_iters=30000)
        
        # Check if valid
        if _valid(rows, k):
            if s < best_score:
                best_score = s
                best_result = rows
    
    # If time remains, continue optimizing the best result with more iterations
    if best_result is not None and time.time() < deadline - 10:
        remaining = deadline - time.time()
        # Use a tighter cooling schedule for the final polish
        s, rows = _sa_optimize(best_result, k, deadline - 5, rng, max_iters=int(remaining * 30000))
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows
        
        # Final local polish: try small adjustments to reduce scope
        if time.time() < deadline - 3:
            final_rows = [list(r) for r in best_result]
            final_scope = _scope(final_rows)
            polish_deadline = deadline - 2
            for _ in range(1000):
                if time.time() > polish_deadline:
                    break
                # Try to reduce the maximum entry by 1
                max_val = final_scope
                for i, r in enumerate(final_rows):
                    for j, v in enumerate(r):
                        if v == max_val and j > 0:
                            # Try to decrease this value
                            new_val = v - 1
                            if new_val > r[j-1]:
                                # Check if this creates a conflict
                                old_diffs = _diffs_for_row(r)
                                r[j] = new_val
                                new_diffs = _diffs_for_row(r)
                                if len(set(new_diffs)) == len(new_diffs):
                                    # Check global conflicts
                                    all_diffs = _build_all_diffs(final_rows)
                                    if len(set(all_diffs)) == len(all_diffs):
                                        final_scope -= 1
                                        break
                                r[j] = v
                    if time.time() > polish_deadline:
                        break
                else:
                    continue
                break
            if final_scope < best_score:
                best_score = final_scope
                best_result = final_rows
    
    if best_result is not None:
        return best_result
    
    # Fallback: return the best greedy candidate
    return candidates[0]
