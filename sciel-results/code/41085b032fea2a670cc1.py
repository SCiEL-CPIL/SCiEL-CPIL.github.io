import time
import random
import math
from collections import Counter, defaultdict
from itertools import combinations

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

def _reconstruct_from_diffs(diff_set, k, deadline):
    """
    Given a set of C(k+1,2) differences, try to reconstruct a valid row.
    The row has k+1 elements starting at 0, strictly increasing.
    The differences are all pairwise differences.
    
    We use backtracking: place elements one by one, checking consistency.
    """
    n = k + 1  # number of elements
    diffs = sorted(diff_set)
    total_diffs = len(diffs)
    if total_diffs != n * (n - 1) // 2:
        return None
    
    # The largest difference must be the last element (since row starts at 0)
    # So row[-1] = max(diffs)
    # The smallest differences must include row[1] - row[0] = row[1]
    
    # Strategy: use backtracking with constraint propagation
    # row[0] = 0 always
    # We need to find row[1], row[2], ..., row[k]
    
    best = None
    rng = random.Random(42)
    
    def backtrack(pos, current_row, remaining_diffs, deadline_check):
        nonlocal best
        if time.time() > deadline_check:
            return False
        if pos == n:
            # Verify all differences match
            row_diffs = set()
            for i in range(n):
                for j in range(i + 1, n):
                    row_diffs.add(current_row[j] - current_row[i])
            if row_diffs == diff_set:
                if best is None or _scope([current_row]) < _scope([best]):
                    best = list(current_row)
                return True
            return False
        
        # Try to place element at position pos
        # The minimum possible value is current_row[pos-1] + 1
        # The maximum is constrained by remaining differences
        
        if pos == 0:
            candidates = [0]
        else:
            prev = current_row[pos - 1]
            # Try values from prev+1 upward, but bounded
            # The new element must create differences with all previous elements
            # that are in diff_set
            max_val = prev + 100  # reasonable upper bound
            for val in range(prev + 1, min(max_val, prev + 50)):
                # Check if all differences (val - current_row[i]) for i < pos are in diff_set
                valid = True
                new_diffs = []
                for i in range(pos):
                    d = val - current_row[i]
                    if d not in remaining_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    # Check these diffs aren't already used (they should be in remaining)
                    current_row.append(val)
                    new_remaining = set(remaining_diffs)
                    for d in new_diffs:
                        new_remaining.discard(d)
                    if backtrack(pos + 1, current_row, new_remaining, deadline_check):
                        current_row.pop()
                        return True
                    current_row.pop()
        
        # Also try larger jumps
        if pos > 0:
            prev = current_row[pos - 1]
            for val in [prev + 50, prev + 100, prev + 200]:
                valid = True
                new_diffs = []
                for i in range(pos):
                    d = val - current_row[i]
                    if d not in remaining_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                if valid:
                    current_row.append(val)
                    new_remaining = set(remaining_diffs)
                    for d in new_diffs:
                        new_remaining.discard(d)
                    if backtrack(pos + 1, current_row, new_remaining, deadline_check):
                        current_row.pop()
                        return True
                    current_row.pop()
        
        return False
    
    result = backtrack(0, [], set(diff_set), deadline)
    return best


def _reconstruct_smart(diff_set, k, deadline):
    """
    Smarter reconstruction using the structure of difference sets.
    Key insight: if we know the sorted differences, we can use them to
    constrain the row values more tightly.
    """
    n = k + 1
    diffs = sorted(diff_set)
    max_diff = diffs[-1]
    
    # The row must end at max_diff (since row[0]=0, row[-1] is the max element)
    # So row[k] = max_diff
    
    # Use a different approach: work backwards from the end
    # Or use the fact that the smallest differences give us early elements
    
    # Let's try: the row is [0, a1, a2, ..., ak] where ak = max_diff
    # Differences include: a1, a2, ..., ak, a2-a1, a3-a1, ..., ak-a1, a3-a2, ..., ak-a2, ...
    
    # Strategy: enumerate possible values for a1 (smallest non-zero element)
    # a1 must be in diff_set
    # Then a2-a1 must be in diff_set, etc.
    
    best = None
    best_scope = float('inf')
    
    # a1 candidates: small values in diff_set
    a1_candidates = [d for d in diffs if d <= 30]
    
    for a1 in a1_candidates:
        if time.time() > deadline:
            break
        # Now we need a2, a3, ..., a_{k-1} such that all differences work
        # a_k = max_diff
        
        def rec(pos, current, remaining, dl):
            nonlocal best, best_scope
            if time.time() > dl:
                return False
            if pos == k:
                # Check: current should be [0, a1, a2, ..., a_{k-1}]
                # a_k = max_diff
                full_row = current + [max_diff]
                # Verify
                row_diffs = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        d = full_row[j] - full_row[i]
                        if d not in diff_set:
                            return False
                        row_diffs.add(d)
                if row_diffs == diff_set:
                    scope = max_diff
                    if scope < best_scope:
                        best_scope = scope
                        best = list(full_row)
                    return True
                return False
            
            # Place element at position pos (which is a_{pos})
            prev = current[-1]
            # The new element must be > prev
            # Differences with all previous elements must be in remaining
            # Also, the new element must be < max_diff (unless pos == k-1... but we handle k separately)
            
            # Upper bound: max_diff - 1 (since we still need room for max_diff)
            # Actually, the new element can be up to max_diff - 1
            upper = max_diff - 1 if pos < k - 1 else max_diff
            
            # Try candidate values
            # The value must create differences with all previous elements that are in remaining
            candidates = set()
            for d in remaining:
                # If new_val - current[i] = d for some i, then new_val = current[i] + d
                for i in range(len(current)):
                    val = current[i] + d
                    if prev < val <= upper:
                        candidates.add(val)
            
            # Sort candidates for deterministic ordering
            candidates = sorted(candidates)
            
            for val in candidates:
                # Check all differences
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
                
                # Also check no duplicates in new_diffs (shouldn't happen if remaining is a set)
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


def _find_diff_sets(n_rows, k, max_scope, deadline, rng):
    """
    Find n_rows disjoint sets of C(k+1,2) differences each, all within [1, max_scope].
    Use a structured approach.
    """
    diffs_per_row = k * (k + 1) // 2
    total_needed = n_rows * diffs_per_row
    
    # Available differences: 1 to max_scope
    # We need to partition a subset of [1, max_scope] into n_rows sets of size diffs_per_row
    
    # Strategy: use a greedy + local search approach on the difference sets
    # Each "row" in difference-space is a set of diffs_per_row values
    
    # First, generate candidate difference sets
    # A valid difference set for a row of length k+1 must satisfy:
    # - It contains C(k+1,2) distinct positive integers
    # - The largest is the "scope" of that row
    # - There must exist a valid row realizing these differences
    
    # For now, just find disjoint sets, then try to reconstruct
    
    used = set()
    diff_sets = []
    
    for row_idx in range(n_rows):
        if time.time() > deadline:
            break
        # Try to find a set of diffs_per_row unused differences
        # Bias toward smaller differences to keep scope low
        target_max = max_scope - row_idx * 2  # slightly tighter for later rows
        
        attempts = 0
        while attempts < 100:
            if time.time() > deadline:
                break
            attempts += 1
            
            # Randomly sample diffs_per_row values from [1, target_max] not in used
            available = [d for d in range(1, target_max + 1) if d not in used]
            if len(available) < diffs_per_row:
                break
            
            # Random sample
            sample = rng.sample(available, diffs_per_row)
            
            # Check if this could be a valid difference set
            # Basic check: the sum of differences should be consistent
            # For a row [0, a1, ..., ak], the sum of all differences is:
            # sum_{i<j} (a_j - a_i) = sum_j a_j * j - sum_i a_i * (n-1-i)
            # This is complex; let's just try to reconstruct
            
            # Quick filter: the largest difference should be reasonable
            max_d = max(sample)
            if max_d > max_scope:
                continue
            
            diff_sets.append(sample)
            used.update(sample)
            break
        
        if len(diff_sets) < n_rows:
            # Fallback: just take the next available
            for d in range(1, max_scope + 1):
                if d not in used and len(diff_sets[-1] if diff_sets else []) < diffs_per_row:
                    if diff_sets:
                        diff_sets[-1].append(d)
                        used.add(d)
                    else:
                        diff_sets.append([d])
                        used.add(d)
                if diff_sets and len(diff_sets[-1]) == diffs_per_row:
                    break
                if len(diff_sets) >= n_rows:
                    break
    
    return diff_sets


def _sa_optimize(rows, k, deadline, rng, max_iters=50000):
    """
    Simulated Annealing to minimize the scope.
    """
    n_rows = len(rows)
    current = [list(r) for r in rows]
    current_scope = _scope(current)
    
    # Track all differences for quick collision checking
    all_diffs = []
    for r in current:
        all_diffs.extend(_diffs_for_row(r))
    diff_count = Counter(all_diffs)
    collisions = sum(max(0, c - 1) for c in diff_count.values())
    
    best = [list(r) for r in current]
    best_scope = current_scope
    best_collisions = collisions
    
    T = 50.0
    cooling = 0.999
    min_T = 0.05
    
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
        
        if move_type < 0.3:
            # Scope compression: try to reduce a value
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
        new_scope = max(current_scope, max(new_row))
        
        # Compute collision change
        for d in old_diffs:
            diff_count[d] -= 1
            if diff_count[d] == 0:
                del diff_count[d]
        
        for d in new_diffs:
            diff_count[d] = diff_count.get(d, 0) + 1
        
        new_collisions = sum(max(0, c - 1) for c in diff_count.values())
        
        # Cost function
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
            current_scope = new_scope
            collisions = new_collisions
            if new_scope < best_scope or (new_scope == best_scope and new_collisions < best_collisions):
                best_scope = new_scope
                best_collisions = new_collisions
                best = [list(r) for r in current]
        else:
            # Revert
            current[i][j] = old_val
            for d in new_diffs:
                diff_count[d] -= 1
                if diff_count[d] == 0:
                    del diff_count[d]
            for d in old_diffs:
                diff_count[d] = diff_count.get(d, 0) + 1
        
        T = max(T * cooling, min_T)
    
    return best_scope, best


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


def construct(n_rows, k):
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    rng = random.Random(123)
    
    diffs_per_row = k * (k + 1) // 2
    
    # Phase 1: Generate initial candidates using greedy methods
    candidates = []
    
    # Standard greedy
    candidates.append(_fast_greedy(k, n_rows, rng, spread=False))
    
    # Spread greedy
    candidates.append(_fast_greedy(k, n_rows, rng, spread=True))
    
    # Randomized greedy
    for seed in range(5):
        r = random.Random(seed)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
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
    
    best_result = None
    best_score = float('inf')
    
    # Phase 2: Run SA on best candidates
    for idx, cand in enumerate(candidates[:3]):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        sa_deadline = time.time() + remaining * (0.4 if idx == 0 else 0.25)
        s, rows = _sa_optimize(cand, k, sa_deadline, rng, max_iters=30000)
        
        if _valid(rows, k):
            if s < best_score:
                best_score = s
                best_result = rows
    
    # Phase 3: Structural approach - find difference sets and reconstruct
    if time.time() < deadline - 30:
        # Try to find better solutions via difference set search
        for max_scope_attempt in [112, 110, 108, 106, 105]:
            if time.time() > deadline - 10:
                break
            
            # Try multiple random difference set configurations
            for attempt in range(20):
                if time.time() > deadline - 10:
                    break
                
                r2 = random.Random(attempt * 7919 + max_scope_attempt)
                used = set()
                diff_sets = []
                success = True
                
                for row_idx in range(n_rows):
                    # Find a set of diffs_per_row unused differences
                    available = [d for d in range(1, max_scope_attempt + 1) if d not in used]
                    if len(available) < diffs_per_row:
                        success = False
                        break
                    
                    # Use a structured selection: prefer differences that form valid patterns
                    # For a row [0, a1, ..., ak], the differences have structure
                    # Let's try to build incrementally
                    
                    # Simple approach: random sample with bias toward small values
                    # Weight by 1/d to prefer small differences
                    weights = [1.0 / d for d in available]
                    total_w = sum(weights)
                    
                    selected = set()
                    for _ in range(diffs_per_row):
                        # Weighted random selection
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
                
                if not success or len(diff_sets) != n_rows:
                    continue
                
                # Try to reconstruct each difference set into a row
                rows = []
                all_valid = True
                for ds in diff_sets:
                    if time.time() > deadline - 10:
                        all_valid = False
                        break
                    row = _reconstruct_smart(ds, k, deadline - 5)
                    if row is None:
                        all_valid = False
                        break
                    rows.append(row)
                
                if all_valid and _valid(rows, k):
                    s = _scope(rows)
                    if s < best_score:
                        best_score = s
                        best_result = rows
                        break
    
    # Phase 4: Final polish with SA
    if best_result is not None and time.time() < deadline - 15:
        remaining = deadline - time.time()
        s, rows = _sa_optimize(best_result, k, deadline - 5, rng, max_iters=int(remaining * 20000))
        if _valid(rows, k) and s < best_score:
            best_score = s
            best_result = rows
    
    # Phase 5: Local polish - try to reduce max entry by 1
    if best_result is not None and time.time() < deadline - 5:
        final_rows = [list(r) for r in best_result]
        final_scope = _scope(final_rows)
        polish_deadline = deadline - 2
        
        for _ in range(500):
            if time.time() > polish_deadline:
                break
            
            max_val = final_scope
            reduced = False
            for i, r in enumerate(final_rows):
                for j, v in enumerate(r):
                    if v == max_val and j > 0:
                        new_val = v - 1
                        if new_val > r[j-1]:
                            old_diffs = _diffs_for_row(r)
                            r[j] = new_val
                            new_diffs = _diffs_for_row(r)
                            if len(set(new_diffs)) == len(new_diffs):
                                # Check global
                                all_diffs = []
                                for rr in final_rows:
                                    all_diffs.extend(_diffs_for_row(rr))
                                if len(set(all_diffs)) == len(all_diffs):
                                    final_scope -= 1
                                    reduced = True
                                    break
                            r[j] = v
                if reduced:
                    break
            
            if not reduced:
                break
        
        if final_scope < best_score:
            best_score = final_scope
            best_result = final_rows
    
    if best_result is not None:
        return best_result
    
    # Fallback
    return candidates[0] if candidates else _fast_greedy(k, n_rows, rng)
