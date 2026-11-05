import time
import random
import math
import sys


def _row_diffs(row):
    """Return list of all positive differences within a row."""
    n = len(row)
    diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(row[j] - row[i])
    return diffs


def _row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            m |= (1 << (row[j] - row[i]))
    return m


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))


def _find_min_scope_row(forbidden_mask, num_elements, max_scope):
    """
    Find a row of num_elements increasing integers starting at 0
    with minimum scope, avoiding all differences in forbidden_mask.
    Uses backtracking with pruning.
    Returns (row, scope) or (None, None).
    """
    # We need C(num_elements, 2) distinct differences, all not in forbidden_mask
    # Try scopes from lower bound upward
    num_pairs = num_elements * (num_elements - 1) // 2
    
    # Lower bound: need at least num_pairs distinct positive integers not in forbidden
    # Simple lower bound: scope >= num_elements - 1 (trivial)
    # Better: the k-th smallest available difference must be <= scope
    lower = num_elements - 1
    
    for scope in range(lower, max_scope + 1):
        row = [0]
        result = _backtrack_row(row, forbidden_mask, num_elements, scope)
        if result is not None:
            return result, result[-1]
    
    return None, None


def _backtrack_row(row, forbidden_mask, num_elements, max_val):
    """
    Backtrack to complete row to num_elements elements, all <= max_val.
    Returns completed row or None.
    """
    if len(row) == num_elements:
        return row
    
    last = row[-1]
    
    for val in range(last + 1, max_val + 1):
        # Check new differences
        valid = True
        new_diffs = []
        for prev in row:
            d = val - prev
            # Check against forbidden
            if (forbidden_mask >> d) & 1:
                valid = False
                break
            # Check against existing diffs in this row
            if d in new_diffs:
                valid = False
                break
            new_diffs.append(d)
        
        if not valid:
            continue
        
        # Pruning: check if remaining positions can be filled
        # We need (num_elements - len(row) - 1) more elements
        # Each new element adds (len(row) + 1) new differences
        # Total remaining diffs needed: sum_{i=len(row)+1}^{num_elements-1} i
        # Actually simpler: just recurse
        
        row.append(val)
        result = _backtrack_row(row, forbidden_mask, num_elements, max_val)
        row.pop()
        
        if result is not None:
            return result
    
    return None


def _find_row_given_forbidden(forbidden_mask, num_elements, target_scope):
    """
    Find any valid row with scope <= target_scope avoiding forbidden_mask.
    Uses randomized backtracking for speed.
    Returns row or None.
    """
    row = [0]
    
    def backtrack(r, remaining, max_val):
        if remaining == 0:
            return r[:]
        
        last = r[-1]
        # Try values from last+1 to min(max_val, target_scope)
        # Use some randomization
        candidates = list(range(last + 1, min(max_val, target_scope) + 1))
        if len(candidates) > 20:
            # Sample a subset, biased toward smaller values
            random.shuffle(candidates)
            candidates = candidates[:20]
            candidates.sort()
        
        for val in candidates:
            valid = True
            new_diffs = set()
            for prev in r:
                d = val - prev
                if (forbidden_mask >> d) & 1:
                    valid = False
                    break
                if d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            r.append(val)
            result = backtrack(r, remaining - 1, max_val)
            r.pop()
            
            if result is not None:
                return result
        
        return None
    
    return backtrack(row, num_elements - 1, target_scope)


def _greedy_construct(n_rows, num_elements, rng, max_scope_hint=None):
    """
    Greedy: construct rows one by one, each minimizing its own scope
    given the forbidden differences from previous rows.
    """
    rows = []
    used_mask = 0
    
    for row_idx in range(n_rows):
        # Find minimum scope row avoiding used_mask
        # Start from a reasonable lower bound
        lower = num_elements - 1
        # Upper bound: be generous
        upper = max_scope_hint if max_scope_hint else 300
        
        best_row = None
        best_scope = None
        
        # Try to find a row with small scope
        for scope in range(lower, upper + 1):
            row = _find_row_given_forbidden(used_mask, num_elements, scope)
            if row is not None:
                best_row = row
                best_scope = scope
                break
        
        if best_row is None:
            # Fallback: just extend greedily
            row = [0]
            for _ in range(num_elements - 1):
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    conflict = False
                    for d in new_diffs:
                        if (used_mask >> d) & 1:
                            conflict = True
                            break
                    if not conflict and len(new_diffs) == len(set(new_diffs)):
                        break
                    x += 1
                row.append(x)
                for d in new_diffs:
                    used_mask |= (1 << d)
            best_row = row
            best_scope = row[-1]
        
        rows.append(best_row)
        used_mask |= _row_mask(best_row)
    
    return rows


def _local_search_row_swap(rows, num_elements, rng, time_limit, deadline):
    """
    Local search: for each row, try to re-optimize it given all other rows' differences.
    This is the key structural change: row-level re-optimization.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    n_rows = len(current_rows)
    
    def compute_forbidden_except(exclude_idx):
        mask = 0
        for i in range(n_rows):
            if i != exclude_idx:
                mask |= _row_mask(current_rows[i])
        return mask
    
    # Iterate: try to improve each row
    improved = True
    max_rounds = 20
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        if not improved:
            break
        improved = False
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                break
            
            forbidden = compute_forbidden_except(row_idx)
            old_row = current_rows[row_idx]
            old_scope_val = old_row[-1]
            
            # Try to find a row with smaller scope
            # Search scopes below current
            for new_scope in range(num_elements - 1, old_scope_val):
                if time.time() > deadline:
                    break
                new_row = _find_row_given_forbidden(forbidden, num_elements, new_scope)
                if new_row is not None:
                    current_rows[row_idx] = new_row
                    new_total_scope = _scope(current_rows)
                    if new_total_scope < best_scope:
                        best_scope = new_total_scope
                        best_rows = [r[:] for r in current_rows]
                    improved = True
                    break
            
            if time.time() > deadline:
                break
    
    return best_rows, best_scope


def _simulated_annealing_rows(rows, num_elements, rng, time_limit, deadline):
    """
    Row-level simulated annealing: pick a row, try to replace it with a 
    randomly perturbed version that has similar or better scope.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    n_rows = len(current_rows)
    temp = max(5.0, current_scope * 0.1)
    cooling = 0.999
    
    def compute_forbidden_except(exclude_idx):
        mask = 0
        for i in range(n_rows):
            if i != exclude_idx:
                mask |= _row_mask(current_rows[i])
        return mask
    
    while time.time() < deadline and temp > 0.5:
        row_idx = rng.randint(0, n_rows - 1)
        forbidden = compute_forbidden_except(row_idx)
        old_row = current_rows[row_idx]
        old_scope_val = old_row[-1]
        
        # Try to find a replacement row
        # Perturb: try scope in [num_elements-1, old_scope_val + 5]
        target_scope = rng.randint(num_elements - 1, old_scope_val + 5)
        
        new_row = _find_row_given_forbidden(forbidden, num_elements, target_scope)
        
        if new_row is None:
            temp *= cooling
            continue
        
        new_total_scope = max(max(r[-1] for i, r in enumerate(current_rows) if i != row_idx), new_row[-1])
        delta = new_total_scope - current_scope
        
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            current_rows[row_idx] = new_row
            current_scope = new_total_scope
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [r[:] for r in current_rows]
        
        temp *= cooling
    
    return best_rows, best_scope


def _construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 140.0
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multiple greedy constructions with different seeds and orderings
    for seed in range(20):
        if time.time() > DEADLINE - 60:
            break
        rng_local = random.Random(seed * 13 + 7)
        
        # Try different max_scope hints
        hint = counting_bound + 15 + seed
        rows = _greedy_construct(n_rows, num_elements, rng_local, max_scope_hint=hint)
        
        if rows and _is_valid(rows):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [r[:] for r in rows]
    
    # Also try with a tighter hint
    for seed in range(10):
        if time.time() > DEADLINE - 50:
            break
        rng_local = random.Random(seed * 31 + 11)
        hint = counting_bound + 5
        rows = _greedy_construct(n_rows, num_elements, rng_local, max_scope_hint=hint)
        
        if rows and _is_valid(rows):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = [r[:] for r in rows]
    
    # Fallback if nothing found
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used) and len(new_diffs) == len(set(new_diffs)):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _scope(best_rows)
    
    # Phase 2: Row-level local search (re-optimize individual rows)
    remaining = DEADLINE - time.time()
    if remaining > 30:
        improved_rows, improved_scope = _local_search_row_swap(
            best_rows, num_elements, rng,
            time_limit=remaining,
            deadline=DEADLINE - 25
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 3: Row-level SA
    if time.time() < DEADLINE - 20:
        improved_rows, improved_scope = _simulated_annealing_rows(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - time.time(),
            deadline=DEADLINE - 15
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 4: Another round of local search if time permits
    if time.time() < DEADLINE - 15:
        improved_rows, improved_scope = _local_search_row_swap(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - time.time(),
            deadline=DEADLINE - 10
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Final validation
    if best_rows and _is_valid(best_rows):
        return best_rows
    else:
        # Fallback: sequential construction
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used) and len(new_diffs) == len(set(new_diffs)):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    return _construct(n_rows, k)
