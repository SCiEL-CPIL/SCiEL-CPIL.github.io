"""
Strategy: Targeted scope search with randomized backtracking + forward checking.
For a given scope M, we try to construct n_rows rows of k+1 increasing integers
starting at 0 such that all positive differences are globally distinct.
We use backtracking with forward checking: when placing an element, we check
that it doesn't create internal duplicates or conflict with already-used diffs.
We binary search on M to find the smallest feasible scope.
"""
import time
import random
import sys
import math

# EVOLVE-BLOCK-START

def _row_diffs(row):
    """Return list of all positive differences within a row (i<j)."""
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))


def _backtrack_row(used_diffs, max_scope, num_elements, rng, max_attempts=20000):
    """
    Backtrack to construct one row of num_elements increasing integers starting at 0,
    with max element <= max_scope, avoiding differences in used_diffs.
    Uses forward checking: at each step, only consider values that don't create
    internal duplicate diffs or conflict with used_diffs.
    Returns row or None.
    """
    if num_elements <= 1:
        return [0]
    if num_elements == 2:
        # Need to pick one value v in 1..max_scope such that v not in used_diffs
        candidates = [v for v in range(1, max_scope + 1) if v not in used_diffs]
        if not candidates:
            return None
        return [0, rng.choice(candidates)]
    
    # General case: build row incrementally
    # row[0] = 0, need to pick row[1], ..., row[num_elements-1]
    # row[i] < row[i+1], row[num_elements-1] <= max_scope
    
    # We'll use iterative deepening with backtracking
    # For efficiency, we'll use a recursive approach with memoization of failed states
    
    def backtrack(current_row, used_internal_diffs):
        """
        current_row: list of values chosen so far (starts with [0])
        used_internal_diffs: set of differences already created within this row
        Returns completed row or None.
        """
        if len(current_row) == num_elements:
            # Check that max element <= max_scope (should be enforced during selection)
            if current_row[-1] <= max_scope:
                # Final check: all diffs in this row are not in used_diffs
                all_diffs = _row_diffs(current_row)
                for d in all_diffs:
                    if d in used_diffs:
                        return None
                return current_row[:]
            return None
        
        pos = len(current_row)
        min_val = current_row[-1] + 1
        max_val = max_scope
        
        # Forward checking: for each candidate value, check if it creates
        # internal duplicate diffs or conflicts with used_diffs
        candidates = []
        for v in range(min_val, max_val + 1):
            # Check new diffs created by adding v
            new_diffs = set()
            ok = True
            for i in range(len(current_row)):
                d = v - current_row[i]
                if d in used_internal_diffs:
                    ok = False
                    break
                if d in used_diffs:
                    ok = False
                    break
                new_diffs.add(d)
            if ok:
                # Also need to ensure that we can still complete the row
                # remaining elements: num_elements - len(current_row) - 1
                remaining = num_elements - len(current_row) - 1
                if remaining > 0:
                    # Need at least 'remaining' more values > v, up to max_scope
                    if max_scope - v < remaining:
                        continue
                candidates.append((v, new_diffs))
        
        if not candidates:
            return None
        
        # Shuffle candidates for randomization
        rng.shuffle(candidates)
        
        # Limit candidates to avoid exponential blowup
        if len(candidates) > 50:
            # Use a heuristic: prefer values that are not too large (to leave room)
            # but also not too small (to avoid tight constraints)
            # We'll take a random subset biased toward middle values
            mid = (min_val + max_val) // 2
            candidates.sort(key=lambda x: abs(x[0] - mid))
            candidates = candidates[:50]
            rng.shuffle(candidates)
        
        for v, new_diffs in candidates:
            new_internal = used_internal_diffs | new_diffs
            result = backtrack(current_row + [v], new_internal)
            if result is not None:
                return result
        
        return None
    
    # Try multiple times with different random orderings
    for _ in range(max_attempts):
        result = backtrack([0], set())
        if result is not None:
            return result
        # Occasionally break early if we've been trying too long
        if _ % 1000 == 0 and _ > 0:
            # Check if we're making progress by trying a simpler heuristic
            pass
    
    return None


def _construct_for_scope(n_rows, num_elements, max_scope, rng, time_limit):
    """
    Try to construct n_rows rows with max element <= max_scope.
    Returns list of rows or None.
    """
    start = time.time()
    used_diffs = set()
    rows = []
    
    for row_idx in range(n_rows):
        if time.time() > time_limit:
            return None
        
        row = _backtrack_row(used_diffs, max_scope, num_elements, rng, max_attempts=5000)
        if row is None:
            return None
        
        rows.append(row)
        # Add all diffs from this row to used_diffs
        for d in _row_diffs(row):
            used_diffs.add(d)
    
    return rows


def _local_search_refine(rows, num_elements, max_scope, rng, time_limit):
    """
    Given a valid solution with scope max_scope, try to reduce the scope
    by moving elements to smaller values.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    n_rows = len(current_rows)
    
    # Precompute used diffs
    def get_all_diffs(row):
        return set(_row_diffs(row))
    
    row_diff_sets = [get_all_diffs(r) for r in current_rows]
    all_diffs = set()
    for ds in row_diff_sets:
        all_diffs |= ds
    
    max_iter = 50000
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, len(current_rows[row_idx]) - 1)
        old_val = current_rows[row_idx][pos]
        
        row = current_rows[row_idx]
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < len(row) - 1 else max_scope
        
        if low >= high:
            continue
        
        # Try to reduce
        if rng.random() < 0.6:
            search_high = min(old_val - 1, high)
            if low > search_high:
                continue
            new_val = rng.randint(low, search_high)
        else:
            new_val = rng.randint(low, high)
        
        if new_val == old_val:
            continue
        
        # Check if moving is valid
        row[pos] = new_val
        new_diffs = get_all_diffs(row)
        
        # Check internal uniqueness
        if len(new_diffs) != len(_row_diffs(row)):
            row[pos] = old_val
            continue
        
        # Check no conflict with other rows
        other_diffs = set()
        for i, ds in enumerate(row_diff_sets):
            if i != row_idx:
                other_diffs |= ds
        
        if new_diffs & other_diffs:
            row[pos] = old_val
            continue
        
        # Valid move
        row_diff_sets[row_idx] = new_diffs
        new_scope = _scope(current_rows)
        
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [r[:] for r in current_rows]
    
    return best_rows, best_scope


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 140.0  # safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    
    # Counting bound
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    # Binary search on scope M
    # Lower bound: counting_bound
    # Upper bound: start generous, e.g., counting_bound + 50
    lo = counting_bound
    hi = counting_bound + 60
    
    # First, find any valid solution with a generous scope
    best_rows = None
    best_scope = float('inf')
    
    # Try to find a solution with scope = hi
    if time.time() < DEADLINE - 60:
        rows = _construct_for_scope(n_rows, num_elements, hi, rng, time_limit=DEADLINE - 60)
        if rows is not None:
            best_rows = rows
            best_scope = _scope(rows)
    
    # If we didn't find one, try even larger scopes
    if best_rows is None:
        for extra in range(0, 100, 10):
            if time.time() > DEADLINE - 30:
                break
            test_scope = hi + extra
            rows = _construct_for_scope(n_rows, num_elements, test_scope, rng, time_limit=DEADLINE - 30)
            if rows is not None:
                best_rows = rows
                best_scope = _scope(rows)
                break
    
    # If still nothing, use fallback sequential construction
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _scope(rows)
    
    # Now binary search for the minimum scope
    # We know best_scope is feasible. Try to find smaller.
    # Use binary search between counting_bound and best_scope
    search_lo = counting_bound
    search_hi = best_scope - 1  # we want to find if something smaller works
    
    # But binary search requires that if M is feasible, all M' > M are feasible.
    # This is true for this problem (if you can fit in M, you can fit in M+1).
    # So we can binary search.
    
    # However, our backtracking may not find a solution even if one exists.
    # So we'll do a linear search downward from best_scope, with limited attempts.
    
    # Actually, let's do a smarter approach: try scopes from counting_bound upward
    # but with limited time per scope.
    
    # Phase 2: Try to reduce scope
    current_best = best_scope
    for test_scope in range(counting_bound, current_best):
        if time.time() > DEADLINE - 20:
            break
        
        # Try to construct with this scope
        rows = _construct_for_scope(n_rows, num_elements, test_scope, rng, 
                                     time_limit=min(time.time() + 15, DEADLINE - 20))
        if rows is not None:
            s = _scope(rows)
            if s < current_best:
                current_best = s
                best_rows = rows
            # If we found a solution exactly at test_scope, we can stop
            # (since we're going upward from counting_bound)
            if s <= test_scope:
                break
    
    # Phase 3: Local search to further refine
    if time.time() < DEADLINE - 10:
        refined_rows, refined_scope = _local_search_refine(
            best_rows, num_elements, current_best, rng,
            time_limit=DEADLINE - 15
        )
        if refined_scope < current_best:
            current_best = refined_scope
            best_rows = refined_rows
    
    # Final validation
    if _is_valid(best_rows):
        return best_rows
    else:
        # Fallback
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

# EVOLVE-BLOCK-END
