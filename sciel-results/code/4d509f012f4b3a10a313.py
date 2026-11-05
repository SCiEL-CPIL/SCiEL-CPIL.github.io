import time
import random
import math
import bisect
import copy

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Phase 1: Get an initial feasible solution via randomized greedy
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple seeds to get a good starting point
    for attempt in range(10):
        if time.time() > deadline - 60:
            break
        rows = _greedy_construct(n_rows, k, seed=attempt * 137 + 42)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = rows
    
    if best_rows is None:
        # Fallback: very loose greedy
        best_rows = _greedy_construct(n_rows, k, seed=999)
        if best_rows is None:
            # Last resort: just return something valid-ish (might be invalid if n_rows/k large, but avoids crash)
            best_rows = []
            used = set()
            for _ in range(n_rows):
                row = [0]
                for _ in range(k):
                    x = row[-1] + 1
                    while True:
                        diffs = {x - a for a in row}
                        if not (diffs & used):
                            break
                        x += 1
                    row.append(x)
                    used.update(diffs)
                best_rows.append(row)
        best_scope = _scope(best_rows)
    
    # Phase 2: Local Search to reduce scope
    if time.time() < deadline - 10:
        improved = _local_search(best_rows, n_rows, k, start_time, deadline)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _all_diffs(rows):
    """Return set of all positive differences across all rows."""
    diffs = set()
    for row in rows:
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(row[j] - row[i])
    return diffs

def _validate(rows, n_rows, k):
    """Check if rows form a valid difference triangle set."""
    all_diffs = set()
    for row in rows:
        if len(row) != k + 1:
            return False
        if row[0] != 0:
            return False
        for i in range(1, len(row)):
            if row[i] <= row[i-1]:
                return False
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                d = row[j] - row[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True

def _greedy_construct(n_rows, k, seed):
    """Randomized greedy for all rows."""
    rng = random.Random(seed)
    used = set()
    rows = []

    for _ in range(n_rows):
        row = _greedy_row(used, k, rng)
        if row is None:
            return None
        rows.append(row)
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                d = row[j] - row[i]
                used.add(d)

    return rows

def _greedy_row(used_diffs, k, rng):
    """Best-fit greedy for a single row."""
    row = [0]

    for _ in range(k):
        valid_xs = []
        scored_xs = []
        last_x = row[-1]
        limit = 300

        x = last_x + 1
        while len(valid_xs) < 30 and x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used_diffs):
                valid_xs.append(x)
                score = _packing_score(new_diffs, sorted(used_diffs), x)
                scored_xs.append((score, x))
            x += 1

        if not valid_xs:
            x = last_x + 1
            while len(valid_xs) < 30:
                new_diffs = {x - a for a in row}
                if not (new_diffs & used_diffs):
                    valid_xs.append(x)
                    score = _packing_score(new_diffs, sorted(used_diffs), x)
                    scored_xs.append((score, x))
                x += 1

        if not valid_xs:
            return None

        scored_xs.sort(key=lambda t: t[0])
        top_n = min(5, len(scored_xs))
        chosen_x = scored_xs[rng.randint(0, top_n - 1)][1]
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_diffs.update(new_diffs)

    return row

def _packing_score(new_diffs, used_sorted, x):
    """Score a candidate x by how well its differences pack into used space."""
    if not used_sorted:
        return x * 1.0

    total_dist = 0.0
    for d in new_diffs:
        idx = bisect.bisect_left(used_sorted, d)
        min_dist = float('inf')
        if idx < len(used_sorted):
            min_dist = min(min_dist, abs(d - used_sorted[idx]))
        if idx > 0:
            min_dist = min(min_dist, abs(d - used_sorted[idx - 1]))
        total_dist += min_dist

    return total_dist / max(len(new_diffs), 1) + 0.1 * x

def _local_search(initial_rows, n_rows, k, start_time, deadline):
    """
    Local search to minimize the scope.
    Strategy:
    1. Maintain a set of all used differences.
    2. Identify the row with the maximum last element.
    3. Try to reduce the last element (and potentially others) to a smaller valid value.
    4. If successful, update the state and repeat.
    5. If stuck, perturb a random row and restart the local improvement phase.
    """
    rng = random.Random(42)
    state = [list(r) for r in initial_rows]
    best_state = [list(r) for r in state]
    best_scope = _scope(state)
    
    # Helper to compute all diffs for a set of rows
    def compute_diffs(rows):
        diffs = set()
        for row in rows:
            n = len(row)
            for i in range(n):
                for j in range(i+1, n):
                    diffs.add(row[j] - row[i])
        return diffs

    # Helper to get diffs of all rows except one
    def get_other_diffs(rows, exclude_idx):
        diffs = set()
        for i, row in enumerate(rows):
            if i == exclude_idx:
                continue
            n = len(row)
            for p in range(n):
                for q in range(p+1, n):
                    diffs.add(row[q] - row[p])
        return diffs

    # Initial diff set
    current_diffs = compute_diffs(state)
    
    max_iterations = 100000
    iter_count = 0
    
    while time.time() < deadline - 5 and iter_count < max_iterations:
        iter_count += 1
        
        # Find the row with the largest last element
        max_scope_row_idx = max(range(n_rows), key=lambda i: state[i][-1])
        current_max = state[max_scope_row_idx][-1]
        
        # Try to improve this specific row
        improved = False
        other_diffs = get_other_diffs(state, max_scope_row_idx)
        
        # We try to reduce the last element. We can also reduce intermediate elements if it helps.
        # Simple strategy: Try to replace the last element with a smaller valid one.
        # If that fails, try to replace the second-to-last, etc.
        
        row_to_fix = list(state[max_scope_row_idx])
        
        # Try to reduce the last element
        target_pos = k # index of last element
        current_val = row_to_fix[target_pos]
        
        # Search for a smaller valid value for row_to_fix[target_pos]
        # The new value must be > row_to_fix[target_pos-1]
        min_val = row_to_fix[target_pos-1] + 1
        found_new_val = None
        
        # Search downwards from current_val - 1
        for new_val in range(current_val - 1, min_val - 1, -1):
            # Check if new_val is valid with the rest of the row
            # The rest of the row is row_to_fix[0:target_pos]
            # We need to ensure all diffs between new_val and previous elements are not in other_diffs
            # AND not in the diffs generated by the previous elements themselves (which are fixed)
            
            # Actually, we must ensure the NEW row's diffs don't conflict with other_diffs.
            # The diffs *within* the row are determined by the row elements.
            # If we only change the last element, the internal diffs involving the last element change.
            # The internal diffs NOT involving the last element remain the same.
            
            # Let's check if the new set of diffs for this row is disjoint from other_diffs.
            # And also disjoint from the "fixed" internal diffs? No, the internal diffs are part of the row's contribution.
            # The condition is: All diffs in the new row must be unique across ALL rows.
            # So: (Diffs(new_row) \ Union(Diffs(other_rows))) must be empty? No.
            # Diffs(new_row) must be disjoint from Union(Diffs(other_rows)).
            # AND Diffs(new_row) must be unique internally? No, a single row can have repeated diffs?
            # "all differences between two entries of the same row are distinct over all rows"
            # This implies:
            # 1. Within a row, all pairwise differences are distinct? No, "distinct over all rows" usually means the set of all differences from all rows contains no duplicates.
            # Wait, the standard definition of Difference Triangle Set (DTS) or Golomb Ruler variant:
            # "all differences ... are distinct over all rows".
            # This means if d1 is a diff in row A and d2 is a diff in row B, d1 != d2.
            # It also implies that within a single row, if a diff appears twice, it's a duplicate.
            # So yes, the set of all differences from all rows must have size equal to the total number of pairwise differences.
            
            # So, we just need: Diffs(new_row) is disjoint from other_diffs.
            # Note: Diffs(new_row) might contain duplicates internally?
            # If row is [0, 1, 2], diffs are 1, 1, 2. 1 appears twice.
            # The problem statement says "all differences ... are distinct".
            # This usually implies the set of values is unique.
            # So a row like [0,1,2] is invalid because diff 1 appears twice.
            # So we must also ensure Diffs(new_row) has no duplicates internally.
            
            new_row_diffs = set()
            valid_internal = True
            n = len(row_to_fix)
            # Construct the hypothetical new row
            hyp_row = row_to_fix[:target_pos] + [new_val]
            
            for i in range(len(hyp_row)):
                for j in range(i+1, len(hyp_row)):
                    d = hyp_row[j] - hyp_row[i]
                    if d in new_row_diffs:
                        valid_internal = False
                        break
                    new_row_diffs.add(d)
                if not valid_internal:
                    break
            
            if not valid_internal:
                continue
                
            # Check against other rows
            if new_row_diffs & other_diffs:
                continue
                
            # It's valid!
            found_new_val = new_val
            break
            
        if found_new_val is not None:
            # Apply the change
            row_to_fix[target_pos] = found_new_val
            state[max_scope_row_idx] = row_to_fix
            
            # Update global diffs
            # Remove old diffs of this row, add new ones
            old_row = state[max_scope_row_idx] # This is the new row now? No, we just assigned it.
            # We need to update current_diffs.
            # Easier: Recompute current_diffs? O(N*K^2). Acceptable for small K.
            current_diffs = compute_diffs(state)
            
            new_scope = _scope(state)
            if new_scope < best_scope:
                best_scope = new_scope
                best_state = [list(r) for r in state]
            
            improved = True
        else:
            # Could not improve the max row by reducing its last element.
            # Try to perturb a random row to escape local optimum
            if iter_count % 100 == 0: # Perturb periodically
                pert_idx = rng.randint(0, n_rows - 1)
                if pert_idx == max_scope_row_idx:
                    pert_idx = (pert_idx + 1) % n_rows
                
                other_diffs_pert = get_other_diffs(state, pert_idx)
                new_row = _greedy_row(other_diffs_pert, k, rng)
                if new_row is not None:
                    state[pert_idx] = new_row
                    current_diffs = compute_diffs(state)
        
    # Final validation
    if _validate(best_state, n_rows, k):
        return best_state
    else:
        if _validate([list(r) for r in initial_rows], n_rows, k):
            return [list(r) for r in initial_rows]
        return None
