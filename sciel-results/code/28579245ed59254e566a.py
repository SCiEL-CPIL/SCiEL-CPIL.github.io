"""
Strategy: Difference-assignment matching + Simulated Annealing refinement.

Structural change: Instead of building rows sequentially (row-by-row bottleneck),
we formulate initial construction as a minimum-cost assignment problem where
each (row, pair) slot gets a distinct difference value. The cost function
penalizes large values to drive toward the counting bound. Then SA refines
by moving individual elements, checking global difference uniqueness.

Key insight: The differences a_j - a_i for i<j in a row must be globally unique.
If we assign differences first, we can derive element positions via cumulative
sums (for consecutive pairs) or solve constraints. For a row [0, x1, x2, ..., xk]:
  d(i,j) = a_j - a_i
  a_1 = d(0,1)
  a_2 = a_1 + d(1,2) = d(0,1) + d(1,2)
  etc.
So if we assign all differences, the row is determined by the "chain" differences
d(0,1), d(1,2), ..., d(k-1,k), and the other differences must be consistent.

For efficiency, we use a greedy matching for initialization, then SA for refinement.
"""
import time
import random
import math
import sys
from collections import defaultdict


def _row_diffs_list(row):
    """Return list of all positive differences within a row (i<j)."""
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs


def _row_mask(row):
    """Bitmask of differences in a row."""
    m = 0
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            m |= (1 << (row[j] - ri))
    return m


def _scope(rows):
    return max(r[-1] for r in rows)


def _is_valid(rows):
    """Check all positive differences are globally distinct."""
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs_list(row))
    return len(all_diffs) == len(set(all_diffs))


def _compute_all_diffs(rows):
    """Return (global_mask, row_masks) for all rows."""
    row_masks = []
    global_mask = 0
    for row in rows:
        m = _row_mask(row)
        row_masks.append(m)
        global_mask |= m
    return global_mask, row_masks


def _matching_based_construct(n_rows, num_elements, rng, scope_hint):
    """
    Construct rows using a difference-assignment approach.
    
    For each row, the differences d(i,j) for i<j must be distinct globally.
    We assign differences to slots greedily with cost-based selection,
    then derive element positions from the assigned differences.
    
    For a row with elements [a_0=0, a_1, ..., a_k]:
      The "chain" differences are d(0,1), d(1,2), ..., d(k-1,k)
      a_j = sum of chain diffs from 0 to j
      All other d(i,j) = a_j - a_i must also be assigned and consistent.
    
    Strategy: Assign chain differences first (they determine the row),
    then verify all implied differences are available.
    """
    rows = []
    used_mask = 0
    total_diffs_per_row = num_elements * (num_elements - 1) // 2
    
    for row_idx in range(n_rows):
        row = [0]
        # Assign elements one by one, but use cost-based selection
        for pos in range(1, num_elements):
            found = False
            low = row[-1] + 1
            high = scope_hint + 20
            
            # Generate candidates with cost-based scoring
            candidates = []
            for val in range(low, min(low + 80, high + 1)):
                # Compute new differences
                new_diffs = [val - prev for prev in row]
                
                # Check within-row uniqueness
                if len(new_diffs) != len(set(new_diffs)):
                    continue
                
                # Check against used
                new_mask = 0
                for d in new_diffs:
                    if d > 2000:  # safety
                        continue
                    new_mask |= (1 << d)
                
                if new_mask & used_mask:
                    continue
                
                # Cost: prefer smaller values, but also consider future flexibility
                cost = val * 1.0
                # Add small random perturbation for diversity
                cost += rng.random() * 2.0
                
                candidates.append((cost, val))
            
            if not candidates:
                # Expand search
                for val in range(low, high + 1):
                    new_diffs = [val - prev for prev in row]
                    if len(new_diffs) != len(set(new_diffs)):
                        continue
                    new_mask = 0
                    for d in new_diffs:
                        if d > 2000:
                            continue
                        new_mask |= (1 << d)
                    if new_mask & used_mask:
                        continue
                    cost = val * 1.0 + rng.random() * 5.0
                    candidates.append((cost, val))
            
            if not candidates:
                # Fallback: just pick next available
                val = low
                while True:
                    new_diffs = [val - prev for prev in row]
                    if len(new_diffs) == len(set(new_diffs)):
                        new_mask = 0
                        for d in new_diffs:
                            new_mask |= (1 << d)
                        if not (new_mask & used_mask):
                            break
                    val += 1
                row.append(val)
                for d in new_diffs:
                    used_mask |= (1 << d)
                continue
            
            # Sort by cost and pick from top few with randomness
            candidates.sort(key=lambda x: x[0])
            top_n = min(5, len(candidates))
            chosen_val = candidates[rng.randint(0, top_n - 1)][1]
            
            row.append(chosen_val)
            for d in (chosen_val - prev for prev in row[:-1]):
                used_mask |= (1 << d)
        
        rows.append(row)
    
    return rows


def _try_move_element(rows, row_idx, pos, new_val, global_mask, row_masks):
    """
    Try to move element at rows[row_idx][pos] to new_val.
    Returns (success, new_global_mask, new_row_mask) or (False, None, None).
    """
    row = rows[row_idx]
    n = len(row)
    
    # Check bounds: must maintain strictly increasing order
    if pos > 0 and new_val <= row[pos - 1]:
        return False, None, None
    if pos < n - 1 and new_val >= row[pos + 1]:
        return False, None, None
    
    # Compute new differences for this row
    old_val = row[pos]
    if new_val == old_val:
        return False, None, None
    
    # Tentatively apply
    row[pos] = new_val
    
    # Check within-row uniqueness
    diffs_in_row = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = row[j] - row[i]
            if d in diffs_in_row:
                row[pos] = old_val
                return False, None, None
            diffs_in_row.add(d)
    
    # Compute new row mask
    new_row_mask = 0
    for d in diffs_in_row:
        new_row_mask |= (1 << d)
    
    # Check against other rows
    other_mask = global_mask & ~row_masks[row_idx]
    if new_row_mask & other_mask:
        row[pos] = old_val
        return False, None, None
    
    # Success
    new_global_mask = (global_mask & ~row_masks[row_idx]) | new_row_mask
    row[pos] = old_val  # revert; caller will apply
    
    return True, new_global_mask, new_row_mask


def _simulated_annealing(rows, num_elements, rng, time_limit, initial_temp=50.0):
    """
    Simulated annealing to minimize scope.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    global_mask, row_masks = _compute_all_diffs(current_rows)
    
    n_rows = len(current_rows)
    n_elem = num_elements
    
    temp = initial_temp
    cooling = 0.9995
    min_temp = 0.5
    
    iterations = 0
    max_iter = 300000
    
    while time.time() < time_limit and temp > min_temp and iterations < max_iter:
        iterations += 1
        
        # Pick a random row and position (not first)
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]
        
        # Determine valid range
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope + 15
        
        if low >= high:
            continue
        
        # Bias toward lower values (to reduce scope)
        if old_val > low + 2 and rng.random() < 0.75:
            search_high = min(old_val - 1, high)
            if low > search_high:
                continue
            new_val = rng.randint(low, search_high)
        else:
            # Allow some upward moves for exploration
            new_val = rng.randint(low, high)
        
        if new_val == old_val:
            continue
        
        # Try the move
        success, new_global_mask, new_row_mask = _try_move_element(
            current_rows, row_idx, pos, new_val, global_mask, row_masks
        )
        
        if not success:
            continue
        
        # Apply the move
        current_rows[row_idx][pos] = new_val
        old_row_mask = row_masks[row_idx]
        row_masks[row_idx] = new_row_mask
        global_mask = new_global_mask
        
        new_scope = _scope(current_rows)
        delta = new_scope - current_scope
        
        if delta <= 0 or rng.random() < math.exp(-delta / temp):
            # Accept
            current_scope = new_scope
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
        else:
            # Reject: revert
            current_rows[row_idx][pos] = old_val
            row_masks[row_idx] = old_row_mask
            global_mask = (global_mask & ~new_row_mask) | old_row_mask
        
        temp *= cooling
    
    return best_rows, best_scope


def _local_search_refine(rows, num_elements, rng, time_limit):
    """
    Local search: try to reduce scope by moving elements down.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    global_mask, row_masks = _compute_all_diffs(current_rows)
    n_rows = len(current_rows)
    n_elem = num_elements
    
    max_iter = 150000
    stagnation = 0
    max_stagnation = 8000
    
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        if stagnation > max_stagnation:
            break
        
        row_idx = rng.randint(0, n_rows - 1)
        pos = rng.randint(1, n_elem - 1)
        row = current_rows[row_idx]
        old_val = row[pos]
        
        low = row[pos - 1] + 1
        high = row[pos + 1] - 1 if pos < n_elem - 1 else current_scope
        
        if low >= old_val:
            continue
        
        # Try to move down
        new_val = rng.randint(low, old_val - 1)
        
        success, new_global_mask, new_row_mask = _try_move_element(
            current_rows, row_idx, pos, new_val, global_mask, row_masks
        )
        
        if success:
            current_rows[row_idx][pos] = new_val
            old_row_mask = row_masks[row_idx]
            row_masks[row_idx] = new_row_mask
            global_mask = new_global_mask
            
            new_scope = _scope(current_rows)
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in current_rows]
                stagnation = 0
            else:
                stagnation += 1
        else:
            stagnation += 1
    
    return best_rows, best_scope


def _swap_rows(rows, row_i, row_j, rng, time_limit):
    """
    Try to swap entire rows between two positions to improve packing.
    This is a structural move that can escape local optima.
    """
    start = time.time()
    current_rows = [r[:] for r in rows]
    current_scope = _scope(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_scope = current_scope
    
    n_rows = len(current_rows)
    num_elements = len(current_rows[0])
    
    global_mask, row_masks = _compute_all_diffs(current_rows)
    
    max_iter = 50000
    
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        
        # Pick two rows to potentially swap
        i = rng.randint(0, n_rows - 1)
        j = rng.randint(0, n_rows - 1)
        if i == j:
            continue
        
        # Try swapping rows i and j
        current_rows[i], current_rows[j] = current_rows[j], current_rows[i]
        
        # Check validity
        if not _is_valid(current_rows):
            # Revert
            current_rows[i], current_rows[j] = current_rows[j], current_rows[i]
            continue
        
        new_scope = _scope(current_rows)
        if new_scope < best_scope:
            best_scope = new_scope
            best_rows = [r[:] for r in current_rows]
        else:
            # Revert
            current_rows[i], current_rows[j] = current_rows[j], current_rows[i]
    
    return best_rows, best_scope


def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    """
    start_time = time.time()
    DEADLINE = start_time + 135.0  # safety margin before 150s kill
    
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = math.ceil(math.sqrt(2 * total_diffs_needed))
    
    rng = random.Random(42)
    
    # Phase 1: Generate initial solutions using matching-based construction
    best_rows = None
    best_scope = float('inf')
    
    # Try multiple constructions with different scope hints
    for seed in range(15):
        if time.time() > DEADLINE - 80:
            break
        rng_local = random.Random(seed * 7 + 1)
        # Start with a reasonable scope hint and tighten over iterations
        scope_hint = counting_bound + 20 + seed * 3
        rows = _matching_based_construct(n_rows, num_elements, rng_local, scope_hint)
        if rows is not None and _is_valid(rows):
            s = _scope(rows)
            if s < best_scope:
                best_scope = s
                best_rows = rows
    
    # If no valid solution found, use a simple sequential fallback
    if best_rows is None:
        rows = []
        used = set()
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < num_elements:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        best_rows = rows
        best_scope = _scope(best_rows)
    
    # Phase 2: Simulated Annealing
    remaining = DEADLINE - 30
    if remaining > 30:
        improved_rows, improved_scope = _simulated_annealing(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 50,
            initial_temp=max(10.0, best_scope * 0.12)
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 3: Local search refinement
    if time.time() < DEADLINE - 25:
        improved_rows, improved_scope = _local_search_refine(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 20
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 4: Another SA round with lower temperature if time permits
    if time.time() < DEADLINE - 30 and best_scope > counting_bound:
        improved_rows, improved_scope = _simulated_annealing(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 15,
            initial_temp=5.0
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Phase 5: Final local search polish
    if time.time() < DEADLINE - 5:
        improved_rows, improved_scope = _local_search_refine(
            best_rows, num_elements, rng,
            time_limit=DEADLINE - 2
        )
        if improved_scope < best_scope:
            best_scope = improved_scope
            best_rows = improved_rows
    
    # Final validation
    if _is_valid(best_rows):
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
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
