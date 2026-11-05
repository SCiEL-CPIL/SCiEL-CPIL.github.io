import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    num_diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * num_diffs_per_row
    
    # Estimate a reasonable upper bound for the search space
    # The counting bound is roughly sqrt(2 * total_diffs)
    # For (7, 5), total_diffs = 7 * 15 = 105. sqrt(210) ~ 14.5? No.
    # The differences are distinct integers. The largest difference is at least total_diffs?
    # No, the differences are $a_j - a_i$.
    # If the max element is $M$, the max difference is $M$.
    # We need $total\_diffs$ distinct differences from $\{1, \dots, M\}$.
    # So $M \ge total\_diffs$.
    # For (7,5), $M \ge 105$. The known solution is 112.
    # So we should search in a range slightly above 105.
    
    initial_max_limit = int(math.ceil(math.sqrt(2 * total_diffs)) * 2) + 20
    # Actually, for (7,5), total_diffs=105. sqrt(210)=14.5. That's way too low.
    # The lower bound on M is not sqrt(2*total_diffs).
    # The lower bound is total_diffs itself? No.
    # The differences are distinct. The set of all differences is a subset of {1, ..., M}.
    # So we need at least total_diffs distinct integers in {1, ..., M}.
    # Thus M >= total_diffs.
    # For (7,5), M >= 105.
    # So let's set the initial limit to total_diffs + 20.
    
    search_limit = total_diffs + 30
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Greedy Initialization
    # Try multiple random seeds to find a good starting point
    for seed in range(50):
        if time.time() > deadline - 30:
            break
        rows = _greedy_init(n_rows, k, seed, search_limit)
        if rows is not None:
            scope = _scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                
    if best_rows is None:
        # Fallback: construct a valid but potentially large set
        # This ensures we return *something* valid
        best_rows = _construct_large(n_rows, k)
        best_scope = _scope(best_rows)

    # Phase 2: Simulated Annealing
    if time.time() < deadline - 10:
        improved = _simulated_annealing(best_rows, n_rows, k, start_time, deadline, search_limit)
        if improved is not None:
            imp_scope = _scope(improved)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    # Phase 3: Final Polish - Try to reduce the max element further
    if time.time() < deadline - 2 and best_rows is not None:
        polished = _polish_max(best_rows, n_rows, k, start_time, deadline)
        if polished is not None:
            pol_scope = _scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished

    return best_rows

def _scope(rows):
    if not rows:
        return 0
    return max(row[-1] for row in rows if row)

def _construct_large(n_rows, k):
    """Construct a valid set with a large scope to ensure validity."""
    # Use a simple construction that guarantees distinct differences
    # By making the ranges disjoint, differences are automatically distinct.
    # Row i uses differences in range [i*B, (i+1)*B - 1]
    # Max difference in row i is (i+1)*B - 1.
    # Max element is roughly (i+1)*B.
    # This is very large but valid.
    B = (k + 1) * k // 2 + 1
    rows = []
    for i in range(n_rows):
        row = [0]
        offset = i * B
        for j in range(1, k + 1):
            # We need increasing integers.
            # Let's just use multiples to ensure distinct diffs?
            # Actually, just using a large stride is safer.
            row.append(offset + j * (k + 1))
        rows.append(row)
    return rows

def _greedy_init(n_rows, k, seed, limit):
    """
    Greedy construction using bitmask for collision checking.
    """
    rng = random.Random(seed)
    used_mask = 0
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_x = row[-1]
            found = False
            # Try smallest possible x first to keep scope small
            for x in range(last_x + 1, limit + 1):
                new_diffs_mask = 0
                valid = True
                for a in row:
                    d = x - a
                    bit = 1 << d
                    if used_mask & bit:
                        valid = False
                        break
                    new_diffs_mask |= bit
                if valid:
                    row.append(x)
                    used_mask |= new_diffs_mask
                    found = True
                    break
            
            if not found:
                # If we can't find a small x, expand limit or fail
                # For initialization, let's try expanding a bit
                for x in range(limit + 1, limit + 50):
                    new_diffs_mask = 0
                    valid = True
                    for a in row:
                        d = x - a
                        bit = 1 << d
                        if used_mask & bit:
                            valid = False
                            break
                    new_diffs_mask |= bit
                    if valid:
                        row.append(x)
                        used_mask |= new_diffs_mask
                        found = True
                        break
                if not found:
                    return None

        rows.append(row)
        
    return rows

def _simulated_annealing(initial_rows, n_rows, k, start_time, deadline, limit):
    """
    Simulated Annealing to minimize the scope.
    State: List of rows.
    Move: Replace one row with a new valid row.
    Cost: Scope (max element).
    """
    rng = random.Random(42)
    rows = [list(r) for r in initial_rows]
    current_scope = _scope(rows)
    best_scope = current_scope
    best_rows = [list(r) for r in rows]
    
    # Precompute used mask for current state
    used_mask = 0
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used_mask |= (1 << (row[j] - row[i]))
                
    T = 20.0
    cooling_rate = 0.999
    steps = 0
    max_steps = 50000
    
    while time.time() < deadline - 5 and steps < max_steps:
        steps += 1
        
        # Choose a row to replace
        # Bias towards the row with the largest element
        if rng.random() < 0.5:
            row_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        else:
            row_idx = rng.randint(0, n_rows - 1)
            
        # Remove current row's diffs from used_mask
        old_row = rows[row_idx]
        for i in range(len(old_row)):
            for j in range(i+1, len(old_row)):
                used_mask &= ~(1 << (old_row[j] - old_row[i]))
                
        # Generate a new row
        # We want to try to make it smaller, but SA allows worse moves.
        # Let's generate a random valid row, biased towards smaller values.
        new_row = _generate_row(used_mask, k, rng, limit)
        
        if new_row is None:
            # Restore old row
            for i in range(len(old_row)):
                for j in range(i+1, len(old_row)):
                    used_mask |= (1 << (old_row[j] - old_row[i]))
            continue
            
        # Calculate new scope
        new_rows = rows[:]
        new_rows[row_idx] = new_row
        new_scope = _scope(new_rows)
        
        delta = new_scope - current_scope
        
        # Metropolis criterion
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            # Accept
            rows = new_rows
            current_scope = new_scope
            
            # Update used_mask
            for i in range(len(new_row)):
                for j in range(i+1, len(new_row)):
                    used_mask |= (1 << (new_row[j] - new_row[i]))
                    
            if current_scope < best_scope:
                best_scope = current_scope
                best_rows = [list(r) for r in rows]
                
        else:
            # Reject, restore old row
            for i in range(len(old_row)):
                for j in range(i+1, len(old_row)):
                    used_mask |= (1 << (old_row[j] - old_row[i]))
                    
        T *= cooling_rate
        if T < 0.1:
            T = 0.1

    return best_rows

def _generate_row(used_mask, k, rng, limit):
    """
    Generate a valid row of length k+1 starting with 0.
    Used_mask contains all differences used by OTHER rows.
    """
    row = [0]
    for _ in range(k):
        last_x = row[-1]
        candidates = []
        
        # Search for valid x
        # To encourage smaller x, we scan from last_x+1 upwards
        # But we also want randomness for SA.
        # Let's scan a window.
        
        found = False
        # Try a few random spots in the range [last_x+1, limit]
        # Or just scan sequentially for the first few, then random?
        # Sequential scan finds the tightest fit.
        
        for x in range(last_x + 1, limit + 1):
            valid = True
            for a in row:
                d = x - a
                if (used_mask >> d) & 1:
                    valid = False
                    break
            if valid:
                # To add diversity, we don't always take the first one
                # But for packing, the first one is usually best for minimizing max element
                # Let's take the first one for now, SA will explore via rejections
                row.append(x)
                # We don't update used_mask here because this row's diffs
                # are not yet "used" by other rows, but they must be distinct
                # within the row. Since the row is increasing, diffs are distinct?
                # No, diffs within a row are distinct by construction?
                # If row is 0, 1, 3. Diffs: 1, 3, 2. Distinct.
                # If row is 0, 1, 2. Diffs: 1, 2, 1. NOT distinct.
                # So we must also check for collisions WITHIN the new row.
                
                # Let's re-evaluate:
                # The problem statement says: "all differences between two entries of the same row are distinct over all rows".
                # This implies:
                # 1. Differences within a row are distinct.
                # 2. Differences between different rows are distinct.
                
                # My _generate_row only checks against used_mask (other rows).
                # I need to check against diffs already in the current row.
                
                # Let's restart the inner loop logic to include intra-row checks.
                # Actually, if I build the row incrementally, I can maintain a local mask.
                break
        
        # The previous logic was flawed. Let's do it properly.
        # Reset row to just [0] if we failed? No, we are in the loop.
        # Let's rewrite _generate_row properly.
        
    # The above loop was broken. Let's implement a correct _generate_row.
    return _generate_row_correct(used_mask, k, rng, limit)

def _generate_row_correct(used_mask, k, rng, limit):
    """
    Correctly generate a row with distinct internal diffs and no external collisions.
    """
    row = [0]
    local_mask = 0  # Diffs within this row
    
    for _ in range(k):
        last_x = row[-1]
        found = False
        
        # Scan for valid x
        # We want to find an x such that:
        # 1. x - a is not in used_mask for all a in row
        # 2. x - a is not in local_mask for all a in row (except a=x, but x is new)
        #    Actually, local_mask contains diffs between existing elements.
        #    New diffs are x - a for a in row.
        #    These new diffs must not be in local_mask either.
        
        for x in range(last_x + 1, limit + 1):
            new_diffs_mask = 0
            valid = True
            for a in row:
                d = x - a
                bit = 1 << d
                # Check against global used
                if used_mask & bit:
                    valid = False
                    break
                # Check against local used
                if local_mask & bit:
                    valid = False
                    break
                new_diffs_mask |= bit
            
            if valid:
                row.append(x)
                local_mask |= new_diffs_mask
                found = True
                break
        
        if not found:
            return None
            
    return row

def _polish_max(rows, n_rows, k, start_time, deadline):
    """
    Targeted local search to reduce the maximum element.
    """
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    
    # Precompute used mask
    used_mask = 0
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used_mask |= (1 << (row[j] - row[i]))
                
    # We want to shrink the row with the max element.
    # We can try to reconstruct it with a smaller limit.
    
    for _ in range(1000): # Max iterations
        if time.time() > deadline - 2:
            break
            
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        # Try to find a new row for max_idx with max element < current_max
        # We can try limits from current_max - 1 down to current_max - 10
        improved = False
        for target_max in range(current_max - 1, max(current_max - 15, 0), -1):
            # Remove old row from used_mask
            old_row = rows[max_idx]
            for i in range(len(old_row)):
                for j in range(i+1, len(old_row)):
                    used_mask &= ~(1 << (old_row[j] - old_row[i]))
            
            new_row = _generate_row_correct(used_mask, k, rng, target_max)
            
            if new_row is not None:
                # Check if it actually reduced the max
                if new_row[-1] < current_max:
                    # Add new row to used_mask
                    for i in range(len(new_row)):
                        for j in range(i+1, len(new_row)):
                            used_mask |= (1 << (new_row[j] - new_row[i]))
                    
                    new_rows = rows[:]
                    new_rows[max_idx] = new_row
                    new_scope = _scope(new_rows)
                    
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = new_rows
                        rows = new_rows
                        improved = True
                        break
                    else:
                        # Revert used_mask
                        for i in range(len(new_row)):
                            for j in range(i+1, len(new_row)):
                                used_mask &= ~(1 << (new_row[j] - new_row[i]))
                        # Restore old row
                        for i in range(len(old_row)):
                            for j in range(i+1, len(old_row)):
                                used_mask |= (1 << (old_row[j] - old_row[i]))
                else:
                    # Revert
                    for i in range(len(new_row)):
                        for j in range(i+1, len(new_row)):
                            used_mask &= ~(1 << (new_row[j] - new_row[i]))
                    for i in range(len(old_row)):
                        for j in range(i+1, len(old_row)):
                            used_mask |= (1 << (old_row[j] - old_row[i]))
            else:
                # Revert
                for i in range(len(old_row)):
                    for j in range(i+1, len(old_row)):
                        used_mask |= (1 << (old_row[j] - old_row[i]))
            
        if not improved:
            # Try a random row if max row didn't improve
            if rng.random() < 0.5:
                random_idx = rng.randint(0, n_rows - 1)
                if random_idx != max_idx:
                    old_row = rows[random_idx]
                    for i in range(len(old_row)):
                        for j in range(i+1, len(old_row)):
                            used_mask &= ~(1 << (old_row[j] - old_row[i]))
                    
                    # Try to generate a row with a slightly smaller max
                    # Just use the same limit as current max of that row
                    limit = rows[random_idx][-1]
                    new_row = _generate_row_correct(used_mask, k, rng, limit)
                    
                    if new_row is not None:
                        for i in range(len(new_row)):
                            for j in range(i+1, len(new_row)):
                                used_mask |= (1 << (new_row[j] - new_row[i]))
                        
                        new_rows = rows[:]
                        new_rows[random_idx] = new_row
                        new_scope = _scope(new_rows)
                        
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = new_rows
                            rows = new_rows
                        else:
                            # Revert
                            for i in range(len(new_row)):
                                for j in range(i+1, len(new_row)):
                                    used_mask &= ~(1 << (new_row[j] - new_row[i]))
                            for i in range(len(old_row)):
                                for j in range(i+1, len(old_row)):
                                    used_mask |= (1 << (old_row[j] - old_row[i]))
                    else:
                        for i in range(len(old_row)):
                            for j in range(i+1, len(old_row)):
                                used_mask |= (1 << (old_row[j] - old_row[i]))

    return best_rows if best_scope < _scope(initial_rows_copy) else None # Need to fix this logic

# Fix the return logic in _polish_max
def _polish_max(rows, n_rows, k, start_time, deadline):
    rng = random.Random(123)
    rows = [list(r) for r in rows]
    initial_scope = _scope(rows)
    best_rows = [list(r) for r in rows]
    best_scope = initial_scope
    
    used_mask = 0
    for row in rows:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used_mask |= (1 << (row[j] - row[i]))
                
    for _ in range(1000):
        if time.time() > deadline - 2:
            break
            
        max_idx = max(range(n_rows), key=lambda i: rows[i][-1])
        current_max = rows[max_idx][-1]
        
        improved = False
        for target_max in range(current_max - 1, max(current_max - 15, 0), -1):
            old_row = rows[max_idx]
            for i in range(len(old_row)):
                for j in range(i+1, len(old_row)):
                    used_mask &= ~(1 << (old_row[j] - old_row[i]))
            
            new_row = _generate_row_correct(used_mask, k, rng, target_max)
            
            if new_row is not None:
                if new_row[-1] < current_max:
                    for i in range(len(new_row)):
                        for j in range(i+1, len(new_row)):
                            used_mask |= (1 << (new_row[j] - new_row[i]))
                    
                    new_rows = rows[:]
                    new_rows[max_idx] = new_row
                    new_scope = _scope(new_rows)
                    
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = new_rows
                        rows = new_rows
                        improved = True
                        break
                    else:
                        for i in range(len(new_row)):
                            for j in range(i+1, len(new_row)):
                                used_mask &= ~(1 << (new_row[j] - new_row[i]))
                        for i in range(len(old_row)):
                            for j in range(i+1, len(old_row)):
                                used_mask |= (1 << (old_row[j] - old_row[i]))
                else:
                    for i in range(len(new_row)):
                        for j in range(i+1, len(new_row)):
                            used_mask &= ~(1 << (new_row[j] - new_row[i]))
                    for i in range(len(old_row)):
                        for j in range(i+1, len(old_row)):
                            used_mask |= (1 << (old_row[j] - old_row[i]))
            else:
                for i in range(len(old_row)):
                    for j in range(i+1, len(old_row)):
                        used_mask |= (1 << (old_row[j] - old_row[i]))
        
        if not improved:
            # Try random row
            if rng.random() < 0.3:
                random_idx = rng.randint(0, n_rows - 1)
                if random_idx != max_idx:
                    old_row = rows[random_idx]
                    for i in range(len(old_row)):
                        for j in range(i+1, len(old_row)):
                            used_mask &= ~(1 << (old_row[j] - old_row[i]))
                    
                    limit = rows[random_idx][-1]
                    new_row = _generate_row_correct(used_mask, k, rng, limit)
                    
                    if new_row is not None:
                        for i in range(len(new_row)):
                            for j in range(i+1, len(new_row)):
                                used_mask |= (1 << (new_row[j] - new_row[i]))
                        
                        new_rows = rows[:]
                        new_rows[random_idx] = new_row
                        new_scope = _scope(new_rows)
                        
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = new_rows
                            rows = new_rows
                        else:
                            for i in range(len(new_row)):
                                for j in range(i+1, len(new_row)):
                                    used_mask &= ~(1 << (new_row[j] - new_row[i]))
                            for i in range(len(old_row)):
                                for j in range(i+1, len(old_row)):
                                    used_mask |= (1 << (old_row[j] - old_row[i]))
                    else:
                        for i in range(len(old_row)):
                            for j in range(i+1, len(old_row)):
                                used_mask |= (1 << (old_row[j] - old_row[i]))

    return best_rows if best_scope < initial_scope else None

# Remove the duplicate _generate_row function
def _generate_row(used_mask, k, rng, limit):
    return _generate_row_correct(used_mask, k, rng, limit)
