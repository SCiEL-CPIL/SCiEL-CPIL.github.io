import time
import random
import sys

def _compute_diffs(row):
    """Compute all pairwise differences for a single row."""
    n = len(row)
    diffs = set()
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _is_valid(rows):
    """Check if all differences are distinct across all rows."""
    seen = set()
    for row in rows:
        for d in _compute_diffs(row):
            if d in seen:
                return False
            seen.add(d)
    return True

def _get_scope(rows):
    """Return the largest entry in any row."""
    if not rows:
        return 0
    return max(max(row) for row in rows)

def _greedy_init(n_rows, k, used=None, skip_prob=0.0):
    """
    Greedy initialization with optional random skipping to break deterministic traps.
    """
    if used is None:
        used = set()
    else:
        used = set(used)
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                # With some probability, skip a few valid values to create diversity
                if skip_prob > 0 and random.random() < skip_prob and x < 300:
                    skipped = False
                    for extra in range(random.randint(1, 4)):
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            skipped = True
                            break
                    if not skipped:
                        row.append(x)
                        used |= new_diffs
                else:
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
    return rows, used

def _rebuild_row(forbidden, k, max_scope_hint):
    """
    Try to rebuild a row to minimize its maximum element,
    given the set of used differences from other rows.
    Uses greedy with limited backtracking.
    """
    best_row = None
    best_last = max_scope_hint + 10
    
    # Greedy approach with multiple restarts
    for attempt in range(3):
        row = [0]
        local_used = set()
        x = 0
        success = True
        
        while len(row) < k + 1:
            x += 1
            if x > best_last:
                success = False
                break
            new_diffs = {x - a for a in row}
            if new_diffs & local_used:
                continue
            if new_diffs & forbidden:
                continue
            
            row.append(x)
            local_used |= new_diffs
            
        if success and len(row) == k + 1:
            if row[-1] < best_last:
                best_last = row[-1]
                best_row = list(row)
                # Add some randomness for next attempt
                if attempt < 2 and random.random() < 0.3:
                    # Perturb the solution slightly
                    pos = random.randint(1, len(row) - 1)
                    old_val = row[pos]
                    if old_val > 1:
                        row[pos] = old_val + random.randint(1, 3)
                        # Re-sort and check validity
                        row.sort()
                        if row[0] != 0:
                            row[0] = 0
                        # Recompute local_used
                        local_used = _compute_diffs(row)
                        if local_used & forbidden:
                            row = [0]
                            local_used = set()
                            x = 0
                            # Restart this attempt
                            continue
                        
    return best_row

def _sa_optimize(rows, n_rows, k, deadline):
    """
    Optimized Simulated Annealing with Row Rebuilds and Randomized Restarts.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    # Build initial used set
    used = set()
    for row in rows:
        used |= _compute_diffs(row)
        
    current_scope = best_scope
    T = 100.0
    cooling = 0.998
    min_T = 0.5
    
    restart_count = 0
    max_restarts = 5
    
    while T > min_T and time.time() < deadline:
        # Decide move type: 70% perturb element, 30% rebuild row
        if random.random() < 0.7:
            # Perturb a single element
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = random.randint(1, len(row) - 1)
            old_val = row[pos]
            
            # Try to move value closer to 0 or within a range
            # Bias towards reducing scope
            if random.random() < 0.5:
                # Try smaller
                new_val = random.randint(1, max(1, old_val - 1)) if old_val > 1 else 1
            else:
                # Try random in range
                new_val = random.randint(1, current_scope + 5)
                
            if new_val == old_val:
                continue
                
            # Check order
            if pos > 0 and new_val <= row[pos - 1]:
                continue
            if pos < len(row) - 1 and new_val >= row[pos + 1]:
                continue
                
            # Compute old diffs involving pos
            old_diffs = set()
            for i in range(pos):
                old_diffs.add(old_val - row[i])
            for j in range(pos + 1, len(row)):
                old_diffs.add(row[j] - old_val)
                
            # Compute new diffs involving pos
            new_diffs = set()
            for i in range(pos):
                new_diffs.add(new_val - row[i])
            for j in range(pos + 1, len(row)):
                new_diffs.add(row[j] - new_val)
                
            # Check conflict
            remaining = used - old_diffs
            if new_diffs & remaining:
                T *= cooling
                continue
                
            # Valid move
            row[pos] = new_val
            used = (used - old_diffs) | new_diffs
            
            new_scope = _get_scope(rows)
            delta = new_scope - current_scope
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in rows]
            else:
                # Revert
                row[pos] = old_val
                used = (used - new_diffs) | old_diffs
                
        else:
            # Rebuild a row
            r_idx = random.randint(0, n_rows - 1)
            old_row = rows[r_idx]
            old_diffs = _compute_diffs(old_row)
            
            # Remove old row's diffs from global used
            used -= old_diffs
            
            # Try to rebuild
            new_row = _rebuild_row(used, k, best_scope)
            
            if new_row is not None:
                new_diffs = _compute_diffs(new_row)
                
                # Check if new diffs conflict with current used (which excludes old row)
                if not (new_diffs & used):
                    rows[r_idx] = new_row
                    used |= new_diffs
                    
                    new_scope = _get_scope(rows)
                    delta = new_scope - current_scope
                    
                    if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                        current_scope = new_scope
                        if current_scope < best_scope:
                            best_scope = current_scope
                            best_rows = [list(r) for r in rows]
                    else:
                        # Revert
                        rows[r_idx] = old_row
                        used = (used - new_diffs) | old_diffs
                else:
                    # Conflict, revert
                    used |= old_diffs
            else:
                # Failed to rebuild, restore old row
                used |= old_diffs
                
        T *= cooling
        
        # Check for stagnation and trigger restart
        if restart_count < max_restarts and T < 20.0:
            if time.time() < deadline - 10:
                # Try a randomized restart
                restart_count += 1
                skip_prob = 0.1 * restart_count
                new_rows, new_used = _greedy_init(n_rows, k, used=None, skip_prob=skip_prob)
                
                if _is_valid(new_rows):
                    new_scope = _get_scope(new_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in new_rows]
                        rows = [list(r) for r in new_rows]
                        used = new_used
                        current_scope = new_scope
                        T = 50.0  # Reset temperature
    
    return best_rows

def construct(n_rows, k):
    deadline = time.time() + 145.0  # Safety margin
    
    # Try multiple initialization strategies
    best_rows = None
    best_scope = float('inf')
    
    # Strategy 1: Pure greedy
    rows1, _ = _greedy_init(n_rows, k, skip_prob=0.0)
    if _is_valid(rows1):
        scope1 = _get_scope(rows1)
        if scope1 < best_scope:
            best_scope = scope1
            best_rows = [list(r) for r in rows1]
    
    # Strategy 2: Randomized greedy with small skip probability
    for attempt in range(3):
        rows2, _ = _greedy_init(n_rows, k, skip_prob=0.05 + 0.05 * attempt)
        if _is_valid(rows2):
            scope2 = _get_scope(rows2)
            if scope2 < best_scope:
                best_scope = scope2
                best_rows = [list(r) for r in rows2]
    
    if best_rows is None:
        # Fallback to pure greedy
        rows, _ = _greedy_init(n_rows, k, skip_prob=0.0)
        return rows
    
    # Verify
    if not _is_valid(best_rows):
        return best_rows
        
    # Optimize
    rows = _sa_optimize(best_rows, n_rows, k, deadline)
    
    # Final check
    if not _is_valid(rows):
        return best_rows
        
    return rows
