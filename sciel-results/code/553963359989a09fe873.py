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

def _csp_init(n_rows, k, target_scope, deadline):
    """
    CSP backtracking initialization with forward checking.
    Tries to construct n_rows rows where each row has max element <= target_scope.
    """
    used = set()
    rows = []
    
    def build_row(forbidden, max_val):
        """
        Build a single row using DFS with forward checking.
        Returns the row or None if impossible within max_val.
        """
        row = [0]
        local_used = set()
        
        def dfs(idx, last_val):
            if time.time() > deadline:
                return False
            if idx == k:
                return True
            
            # Determine range for next value
            # Must be > last_val and <= max_val
            # Also, we need enough room for remaining k - 1 - idx elements
            # Each step must be at least 1, so last_val + (k - idx) <= max_val
            if last_val + (k - idx) > max_val:
                return False
                
            # Heuristic: try smaller values first to keep scope low, 
            # but also consider spreading out. 
            # For CSP, we want to find *a* valid row quickly.
            # Trying smaller values first is usually good for minimizing max.
            
            # Optimization: Pre-compute potential conflicts?
            # For each candidate x, check if {x - a for a in row} conflicts with forbidden or local_used.
            
            start = last_val + 1
            # We can prune: if we pick x, the remaining elements must fit in [x+1, max_val].
            # This is handled by the recursive check.
            
            # To improve branching, we can shuffle the range slightly or use a heuristic.
            # Let's try values in a specific order: maybe not strictly increasing from start?
            # Actually, for difference sets, small gaps are bad. 
            # But we are constrained by 'forbidden' differences.
            
            # Let's iterate x from start to max_val
            # We can add a small randomization to break symmetry if stuck, 
            # but pure DFS with forward check is strong.
            
            # Optimization: Check if any x in range is valid.
            # We can iterate x.
            for x in range(start, max_val + 1):
                # Forward check: 
                # 1. New diffs must not be in forbidden
                # 2. New diffs must not be in local_used
                # 3. (Implicitly) new diffs must be distinct from each other (guaranteed by strict increase)
                
                new_diffs = {x - a for a in row}
                if new_diffs & forbidden:
                    continue
                if new_diffs & local_used:
                    continue
                
                # Tentatively add
                row.append(x)
                local_used |= new_diffs
                
                if dfs(idx + 1, x):
                    return True
                else:
                    row.pop()
                    local_used -= new_diffs
            
            return False
        
        if dfs(1, 0):
            return list(row)
        return None

    for _ in range(n_rows):
        if time.time() > deadline - 2:
            return None, None
        
        # Try to build a row within target_scope
        # We can add some randomness to the starting point or order if we want diversity,
        # but for a tight bound, deterministic DFS is often best.
        # However, to avoid getting stuck in local minima of the CSP, 
        # we might want to randomize the order of x candidates slightly?
        # No, let's stick to smallest-first for the first pass to find *a* solution.
        
        new_row = build_row(used, target_scope)
        
        if new_row is None:
            # Try with a slightly larger scope for this row? 
            # Or return failure?
            # If we fail, the whole init fails.
            return None, None
        
        rows.append(new_row)
        used |= _compute_diffs(new_row)
        
    return rows, used

def _sa_optimize(rows, n_rows, k, deadline):
    """
    Optimized Simulated Annealing.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _get_scope(rows)
    
    # Build initial used set
    used = set()
    for row in rows:
        used |= _compute_diffs(row)
        
    current_scope = best_scope
    T = 50.0
    cooling = 0.995
    min_T = 0.1
    
    # Track stagnation
    stagnation = 0
    max_stagnation = 2000
    
    # Precompute row diff sets for faster updates?
    row_diffs = [_compute_diffs(r) for r in rows]
    
    while T > min_T and time.time() < deadline - 5:
        # Decide move type: 70% perturb element, 30% rebuild row
        if random.random() < 0.7:
            # Perturb a single element
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            pos = random.randint(1, len(row) - 1)
            old_val = row[pos]
            
            # Try to move value
            # Bias towards reducing scope if this row has the max
            if row[pos] == best_scope:
                # Try smaller values
                new_val = random.randint(1, max(1, old_val - 1)) if old_val > 1 else 1
            else:
                # Try random in range
                new_val = random.randint(1, current_scope + 10)
                
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
            # The diffs that remain from this row are `used - old_diffs`.
            # But `used` contains diffs from ALL rows.
            # We need to ensure new_diffs do not conflict with any diff in `used` that is NOT in `old_diffs`.
            # Note: `old_diffs` might not be a subset of `used` if there were bugs, but assuming valid state, it is.
            # Actually, `used` is the union of all row diffs.
            # `remaining = used - old_diffs` is the set of diffs from other rows + other diffs in this row.
            # We must check `new_diffs & remaining`.
            
            remaining = used - old_diffs
            if new_diffs & remaining:
                T *= cooling
                continue
                
            # Valid move
            row[pos] = new_val
            used = (used - old_diffs) | new_diffs
            row_diffs[r_idx] = new_diffs | (row_diffs[r_idx] - old_diffs) # Update row diff cache? 
            # Actually, row_diffs[r_idx] needs to be recomputed or updated carefully.
            # Let's just recompute row_diffs[r_idx] if we keep it.
            # For simplicity and correctness, let's not maintain row_diffs cache in SA, 
            # or just recompute it when needed. 
            # Since we only need `used`, we update `used`.
            
            new_scope = _get_scope(rows)
            delta = new_scope - current_scope
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in rows]
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                # Revert
                row[pos] = old_val
                used = (used - new_diffs) | old_diffs
                stagnation += 1
                
        else:
            # Rebuild a row
            # Pick the row with the largest max element, or random
            if random.random() < 0.5:
                # Pick row with max element
                max_row_idx = 0
                max_val = -1
                for i, r in enumerate(rows):
                    if r[-1] > max_val:
                        max_val = r[-1]
                        max_row_idx = i
                r_idx = max_row_idx
            else:
                r_idx = random.randint(0, n_rows - 1)
                
            old_row = rows[r_idx]
            old_diffs = _compute_diffs(old_row)
            
            # Remove old row's diffs from global used
            used -= old_diffs
            
            # Try to rebuild with CSP
            # Give it a hint: current best scope
            new_row = _csp_init_single_row(used, k, best_scope, deadline - 1)
            
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
                            stagnation = 0
                        else:
                            stagnation += 1
                    else:
                        # Revert
                        rows[r_idx] = old_row
                        used = (used - new_diffs) | old_diffs
                        stagnation += 1
                else:
                    # Conflict, revert
                    used |= old_diffs
                    stagnation += 1
            else:
                # Failed to rebuild, restore old row
                used |= old_diffs
                stagnation += 1
                
        T *= cooling
        
        # If stagnating, try a restart or increase temperature
        if stagnation > max_stagnation:
            # Restart with a new CSP init or perturb best
            if time.time() < deadline - 10:
                # Try a new CSP init with the current best scope
                new_rows, new_used = _csp_init(n_rows, k, best_scope, deadline - 10)
                if new_rows is not None:
                    new_scope = _get_scope(new_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in new_rows]
                        rows = [list(r) for r in new_rows]
                        used = new_used
                        current_scope = new_scope
                        T = 50.0
                        stagnation = 0
                    else:
                        # If new CSP is worse or same, just reset stagnation
                        stagnation = 0
            else:
                break
                
    return best_rows

def _csp_init_single_row(forbidden, k, max_scope_hint, deadline):
    """
    Rebuild a single row using CSP/DFS.
    """
    def build_row(forbidden, max_val):
        row = [0]
        local_used = set()
        
        def dfs(idx, last_val):
            if time.time() > deadline:
                return False
            if idx == k:
                return True
            
            if last_val + (k - idx) > max_val:
                return False
                
            start = last_val + 1
            for x in range(start, max_val + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & forbidden:
                    continue
                if new_diffs & local_used:
                    continue
                
                row.append(x)
                local_used |= new_diffs
                
                if dfs(idx + 1, x):
                    return True
                else:
                    row.pop()
                    local_used -= new_diffs
            
            return False
        
        if dfs(1, 0):
            return list(row)
        return None

    return build_row(forbidden, max_scope_hint)

def _greedy_init(n_rows, k, skip_prob=0.0):
    """
    Greedy initialization.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                # With some probability, skip to create diversity
                if skip_prob > 0 and random.random() < skip_prob:
                    # Skip a few valid values
                    skipped = False
                    for _ in range(random.randint(1, 5)):
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            row.append(x)
                            used |= new_diffs
                            skipped = True
                            break
                    if not skipped:
                        # If we skipped too far and hit a conflict or limit, just take the first valid
                        # But we already incremented x, so we might need to backtrack or just take x if valid
                        # Actually, the loop above breaks on first valid. If none found, we just continue outer loop
                        # But we must append something. Let's just take the first valid we found before skipping?
                        # Simpler: just append x if valid, else continue.
                        # The logic above is a bit messy. Let's simplify:
                        pass 
                else:
                    row.append(x)
                    used |= new_diffs
        rows.append(row)
    return rows, used

def construct(n_rows, k):
    deadline = time.time() + 145.0  # Safety margin
    
    # Try multiple initialization strategies
    best_rows = None
    best_scope = float('inf')
    
    # Strategy 1: CSP Init with a reasonable upper bound
    # Estimate upper bound: n_rows * (k*(k+1)/2) is the total number of diffs.
    # The max diff is at least sqrt(2 * total_diffs) roughly? 
    # Let's start with a generous bound and let CSP find a tighter one.
    # Or just use greedy to get a bound first.
    
    # Get a quick bound from greedy
    rows_greedy, _ = _greedy_init(n_rows, k, skip_prob=0.0)
    if _is_valid(rows_greedy):
        greedy_scope = _get_scope(rows_greedy)
    else:
        greedy_scope = 200 # Fallback
    
    # Try CSP with a target slightly below greedy scope
    target = max(50, greedy_scope - 10)
    
    # Try CSP init
    rows_csp, _ = _csp_init(n_rows, k, target, deadline - 50)
    if rows_csp is not None and _is_valid(rows_csp):
        csp_scope = _get_scope(rows_csp)
        if csp_scope < best_scope:
            best_scope = csp_scope
            best_rows = [list(r) for r in rows_csp]
    
    # If CSP failed or greedy is better, use greedy
    if best_rows is None:
        if _is_valid(rows_greedy):
            best_scope = greedy_scope
            best_rows = [list(r) for r in rows_greedy]
        else:
            # Fallback to pure greedy
            rows, _ = _greedy_init(n_rows, k, skip_prob=0.0)
            return rows
    else:
        # Compare with greedy
        if greedy_scope < best_scope and _is_valid(rows_greedy):
            best_scope = greedy_scope
            best_rows = [list(r) for r in rows_greedy]
            
    # Optimize with SA
    rows = _sa_optimize(best_rows, n_rows, k, deadline)
    
    # Final check
    if not _is_valid(rows):
        return best_rows
        
    return rows
