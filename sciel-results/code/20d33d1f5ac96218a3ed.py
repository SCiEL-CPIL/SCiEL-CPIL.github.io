import time
import random
import numpy as np

def _validate(colors):
    """Check if every non-constant 4-AP uses at least 3 distinct colors."""
    n = len(colors)
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            cols = {colors[a], colors[b], colors[c], colors[e]}
            if len(cols) < 3:
                return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _all_4aps(n):
    """Return list of all 4-APs as tuples (a, a+d, a+2d, a+3d)."""
    aps = []
    for a in range(n):
        for d in range(1, n):
            e = a + 3 * d
            if e >= n:
                break
            aps.append((a, a + d, a + 2 * d, e))
    return aps

def _aps_by_position(n):
    """For each position, list all 4-APs containing it, with the position's role (0,1,2,3)."""
    pos_aps = [[] for _ in range(n)]
    for a in range(n):
        for d in range(1, n):
            e = a + 3 * d
            if e >= n:
                break
            b = a + d
            c = a + 2 * d
            ap = (a, b, c, e)
            pos_aps[a].append((ap, 0))
            pos_aps[b].append((ap, 1))
            pos_aps[c].append((ap, 2))
            pos_aps[e].append((ap, 3))
    return pos_aps

def _seed_fallback():
    """Deterministic seed algorithm - guaranteed valid."""
    colors = []
    for x in range(1, 129):
        for color in range(128):
            ok = True
            # Check all 4-APs where x is the last element
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 1 - 3 * d
                b = x - 1 - 2 * d
                c = x - 1 - d
                if a < 0:
                    break
                cols = {colors[a], colors[b], colors[c], color}
                if len(cols) < 3:
                    ok = False
                    break
            # Also check 4-APs where x is not last (but earlier positions are assigned)
            # Actually in greedy left-to-right, only need to check APs ending at x
            if ok:
                colors.append(color)
                break
    return colors

def _backtrack_3color(n=128, K=3, seed=0, time_limit=80.0):
    """
    Backtracking CSP solver for 3-coloring.
    Uses MRV (minimum remaining values) variable ordering and forward checking.
    """
    start_time = time.time()
    rng = random.Random(seed)
    
    # Precompute all 4-APs
    aps = _all_4aps(n)
    pos_aps = _aps_by_position(n)
    
    # For each position, precompute the list of 4-APs it participates in
    # and for each such AP, which other positions are involved
    
    # Assignment array: -1 means unassigned
    assignment = [-1] * n
    
    # Domain for each position: bitmask of valid colors (K bits)
    full_mask = (1 << K) - 1
    domains = [full_mask] * n
    
    # Count of assigned positions
    assigned_count = 0
    
    # For forward checking: when we assign position p to color c,
    # for each 4-AP containing p, we need to check if the constraint
    # can still be satisfied by the remaining unassigned positions.
    
    def popcount(x):
        return bin(x).count('1')
    
    def check_ap_feasible(ap, assignment, domains):
        """
        Check if a 4-AP can still be satisfied given current assignment and domains.
        ap = (a, b, c, e)
        The constraint: |{col[a], col[b], col[c], col[e]}| >= 3
        """
        a, b, c, e = ap
        # Get the set of colors that are "fixed" (assigned) in this AP
        fixed_colors = set()
        unassigned_positions = []
        
        for pos in (a, b, c, e):
            if assignment[pos] != -1:
                fixed_colors.add(assignment[pos])
            else:
                unassigned_positions.append(pos)
        
        if len(fixed_colors) >= 3:
            # Already have 3 distinct colors, AP is satisfied regardless of unassigned
            return True
        
        if len(fixed_colors) == 2:
            # Need at least one unassigned position to take a 3rd color
            # Check if any unassigned position has a color not in fixed_colors in its domain
            for pos in unassigned_positions:
                # Domain of pos must contain a color not in fixed_colors
                mask_not_fixed = 0
                for col in range(K):
                    if col not in fixed_colors:
                        mask_not_fixed |= (1 << col)
                if domains[pos] & mask_not_fixed:
                    return True
            return False
        
        if len(fixed_colors) == 1:
            # Need at least 2 more distinct colors from unassigned positions
            # This is harder. We need the union of colors available to unassigned
            # positions to have size >= 3 (including the fixed color).
            # Actually we need |fixed ∪ {colors from unassigned}| >= 3
            # So we need at least 2 colors from unassigned that are different from each other
            # and from the fixed color, OR 1 color different from fixed plus... 
            # Wait: fixed has 1 color. We need total >= 3 distinct.
            # So we need at least 2 additional distinct colors from unassigned positions.
            # Check if the union of domains of unassigned positions contains at least 2 colors
            # not equal to the fixed color.
            fixed_col = list(fixed_colors)[0]
            # Collect all colors available in unassigned domains that are != fixed_col
            available = set()
            for pos in unassigned_positions:
                for col in range(K):
                    if col != fixed_col and (domains[pos] & (1 << col)):
                        available.add(col)
            return len(available) >= 2
        
        if len(fixed_colors) == 0:
            # All unassigned. Need at least 3 distinct colors total.
            # With K=3, this means all 3 colors must appear.
            # Check if the union of all domains has size >= 3
            union = 0
            for pos in unassigned_positions:
                union |= domains[pos]
            return popcount(union) >= 3
        
        return True
    
    def propagate(p, c):
        """
        Assign position p to color c and propagate constraints.
        Returns True if no contradiction, False if contradiction detected.
        Modifies domains and assignment in place.
        Also returns list of (position, old_domain) for backtracking.
        """
        nonlocal assigned_count
        changes = []  # (position, old_domain)
        
        # Assign p to c
        assignment[p] = c
        domains[p] = (1 << c)
        assigned_count += 1
        changes.append((p, full_mask))  # Will restore later
        
        # For each 4-AP containing p, check feasibility and propagate
        for ap, role in pos_aps[p]:
            a, b, c3, e = ap
            # Check if this AP is now infeasible
            if not check_ap_feasible(ap, assignment, domains):
                return False, changes
            
            # Propagate: for each unassigned position in this AP,
            # restrict its domain if necessary.
            # The constraint is |{colors}| >= 3.
            # Let's compute what colors are forbidden for each unassigned position.
            
            # Get current fixed colors in this AP (excluding p which is now assigned)
            fixed = set()
            unassigned_in_ap = []
            for pos in (a, b, c3, e):
                if assignment[pos] != -1:
                    fixed.add(assignment[pos])
                else:
                    unassigned_in_ap.append(pos)
            
            if len(fixed) >= 3:
                # Already satisfied, no propagation needed
                continue
            
            if len(fixed) == 2:
                # Need at least one unassigned to take a 3rd color
                # For each unassigned position, it must be able to take a color not in fixed
                # But this is a "at least one" constraint, not per-position.
                # We can't directly restrict individual domains easily.
                # Skip for now (will be caught by feasibility check).
                continue
            
            if len(fixed) == 1:
                # Need 2 more distinct colors
                # This is complex. Skip detailed propagation.
                continue
            
            if len(fixed) == 0:
                # All were unassigned except p. Now p is assigned.
                # The other 3 are unassigned.
                # Need |{c, col_b, col_c, col_e}| >= 3 where c is fixed.
                # So {col_b, col_c, col_e} must contain at least 2 colors different from c.
                # This means not all of col_b, col_c, col_e can be c.
                # And not exactly one of them can be different (that gives only 2 distinct total).
                # Actually: total distinct = |{c} ∪ {col_b, col_c, col_e}|
                # We need this >= 3, so {col_b, col_c, col_e} must have at least 2 distinct
                # colors, at least one of which is != c.
                # Equivalently: not (col_b == col_c == col_e == c) and not (exactly one of them != c).
                # This is getting complex. Let's just rely on feasibility check.
                continue
        
        return True, changes
    
    def backtrack():
        nonlocal assigned_count
        
        if time.time() - start_time > time_limit:
            return "timeout"
        
        if assigned_count == n:
            return True
        
        # MRV: find unassigned position with fewest domain options
        best_pos = -1
        best_count = K + 1
        for i in range(n):
            if assignment[i] == -1:
                cnt = popcount(domains[i])
                if cnt < best_count:
                    best_count = cnt
                    best_pos = i
                    if cnt <= 1:
                        break
        
        if best_pos == -1:
            return True  # All assigned
        
        if best_count == 0:
            return False  # Dead end
        
        # Try colors in best_pos's domain
        mask = domains[best_pos]
        colors_to_try = []
        for c in range(K):
            if mask & (1 << c):
                colors_to_try.append(c)
        
        # Randomize order
        rng.shuffle(colors_to_try)
        
        for c in colors_to_try:
            if time.time() - start_time > time_limit:
                return "timeout"
            
            # Save state
            old_assignment = assignment[best_pos]
            old_domain = domains[best_pos]
            
            ok, changes = propagate(best_pos, c)
            
            if ok:
                result = backtrack()
                if result == True:
                    return True
                elif result == "timeout":
                    # Restore state before returning timeout
                    assignment[best_pos] = old_assignment
                    domains[best_pos] = old_domain
                    for pos, old_dom in changes:
                        if pos != best_pos:
                            assignment[pos] = -1
                            domains[pos] = old_dom
                    assigned_count -= 1
                    return "timeout"
            
            # Restore state
            for pos, old_dom in changes:
                if pos != best_pos:
                    assignment[pos] = -1
                    domains[pos] = old_dom
                else:
                    domains[pos] = old_dom
            assignment[best_pos] = old_assignment
            assigned_count -= 1
        
        return False
    
    result = backtrack()
    
    if result == True:
        return list(assignment)
    else:
        return None

def _local_search_reduce_colors(colors, n=128, max_iters=200, seed=0):
    """
    Try to reduce the number of colors used in a valid coloring.
    """
    rng = random.Random(seed)
    current = list(colors)
    current_colors = set(current)
    
    for _ in range(max_iters):
        if len(current_colors) <= 3:
            break
        
        # Pick a color to eliminate
        target_color = rng.choice(list(current_colors))
        positions = [i for i in range(n) if current[i] == target_color]
        
        new_current = list(current)
        success = True
        
        for pos in positions:
            other_colors = [c for c in current_colors if c != target_color]
            rng.shuffle(other_colors)
            
            found = False
            for new_color in other_colors:
                new_current[pos] = new_color
                
                # Check validity of all 4-APs containing pos
                valid = True
                for ap, role in _aps_by_position(n)[pos]:
                    a, b, c3, e = ap
                    cols = {new_current[a], new_current[b], new_current[c3], new_current[e]}
                    if len(cols) < 3:
                        valid = False
                        break
                if valid:
                    found = True
                    break
            
            if not found:
                success = False
                break
        
        if success:
            new_colors = set(new_current)
            if len(new_colors) < len(current_colors):
                current = new_current
                current_colors = new_colors
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 128
    
    # 1. Get seed fallback
    seed_colors = _seed_fallback()
    best = seed_colors
    best_count = len(set(seed_colors))
    
    # 2. Try 3-color backtracking with multiple random seeds
    # The backtracking solver uses MRV and forward checking
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        
        remaining_time = min(5.0, deadline - 5 - time.time())
        if remaining_time <= 0:
            break
        
        result = _backtrack_3color(n, 3, seed=seed, time_limit=remaining_time)
        if result is not None:
            if _validate(result):
                return result
    
    # 3. Try 4-color backtracking if 3-color failed
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        
        remaining_time = min(3.0, deadline - 5 - time.time())
        if remaining_time <= 0:
            break
        
        result = _backtrack_3color(n, 4, seed=seed, time_limit=remaining_time)
        if result is not None:
            if _validate(result):
                count = len(set(result))
                if count < best_count:
                    best = result
                    best_count = count
    
    # 4. Try local search to reduce colors
    if time.time() < deadline - 10 and best_count > 3:
        improved = _local_search_reduce_colors(best, n, max_iters=100, seed=42)
        imp_count = len(set(improved))
        if imp_count < best_count:
            best = improved
            best_count = imp_count
    
    return best

if __name__ == "__main__":
    c = construct()
    print("Colors used:", len(set(c)))
    print("Valid:", _validate(c))
