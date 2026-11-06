import time
import random
import sys

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

def _precompute_ap_data(n):
    """Precompute all 4-APs and their index in the AP list."""
    # For each position i, store list of (ap_index, position_index_in_ap)
    pos_to_aps = [[] for _ in range(n)]
    # Store all 4-APs as (a, b, c, e)
    all_aps = []
    ap_index_by_quad = {}
    
    for a in range(n):
        for d in range(1, n):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e >= n:
                break
            ap_idx = len(all_aps)
            all_aps.append((a, b, c, e))
            ap_index_by_quad[(a, b, c, e)] = ap_idx
            pos_to_aps[a].append((ap_idx, 0))
            pos_to_aps[b].append((ap_idx, 1))
            pos_to_aps[c].append((ap_idx, 2))
            pos_to_aps[e].append((ap_idx, 3))
    
    return all_aps, pos_to_aps, ap_index_by_quad

def _count_violations_fast(colors, all_aps):
    """Count the number of 4-APs with fewer than 3 distinct colors."""
    count = 0
    for a, b, c, e in all_aps:
        cols = {colors[a], colors[b], colors[c], colors[e]}
        if len(cols) < 3:
            count += 1
    return count

def _seed_fallback():
    """Deterministic seed algorithm - guaranteed valid."""
    colors = []
    for x in range(1, 129):
        for color in range(128):
            ok = True
            # Check all 4-APs ending at or including position x
            for d in range(1, (x - 1) // 3 + 1):
                if x - 1 - 3 * d < 0:
                    continue
                idx_a = x - 1 - 3 * d
                idx_b = x - 1 - 2 * d
                idx_c = x - 1 - d
                cols = {color, colors[idx_a], colors[idx_b], colors[idx_c]}
                if len(cols) < 3:
                    ok = False
                    break
            if not ok:
                continue
            # Check 4-APs where x-1 is not the last but is included
            for d in range(1, (x - 1) // 2 + 1):
                # x-1 could be at position a, a+d, a+2d, a+3d
                # Already handled a+3d case above
                # a+2d case: a = x-1-2d, a+3d = x-1+d
                if x - 1 + d < len(colors):
                    idx_a = x - 1 - 2 * d
                    if idx_a >= 0:
                        idx_b = x - 1 - d
                        idx_e = x - 1 + d
                        if idx_e < len(colors):
                            cols = {color, colors[idx_a], colors[idx_b], colors[idx_e]}
                            if len(cols) < 3:
                                ok = False
                                break
                # a+d case: a = x-1-d, a+2d = x-1+d, a+3d = x-1+2d
                if x - 1 + 2 * d < len(colors):
                    idx_a = x - 1 - d
                    if idx_a >= 0:
                        idx_c = x - 1 + d
                        idx_e = x - 1 + 2 * d
                        if idx_e < len(colors):
                            cols = {color, colors[idx_a], colors[idx_c], colors[idx_e]}
                            if len(cols) < 3:
                                ok = False
                                break
                # a case: a = x-1, a+d, a+2d, a+3d
                if x - 1 + 3 * d < len(colors):
                    idx_b = x - 1 + d
                    idx_c = x - 1 + 2 * d
                    idx_e = x - 1 + 3 * d
                    if idx_e < len(colors):
                        cols = {color, colors[idx_b], colors[idx_c], colors[idx_e]}
                        if len(cols) < 3:
                            ok = False
                            break
            if ok:
                colors.append(color)
                break
    return colors

def _local_search_3color(colors, all_aps, n, rng, time_limit, max_iter=500000):
    """Local search to minimize violations, trying to reach 0 with 3 colors."""
    start_time = time.time()
    current = list(colors)
    current_violations = _count_violations_fast(current, all_aps)
    
    if current_violations == 0:
        return current
    
    best = list(current)
    best_violations = current_violations
    
    # Precompute: for each AP, which positions are in it
    # For efficiency, when we change position i, we only need to recheck APs containing i
    # Build reverse map: position -> list of AP indices
    pos_to_ap_indices = [[] for _ in range(n)]
    for ap_idx, (a, b, c, e) in enumerate(all_aps):
        pos_to_ap_indices[a].append(ap_idx)
        pos_to_ap_indices[b].append(ap_idx)
        pos_to_ap_indices[c].append(ap_idx)
        pos_to_ap_indices[e].append(ap_idx)
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random position to change
        i = rng.randrange(n)
        old_color = current[i]
        
        # Try changing to a different color (0, 1, or 2)
        new_color = rng.randint(0, 2)
        if new_color == old_color:
            continue
        
        # Calculate change in violations
        # Only APs containing position i can change their violation status
        delta = 0
        for ap_idx in pos_to_ap_indices[i]:
            a, b, c, e = all_aps[ap_idx]
            # Old colors
            old_cols = {current[a], current[b], current[c], current[e]}
            old_bad = len(old_cols) < 3
            # New colors (only position i changes)
            new_cols = set(old_cols)
            # Remove old_color from set if it's no longer present
            new_list = [current[a], current[b], current[c], current[e]]
            pos_idx_in_ap = [a, b, c, e].index(i)
            new_list[pos_idx_in_ap] = new_color
            new_set = set(new_list)
            new_bad = len(new_set) < 3
            
            if old_bad and not new_bad:
                delta -= 1
            elif not old_bad and new_bad:
                delta += 1
        
        if delta < 0:
            # Improvement
            current[i] = new_color
            current_violations += delta
            if current_violations < best_violations:
                best_violations = current_violations
                best = list(current)
            if best_violations == 0:
                return best
        elif delta == 0:
            # Neutral move - accept with some probability
            if rng.random() < 0.3:
                current[i] = new_color
        # If delta > 0, reject
        
    if best_violations < current_violations:
        return best
    return current

def _simulated_annealing_3color(n, rng, time_limit):
    """Simulated annealing to find a 3-coloring with 0 violations."""
    all_aps, pos_to_aps, _ = _precompute_ap_data(n)
    
    # Initialize with random 3-coloring
    current = [rng.randint(0, 2) for _ in range(n)]
    current_violations = _count_violations_fast(current, all_aps)
    
    best = list(current)
    best_violations = current_violations
    
    # Precompute reverse map
    pos_to_ap_indices = [[] for _ in range(n)]
    for ap_idx, (a, b, c, e) in enumerate(all_aps):
        pos_to_ap_indices[a].append(ap_idx)
        pos_to_ap_indices[b].append(ap_idx)
        pos_to_ap_indices[c].append(ap_idx)
        pos_to_ap_indices[e].append(ap_idx)
    
    start_time = time.time()
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        # Pick random position
        i = rng.randrange(n)
        old_color = current[i]
        new_color = rng.randint(0, 2)
        if new_color == old_color:
            continue
        
        # Calculate delta
        delta = 0
        for ap_idx in pos_to_ap_indices[i]:
            a, b, c, e = all_aps[ap_idx]
            old_cols = {current[a], current[b], current[c], current[e]}
            old_bad = len(old_cols) < 3
            
            new_list = [current[a], current[b], current[c], current[e]]
            pos_idx_in_ap = [a, b, c, e].index(i)
            new_list[pos_idx_in_ap] = new_color
            new_set = set(new_list)
            new_bad = len(new_set) < 3
            
            if old_bad and not new_bad:
                delta -= 1
            elif not old_bad and new_bad:
                delta += 1
        
        # Accept or reject
        if delta <= 0:
            current[i] = new_color
            current_violations += delta
            if current_violations < best_violations:
                best_violations = current_violations
                best = list(current)
                if best_violations == 0:
                    return best
        else:
            # Accept with probability exp(-delta/T)
            import math
            if rng.random() < math.exp(-delta / T):
                current[i] = new_color
                current_violations += delta
        
        T *= cooling
    
    if best_violations < current_violations:
        return best
    return current

def _greedy_with_backtrack(n, rng, time_limit=5.0):
    """Greedy assignment with limited backtracking."""
    all_aps, pos_to_aps, _ = _precompute_ap_data(n)
    
    # Build constraint graph: for each position, which other positions share an AP
    # and what the constraint is
    # Actually, let's use a different approach: assign positions in order,
    # and for each position, try colors 0, 1, 2 in order, checking constraints
    
    colors = [-1] * n
    backtrack_limit = 100000
    backtrack_count = [0]
    
    def assign(pos):
        if time.time() - start_time > time_limit:
            return False
        if pos == n:
            return True
        
        # Try colors in random order
        color_order = [0, 1, 2]
        rng.shuffle(color_order)
        
        for color in color_order:
            if backtrack_count[0] > backtrack_limit:
                return False
            backtrack_count[0] += 1
            
            # Check if assigning color to pos is valid
            valid = True
            for ap_idx, pos_in_ap in pos_to_aps[pos]:
                a, b, c, e = all_aps[ap_idx]
                # Check if all other positions in this AP are assigned
                positions = [a, b, c, e]
                all_assigned = True
                for j, p in enumerate(positions):
                    if p == pos:
                        continue
                    if colors[p] == -1:
                        all_assigned = False
                        break
                if all_assigned:
                    cols = set()
                    for j, p in enumerate(positions):
                        if p == pos:
                            cols.add(color)
                        else:
                            cols.add(colors[p])
                    if len(cols) < 3:
                        valid = False
                        break
            
            if valid:
                colors[pos] = color
                if assign(pos + 1):
                    return True
                colors[pos] = -1
        
        return False
    
    start_time = time.time()
    if assign(0):
        return list(colors)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    
    # Get seed fallback as initial best
    seed_colors = _seed_fallback()
    best = seed_colors
    best_count = len(set(seed_colors))
    
    # Precompute AP data
    all_aps, pos_to_aps, _ = _precompute_ap_data(n)
    
    # Try simulated annealing with multiple restarts
    for restart in range(100):
        if time.time() > deadline - 15:
            break
        
        rng = random.Random(restart * 7919 + 1)
        time_limit = min(2.0, max(0.5, deadline - time.time() - 15))
        
        result = _simulated_annealing_3color(n, rng, time_limit)
        if result is not None:
            if _validate(result):
                count = len(set(result))
                if count < best_count:
                    best = result
                    best_count = count
                    if best_count == 3:
                        return best
    
    # Try greedy with backtracking
    if time.time() < deadline - 10:
        for seed in range(50):
            if time.time() > deadline - 10:
                break
            rng = random.Random(seed * 104729 + 42)
            time_limit = min(3.0, max(0.5, deadline - time.time() - 10))
            result = _greedy_with_backtrack(n, rng, time_limit)
            if result is not None:
                if _validate(result):
                    count = len(set(result))
                    if count < best_count:
                        best = result
                        best_count = count
                        if best_count == 3:
                            return best
    
    # Try local search from the current best
    if time.time() < deadline - 5 and best_count > 3:
        rng = random.Random(12345)
        time_limit = min(3.0, max(0.5, deadline - time.time() - 5))
        # Map best colors to 0,1,2 range if using more than 3 colors
        color_map = {}
        mapped = []
        for c in best:
            if c not in color_map:
                color_map[c] = len(color_map)
            mapped.append(color_map[c])
        # Only proceed if we have exactly 3 colors
        if len(color_map) == 3:
            result = _local_search_3color(mapped, all_aps, n, rng, time_limit)
            if result is not None:
                if _validate(result):
                    count = len(set(result))
                    if count < best_count:
                        best = result
                        best_count = count
    
    return best

