import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute AP constraints
    # For each x (1-based), list of (i1, i2, i3) 0-based indices
    # such that x-1 is the last element of a 4-AP.
    constraints = [[] for _ in range(128)]
    for x in range(1, 129):
        idx = x - 1
        for d in range(1, (x - 1) // 3 + 1):
            a = x - 3 * d
            if a < 1:
                break
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            constraints[idx].append((i1, i2, i3))
    
    # Precompute which positions constrain each position (for forward checking)
    # For position i, which future positions j > i have i in their constraints?
    constrained_by = [[] for _ in range(128)]
    for j in range(128):
        for i1, i2, i3 in constraints[j]:
            for i in (i1, i2, i3):
                if i < j:
                    constrained_by[i].append(j)

    def is_valid_assignment(colors, idx, c):
        """Check if assigning color c to position idx is valid given current colors."""
        for i1, i2, i3 in constraints[idx]:
            c1 = colors[i1]
            c2 = colors[i2]
            c3 = colors[i3]
            # If any of c1, c2, c3 is unassigned, we can't fully validate yet
            # But we can check: if all three are assigned, then {c1,c2,c3,c} must have >= 3 distinct
            if c1 != -1 and c2 != -1 and c3 != -1:
                if len({c1, c2, c3, c}) < 3:
                    return False
        return True

    def search_backtracking(max_c, time_limit):
        colors = [-1] * 128
        bt_start = time.time()
        
        # Variable ordering: use most-constrained-first heuristic
        # Precompute constraint count for each position
        constraint_counts = [len(constraints[i]) for i in range(128)]
        
        # Order positions by constraint count (descending) for more constrained first
        # But we must respect the partial order: position i can only be assigned after
        # all positions in constraints[i] are assigned
        # So we need a topological ordering that respects dependencies
        
        # Build dependency graph: i depends on all j in constraints[i]
        # Use a priority queue approach: always pick the most constrained unassigned
        # position that has all its dependencies satisfied
        
        def get_valid_order():
            """Get a valid assignment order using most-constrained-first heuristic."""
            assigned = [False] * 128
            order = []
            remaining = set(range(128))
            
            while remaining:
                # Find the most constrained unassigned position whose deps are all assigned
                best_idx = None
                best_count = -1
                for idx in remaining:
                    # Check if all dependencies are assigned
                    deps_satisfied = True
                    for i1, i2, i3 in constraints[idx]:
                        if not assigned[i1] and not assigned[i2] and not assigned[i3]:
                            # At least one dep is unassigned - but we need ALL deps assigned
                            # Actually, we need each individual dep to be assigned
                            if not assigned[i1]:
                                deps_satisfied = False
                                break
                            if not assigned[i2]:
                                deps_satisfied = False
                                break
                            if not assigned[i3]:
                                deps_satisfied = False
                                break
                    if deps_satisfied:
                        if constraint_counts[idx] > best_count:
                            best_count = constraint_counts[idx]
                            best_idx = idx
                
                if best_idx is None:
                    # No position has all deps satisfied - shouldn't happen with 0..127
                    # Fall back to sequential order
                    best_idx = min(remaining)
                
                order.append(best_idx)
                assigned[best_idx] = True
                remaining.remove(best_idx)
            
            return order
        
        order = get_valid_order()
        
        def backtrack(depth):
            if time.time() > bt_start + time_limit:
                return False
            
            if depth == 128:
                return True
            
            idx = order[depth]
            
            # Try colors in a smart order: prefer colors that appear more often
            # in the constraints of this position
            color_freq = [0] * max_c
            for i1, i2, i3 in constraints[idx]:
                for ci in (i1, i2, i3):
                    if colors[ci] != -1:
                        color_freq[colors[ci]] += 1
            
            # Sort colors by frequency (most frequent first)
            color_order = sorted(range(max_c), key=lambda c: -color_freq[c])
            
            for c in color_order:
                if is_valid_assignment(colors, idx, c):
                    colors[idx] = c
                    
                    # Forward checking: verify that all future positions still have
                    # at least one valid color
                    consistent = True
                    for future_idx in constrained_by[idx]:
                        if colors[future_idx] == -1:
                            has_valid = False
                            for fc in range(max_c):
                                if is_valid_assignment(colors, future_idx, fc):
                                    has_valid = True
                                    break
                            if not has_valid:
                                consistent = False
                                break
                    
                    if consistent and backtrack(depth + 1):
                        return True
                    
                    colors[idx] = -1
            
            return False

        if backtrack(0):
            return colors
        return None

    # Try to find a 3-coloring with improved backtracking
    # Give it more time: 20 seconds
    res = search_backtracking(3, 20.0)
    if res:
        return res

    # Fallback: Greedy with 4 colors (seed logic)
    def try_construct_greedy(max_c):
        colors = [None] * 128
        used = set()
        for x in range(128):
            if time.time() > deadline:
                return None
            valid_colors = []
            for c in range(max_c):
                ok = True
                for i1, i2, i3 in constraints[x]:
                    c1, c2, c3 = colors[i1], colors[i2], colors[i3]
                    if len({c1, c2, c3, c}) < 3:
                        ok = False
                        break
                if ok:
                    valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            # Heuristic: prefer used colors to minimize distinct count if max_c > needed
            # But here we fix max_c=4. If we use 3, great. If we use 4, ok.
            # Sorting by (0 if used else 1, c) encourages reusing colors.
            valid_colors.sort(key=lambda c: (0 if c in used else 1, c))
            chosen = valid_colors[0]
            colors[x] = chosen
            used.add(chosen)
            
        return colors

    res = try_construct_greedy(4)
    if res:
        return res

    # Fallback: Seed's original loop (slower, but robust)
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
