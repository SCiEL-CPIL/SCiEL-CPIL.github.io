"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s limit
    
    N = 128
    
    # Precompute all 4-term arithmetic progressions
    all_taps = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 1 - 3 * d
        for a in range(max_a + 1):
            all_taps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map each position to the indices of taps it participates in
    pos_taps = [[] for _ in range(N)]
    for i, tap in enumerate(all_taps):
        for p in tap:
            pos_taps[p].append(i)
    
    def is_valid_coloring(colors):
        """Check if every 4-term progression uses at least 3 distinct colors."""
        for a, b, c, d in all_taps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                return False
        return True
    
    def get_distinct_count(colors):
        return len(set(colors))
    
    def count_violations(colors):
        """Count number of 4-term APs with fewer than 3 distinct colors."""
        violations = 0
        for a, b, c, d in all_taps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                violations += 1
        return violations
    
    def hamming_weight_coloring():
        """
        Deterministic construction using Hamming weight of binary representation.
        Position i gets color = popcount(i), which ranges from 0 to 7.
        """
        colors = [bin(i).count('1') for i in range(N)]
        if is_valid_coloring(colors):
            return colors
        return None
    
    def greedy_construct(max_colors):
        """
        Greedy left-to-right construction.
        """
        colors = [-1] * N
        
        for pos in range(N):
            forbidden_mask = 0
            stuck = False
            
            for ti in pos_taps[pos]:
                a, b, c, d = all_taps[ti]
                pts = [a, b, c, d]
                others = [p for p in pts if p != pos]
                
                assigned_colors = []
                all_assigned = True
                for p in others:
                    if colors[p] == -1:
                        all_assigned = False
                        break
                    assigned_colors.append(colors[p])
                
                if not all_assigned:
                    continue
                
                s = set(assigned_colors)
                if len(s) == 1:
                    stuck = True
                    break
                elif len(s) == 2:
                    for c in s:
                        forbidden_mask |= (1 << c)
            
            if stuck:
                return None
            
            found = False
            for c in range(max_colors):
                if not (forbidden_mask & (1 << c)):
                    colors[pos] = c
                    found = True
                    break
            
            if not found:
                return None
        
        if not is_valid_coloring(colors):
            return None
        
        return colors
    
    def is_local_valid(pos, c, new_colors):
        """Check if assigning c to pos is valid with respect to all taps where pos is involved."""
        for ti in pos_taps[pos]:
            a, b, c_tap, d_tap = all_taps[ti]
            pts = [a, b, c_tap, d_tap]
            others = [p for p in pts if p != pos]
            
            vals = [c]
            for p in others:
                vals.append(new_colors[p])
            
            if len(set(vals)) < 3:
                return False
        return True
    
    def incremental_local_search(colors, max_colors, time_limit=5.0):
        """
        Local search with incremental violation counting to find valid coloring
        with at most max_colors distinct colors.
        """
        start_t = time.time()
        current = colors[:]
        current_violations = count_violations(current)
        
        if current_violations == 0:
            return current
        
        best = current[:]
        best_violations = current_violations
        
        # Random restarts with local search
        for restart in range(50):
            if time.time() - start_t > time_limit:
                break
            
            # Start from current best or random
            if restart == 0:
                state = current[:]
            else:
                # Random initialization with max_colors colors
                state = [random.randint(0, max_colors - 1) for _ in range(N)]
            
            state_violations = count_violations(state)
            if state_violations == 0:
                return state
            if state_violations < best_violations:
                best = state[:]
                best_violations = state_violations
            
            # Local search: try to reduce violations
            improved = True
            while improved and time.time() - start_t < time_limit:
                improved = False
                
                # Try changing each position to a different color
                for pos in range(N):
                    if time.time() - start_t > time_limit:
                        break
                    
                    old_color = state[pos]
                    
                    # Try all colors except old_color
                    for new_color in range(max_colors):
                        if new_color == old_color:
                            continue
                        
                        # Calculate violation change
                        delta_violations = 0
                        for ti in pos_taps[pos]:
                            a, b, c, d = all_taps[ti]
                            pts = [a, b, c, d]
                            others = [p for p in pts if p != pos]
                            
                            # Current violation status
                            old_vals = [state[p] for p in pts]
                            old_distinct = len(set(old_vals))
                            old_is_violation = old_distinct < 3
                            
                            # New violation status
                            new_vals = []
                            for p in pts:
                                if p == pos:
                                    new_vals.append(new_color)
                                else:
                                    new_vals.append(state[p])
                            new_distinct = len(set(new_vals))
                            new_is_violation = new_distinct < 3
                            
                            if old_is_violation and not new_is_violation:
                                delta_violations -= 1
                            elif not old_is_violation and new_is_violation:
                                delta_violations += 1
                        
                        if delta_violations < 0:
                            # Accept improvement
                            state[pos] = new_color
                            state_violations += delta_violations
                            improved = True
                            
                            if state_violations == 0:
                                return state
                            
                            if state_violations < best_violations:
                                best = state[:]
                                best_violations = state_violations
                            break
        
        if best_violations < current_violations:
            current = best[:]
            current_violations = best_violations
        
        return current if count_violations(current) == 0 else None
    
    def try_reduce_colors(colors, target_max_colors, time_limit=8.0):
        """
        Try to find a valid coloring using at most target_max_colors colors.
        Uses local search with incremental violations.
        """
        start_t = time.time()
        
        # First, try to recolor to use only colors 0..target_max_colors-1
        # by mapping existing colors to this range
        existing_colors = list(set(colors))
        color_map = {}
        for i, c in enumerate(existing_colors):
            color_map[c] = i % target_max_colors
        
        mapped = [color_map[c] for c in colors]
        
        # Run local search
        result = incremental_local_search(mapped, target_max_colors, time_limit)
        
        if result is not None and is_valid_coloring(result):
            return result
        
        return None
    
    def color_count(colors):
        return len(set(colors))
    
    # 1. Generate initial coloring
    initial_colors = hamming_weight_coloring()
    
    if initial_colors is None:
        for k in range(3, 15):
            initial_colors = greedy_construct(k)
            if initial_colors is not None:
                break
    
    if initial_colors is None:
        initial_colors = [i % 4 for i in range(N)]
        if not is_valid_coloring(initial_colors):
            initial_colors = list(range(N))
    
    current_colors = initial_colors
    current_count = get_distinct_count(current_colors)
    
    # 2. Try to reduce color count using local search
    # Try target counts from low to high
    for target in range(3, current_count + 1):
        if time.time() > deadline - 10:
            break
        
        if target >= current_count:
            break
        
        result = try_reduce_colors(current_colors, target, time_limit=min(8.0, deadline - time.time() - 5))
        
        if result is not None:
            current_colors = result
            current_count = get_distinct_count(current_colors)
            if current_count <= target:
                break
    
    # Final validation
    if not is_valid_coloring(current_colors):
        current_colors = [i % 4 for i in range(N)]
    
    return current_colors
# EVOLVE-BLOCK-END
