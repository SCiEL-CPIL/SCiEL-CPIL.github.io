"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N = 128
    
    # Precompute all 4-term arithmetic progressions
    all_taps = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 1 - 3 * d
        for a in range(max_a + 1):
            all_taps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_taps = len(all_taps)
    
    # Map each position to the indices of taps it participates in
    pos_taps = [[] for _ in range(N)]
    for i, tap in enumerate(all_taps):
        for p in tap:
            pos_taps[p].append(i)
    
    # Precompute for each position, the list of (tap_index, other_positions) for fast delta computation
    pos_tap_info = [[] for _ in range(N)]
    for ti, tap in enumerate(all_taps):
        for p in tap:
            others = [q for q in tap if q != p]
            pos_tap_info[p].append((ti, others))
    
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
    
    def compute_energy(colors, weight=1000):
        """
        Compute energy: sum of squared violations + large weight * number of distinct colors.
        This allows temporarily invalid colorings during SA.
        """
        violations = 0
        for a, b, c, d in all_taps:
            distinct = len({colors[a], colors[b], colors[c], colors[d]})
            if distinct < 3:
                violations += 1
        num_colors = len(set(colors))
        return violations * violations + weight * num_colors
    
    def hamming_weight_coloring():
        """
        Deterministic construction using Hamming weight of binary representation.
        Position i gets color = popcount(i), which ranges from 0 to 7.
        """
        colors = [bin(i).count('1') for i in range(N)]
        if is_valid_coloring(colors):
            return colors
        return None
    
    def greedy_construct(max_colors=8):
        """
        Greedy left-to-right construction using bitmask for fast color selection.
        Checks constraints where pos is the 4th term (most restrictive for left-to-right).
        Also checks where pos is the 3rd term to prevent future conflicts.
        """
        colors = [-1] * N
        
        for pos in range(N):
            forbidden_mask = 0
            stuck = False
            
            # 1. Constraints where pos is the 4th term: (pos-3d, pos-2d, pos-d, pos)
            # All previous positions are assigned.
            max_d4 = pos // 3
            for d in range(1, max_d4 + 1):
                c0 = colors[pos-3*d]
                c1 = colors[pos-2*d]
                c2 = colors[pos-d]
                s = {c0, c1, c2}
                if len(s) == 1:
                    # Impossible state if we maintain validity, but handle gracefully
                    stuck = True
                    break
                elif len(s) == 2:
                    # If we pick a color from s, we have at most 2 distinct colors.
                    # We need >= 3 distinct. So we must pick a color NOT in s.
                    for c in s:
                        forbidden_mask |= (1 << c)
                # If len(s) == 3, any color is fine (even one from s, total distinct becomes 3 or 4)
            
            if stuck:
                return None
            
            # 2. Constraints where pos is the 3rd term: (pos-2d, pos-d, pos, pos+d)
            # pos+d is unassigned.
            # If c[pos-2d] == c[pos-d] == X, then to get >= 3 distinct in the full AP,
            # c[pos] and c[pos+d] must provide 2 more distinct colors not equal to X?
            # Actually: {X, X, c_pos, c_next}. Distinct count = |{X, c_pos, c_next}|.
            # We need this to be >= 3.
            # So {c_pos, c_next} must contain at least one element not in {X}.
            # If c_pos == X, then c_next MUST be != X.
            # This doesn't forbid c_pos == X immediately, but it creates a constraint on c_next.
            # However, if we choose c_pos == X, we restrict future choices.
            # If we choose c_pos != X, we satisfy the condition regardless of c_next (as long as c_next is valid for other constraints).
            # Wait, if c_pos != X, distinct is |{X, c_pos, c_next}|.
            # If c_next == X, distinct is |{X, c_pos}| = 2. INVALID.
            # So if c_pos != X, c_next MUST be != X.
            # If c_pos == X, c_next MUST be != X.
            # In both cases, c_next MUST be != X.
            # So the constraint is: c_next != X.
            # This doesn't forbid any value for c_pos directly, but it's a "soft" constraint.
            # However, if we have multiple such constraints, it might get tight.
            # For now, we don't add to forbidden_mask, but we could if we want to be conservative.
            # Let's stick to hard constraints for the greedy part.
            
            # 3. Constraints where pos is the 2nd term: (pos-d, pos, pos+d, pos+2d)
            # pos-d is assigned. pos, pos+d, pos+2d unassigned.
            # No immediate hard constraint on c_pos from this alone.
            
            # 4. Constraints where pos is the 1st term: (pos, pos+d, pos+2d, pos+3d)
            # All others unassigned. No constraint.
            
            # Find smallest available color
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
        for ti, others in pos_tap_info[pos]:
            vals = [c]
            for p in others:
                vals.append(new_colors[p])
            
            if len(set(vals)) < 3:
                return False
        return True
    
    def compute_violation_delta(pos, old_color, new_color, colors):
        """Compute change in violation count when changing pos from old_color to new_color."""
        delta = 0
        for ti, others in pos_tap_info[pos]:
            # Old state
            old_vals = [old_color] + [colors[p] for p in others]
            old_distinct = len(set(old_vals))
            old_viol = 1 if old_distinct < 3 else 0
            
            # New state
            new_vals = [new_color] + [colors[p] for p in others]
            new_distinct = len(set(new_vals))
            new_viol = 1 if new_distinct < 3 else 0
            
            delta += new_viol - old_viol
        return delta

    def simulated_annealing(initial_colors, max_colors, time_limit=20.0):
        """
        Simulated annealing to find valid coloring with at most max_colors colors.
        Energy = violations^2 + 1000 * num_distinct_colors
        Allows temporarily invalid states to escape local minima.
        """
        start_t = time.time()
        colors = initial_colors[:]
        current_energy = compute_energy(colors)
        best_colors = colors[:]
        best_energy = current_energy
        
        # Temperature schedule
        T_start = 50.0
        T_end = 0.1
        num_iterations = 50000
        cooling_rate = (T_end / T_start) ** (1.0 / num_iterations)
        
        T = T_start
        for it in range(num_iterations):
            if time.time() - start_t > time_limit:
                break
            
            # Choose random position and new color
            pos = random.randint(0, N - 1)
            new_color = random.randint(0, max_colors - 1)
            old_color = colors[pos]
            
            if new_color == old_color:
                T *= cooling_rate
                continue
            
            # Compute energy delta efficiently
            # We need old_viol, new_viol, old_num_colors, new_num_colors
            # Using delta for violations is fast.
            v_delta = compute_violation_delta(pos, old_color, new_color, colors)
            
            # Count violations once for baseline? No, we need the actual count to square it.
            # Let's maintain a running violation count?
            # For simplicity in this refactor, we'll do a slightly more expensive check
            # or just compute full energy if we want to be safe, but that's slow.
            # Let's try to maintain current_violations.
            
            # Actually, let's just use the full energy calculation for correctness first,
            # then optimize if needed. But 50k iterations * 128 positions * ~10 taps each = 64M ops.
            # Python is slow. Let's stick to the delta approach for violations.
            
            # We need current_violations to compute new_violations = current_violations + v_delta
            # Then energy = v^2 + 1000 * nc.
            
            # Let's re-initialize with a running violation count.
            # This requires careful bookkeeping.
            
            # For now, let's just do the full check but limit iterations if it's too slow.
            # Or better: just use the delta for violations and assume we have current_violations.
            
            # Let's refactor SA to maintain state.
            pass # Placeholder, will rewrite below

    def simulated_annealing_v2(initial_colors, max_colors, time_limit=20.0):
        """
        Optimized SA maintaining running violation count.
        """
        start_t = time.time()
        colors = initial_colors[:]
        current_violations = count_violations(colors)
        current_num_colors = len(set(colors))
        current_energy = current_violations * current_violations + 1000 * current_num_colors
        
        best_colors = colors[:]
        best_energy = current_energy
        best_violations = current_violations
        best_num_colors = current_num_colors
        
        # Temperature schedule
        T_start = 50.0
        T_end = 0.1
        num_iterations = 100000  # More iterations with faster delta
        cooling_rate = (T_end / T_start) ** (1.0 / num_iterations)
        
        T = T_start
        for it in range(num_iterations):
            if time.time() - start_t > time_limit:
                break
            
            pos = random.randint(0, N - 1)
            new_color = random.randint(0, max_colors - 1)
            old_color = colors[pos]
            
            if new_color == old_color:
                T *= cooling_rate
                continue
            
            # Compute violation delta
            v_delta = compute_violation_delta(pos, old_color, new_color, colors)
            new_violations = current_violations + v_delta
            
            # Compute color count change
            # If old_color is no longer used, num_colors decreases by 1
            # If new_color is new, num_colors increases by 1
            # If both are used elsewhere, no change
            # If old_color == new_color (handled above), no change
            
            # Check if old_color is used elsewhere
            old_color_count = 0
            new_color_count = 0
            # This is O(N) which is slow.
            # Let's maintain counts.
            
            # For efficiency, let's just recompute set count if we accept?
            # Or maintain a count array.
            
            # Let's maintain a count array for colors.
            # This is better.
            
            # Refactor again to use count array.
            pass

    def simulated_annealing_v3(initial_colors, max_colors, time_limit=20.0):
        """
        SA with count array for O(1) color count updates.
        """
        start_t = time.time()
        colors = initial_colors[:]
        
        # Count occurrences of each color
        color_counts = [0] * max_colors
        for c in colors:
            if c < max_colors:
                color_counts[c] += 1
        
        current_violations = count_violations(colors)
        current_num_colors = sum(1 for c in color_counts if c > 0)
        current_energy = current_violations * current_violations + 1000 * current_num_colors
        
        best_colors = colors[:]
        best_energy = current_energy
        
        T_start = 50.0
        T_end = 0.1
        num_iterations = 200000
        cooling_rate = (T_end / T_start) ** (1.0 / num_iterations)
        
        T = T_start
        for it in range(num_iterations):
            if time.time() - start_t > time_limit:
                break
            
            pos = random.randint(0, N - 1)
            new_color = random.randint(0, max_colors - 1)
            old_color = colors[pos]
            
            if new_color == old_color:
                T *= cooling_rate
                continue
            
            # Violation delta
            v_delta = compute_violation_delta(pos, old_color, new_color, colors)
            new_violations = current_violations + v_delta
            
            # Color count delta
            nc_delta = 0
            if color_counts[old_color] == 1:
                nc_delta -= 1  # old color disappears
            if color_counts[new_color] == 0:
                nc_delta += 1  # new color appears
            
            new_num_colors = current_num_colors + nc_delta
            new_energy = new_violations * new_violations + 1000 * new_num_colors
            
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                colors[pos] = new_color
                color_counts[old_color] -= 1
                color_counts[new_color] += 1
                
                current_violations = new_violations
                current_num_colors = new_num_colors
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors[:]
                    
                    # If valid and fewer colors, we can potentially stop early or continue
                    if current_violations == 0:
                        # Valid coloring
                        pass
            
            T *= cooling_rate
        
        return best_colors, best_energy

    def try_remove_color(colors, target_color, time_limit=3.0):
        """
        Attempts to remove target_color by recoloring its occurrences to other existing colors.
        Uses backtracking with MRV (Minimum Remaining Values) heuristic.
        """
        start_t = time.time()
        positions = [i for i, c in enumerate(colors) if c == target_color]
        if not positions:
            return None
        
        available_colors = [c for c in set(colors) if c != target_color]
        if not available_colors:
            return None
        
        new_colors = colors[:]
        
        # Sort positions by degree (number of taps) descending to fail fast
        positions.sort(key=lambda i: len(pos_taps[i]), reverse=True)
        
        def backtrack(idx):
            if time.time() - start_t > time_limit:
                return False
            
            if idx == len(positions):
                return True
            
            pos = positions[idx]
            
            # Try available colors
            for c in available_colors:
                new_colors[pos] = c
                
                if is_local_valid(pos, c, new_colors):
                    if backtrack(idx + 1):
                        return True
                
                new_colors[pos] = target_color
            
            return False
        
        if backtrack(0):
            return new_colors
        return None

    def local_search_polish(colors, time_limit=10.0):
        """
        Try to reduce color count by merging pairs of colors.
        """
        start_t = time.time()
        current = colors[:]
        
        while time.time() - start_t < time_limit:
            colors_in_use = list(set(current))
            if len(colors_in_use) <= 2:
                break
            
            colors_in_use.sort(key=lambda c: current.count(c))
            
            improved = False
            for i in range(len(colors_in_use)):
                for j in range(i + 1, len(colors_in_use)):
                    if time.time() - start_t >= time_limit:
                        break
                    
                    c1, c2 = colors_in_use[i], colors_in_use[j]
                    
                    # Try to merge c2 into c1 (recolor all c2 to c1)
                    # This is equivalent to try_remove_color with target c2, but restricted to c1?
                    # No, try_remove_color allows any other color.
                    # Let's just use try_remove_color for the least frequent color.
                    
                    # Actually, local_search_polish in the original code tried SA merges.
                    # Let's stick to try_remove_color for simplicity and speed.
                    pass
                
                if improved or time.time() - start_t >= time_limit:
                    break
            
            if not improved:
                break
        
        return current

    # 1. Generate initial coloring
    initial_colors = hamming_weight_coloring()
    
    if initial_colors is None:
        # Try greedy with small number of colors
        for k in range(3, 15):
            initial_colors = greedy_construct(k)
            if initial_colors is not None:
                break
    
    if initial_colors is None:
        # Fallback: random valid coloring via SA
        initial_colors = [random.randint(0, 7) for _ in range(N)]
        res, _ = simulated_annealing_v3(initial_colors, 8, time_limit=10.0)
        if is_valid_coloring(res):
            initial_colors = res
        else:
            # Last resort: identity mapping (128 colors)
            initial_colors = list(range(N))
    
    current_colors = initial_colors
    current_count = get_distinct_count(current_colors)
    
    # 2. Use SA to try to find valid coloring with fewer colors
    if time.time() < deadline - 20:
        # Try SA with decreasing max_colors
        for max_c in range(3, min(current_count + 1, 10)):
            if time.time() > deadline - 15:
                break
            
            # Create random initial coloring with max_c colors
            sa_init = [random.randint(0, max_c - 1) for _ in range(N)]
            
            sa_result, sa_energy = simulated_annealing_v3(sa_init, max_c, time_limit=15.0)
            
            # Check if we got a valid coloring
            if is_valid_coloring(sa_result):
                sa_count = len(set(sa_result))
                if sa_count < current_count:
                    current_colors = sa_result
                    current_count = sa_count
                    break
                elif sa_count == current_count:
                    # Same count, but maybe better structure for further reduction
                    if count_violations(sa_result) == 0:
                        current_colors = sa_result
    
    # 3. Local search to reduce color count further
    if time.time() < deadline - 10:
        # Try removing individual colors
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            colors_in_use = list(set(current_colors))
            colors_in_use.sort(key=lambda c: current_colors.count(c))
            
            for c in colors_in_use:
                if time.time() > deadline - 8:
                    break
                
                new_colors = try_remove_color(current_colors, c, time_limit=3.0)
                if new_colors is not None:
                    new_count = get_distinct_count(new_colors)
                    if new_count < current_count:
                        current_colors = new_colors
                        current_count = new_count
                        improved = True
                        break
    
    # Final validation
    if not is_valid_coloring(current_colors):
        # If something went wrong, fall back to a known valid coloring
        # Try greedy again
        fb = greedy_construct(8)
        if fb is not None:
            current_colors = fb
        else:
            current_colors = [i % 4 for i in range(N)]
            if not is_valid_coloring(current_colors):
                current_colors = list(range(N))
    
    return current_colors
# EVOLVE-BLOCK-END
