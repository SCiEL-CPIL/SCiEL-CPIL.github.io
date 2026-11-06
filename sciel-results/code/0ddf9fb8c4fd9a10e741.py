"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    N = 128
    
    # Precompute all 4-term arithmetic progressions
    all_taps = []
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 1 - 3 * d
        for a in range(max_a + 1):
            all_taps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    n_taps = len(all_taps)
    
    # Map each position to the indices of taps it participates in
    pos_taps = [[] for _ in range(N)]
    for i, tap in enumerate(all_taps):
        for p in tap:
            pos_taps[p].append(i)
    
    def count_violations(colors):
        """Count number of 4-APs with fewer than 3 distinct colors."""
        violations = 0
        for a, b, c, d in all_taps:
            cs = {colors[a], colors[b], colors[c], colors[d]}
            if len(cs) < 3:
                violations += 1
        return violations
    
    def is_valid(colors):
        return count_violations(colors) == 0
    
    def distinct_count(colors):
        return len(set(colors))
    
    def compute_objective(colors):
        """Objective: violations * 1000 + distinct_count"""
        viol = count_violations(colors)
        dist = distinct_count(colors)
        return viol * 1000 + dist, viol, dist
    
    def fast_delta(pos, old_color, new_color, colors, color_counts):
        """
        Calculate change in (violations, distinct_count) when changing pos.
        Returns (delta_viol, delta_distinct)
        """
        if old_color == new_color:
            return 0, 0
        
        # Delta violations
        delta_viol = 0
        for ti in pos_taps[pos]:
            a, b, c, d = all_taps[ti]
            vals_before = []
            vals_after = []
            for p in [a, b, c, d]:
                if p == pos:
                    vals_before.append(old_color)
                    vals_after.append(new_color)
                else:
                    vals_before.append(colors[p])
                    vals_after.append(colors[p])
            
            before_bad = len(set(vals_before)) < 3
            after_bad = len(set(vals_after)) < 3
            
            if before_bad and not after_bad:
                delta_viol -= 1
            elif not before_bad and after_bad:
                delta_viol += 1
        
        # Delta distinct count
        delta_distinct = 0
        # old_color count decreases by 1
        if color_counts[old_color] == 1:
            delta_distinct -= 1  # old_color disappears
        # new_color count increases by 1
        if color_counts[new_color] == 0:
            delta_distinct += 1  # new_color appears
        
        return delta_viol, delta_distinct
    
    def simulated_annealing(initial_colors, max_iter=200000, time_limit=40.0):
        """
        Simulated annealing to minimize objective = violations * 1000 + distinct_count
        """
        start_t = time.time()
        current = initial_colors[:]
        
        # Compute color counts
        color_counts = {}
        for c in current:
            color_counts[c] = color_counts.get(c, 0) + 1
        
        curr_obj, curr_viol, curr_dist = compute_objective(current)
        best = current[:]
        best_obj, best_viol, best_dist = curr_obj, curr_viol, curr_dist
        
        # Temperature schedule
        temp_start = 50.0
        temp_end = 0.1
        cooling = (temp_end / temp_start) ** (1.0 / max_iter)
        temp = temp_start
        
        for iteration in range(max_iter):
            if time.time() - start_t > time_limit:
                break
            if time.time() > deadline - 5:
                break
            
            # Pick a random position
            pos = random.randint(0, N - 1)
            old_color = current[pos]
            
            # Pick a new color: either an existing color or a new one
            colors_in_use = list(color_counts.keys())
            if random.random() < 0.7:
                # Use an existing color
                new_color = random.choice(colors_in_use)
            else:
                # Try a new color (to allow exploring different color sets)
                new_color = random.randint(0, max(colors_in_use) + 2) if colors_in_use else 0
            
            if new_color == old_color:
                continue
            
            delta_viol, delta_distinct = fast_delta(pos, old_color, new_color, current, color_counts)
            delta_obj = delta_viol * 1000 + delta_distinct
            
            if delta_obj <= 0:
                # Accept
                current[pos] = new_color
                color_counts[old_color] -= 1
                if color_counts[old_color] == 0:
                    del color_counts[old_color]
                color_counts[new_color] = color_counts.get(new_color, 0) + 1
                
                curr_obj += delta_obj
                curr_viol += delta_viol
                curr_dist += delta_distinct
                
                if curr_obj < best_obj:
                    best_obj = curr_obj
                    best_viol = curr_viol
                    best_dist = curr_dist
                    best = current[:]
            else:
                # Accept with probability exp(-delta/temp)
                if random.random() < np.exp(-delta_obj / temp):
                    current[pos] = new_color
                    color_counts[old_color] -= 1
                    if color_counts[old_color] == 0:
                        del color_counts[old_color]
                    color_counts[new_color] = color_counts.get(new_color, 0) + 1
                    
                    curr_obj += delta_obj
                    curr_viol += delta_viol
                    curr_dist += delta_distinct
            
            temp *= cooling
        
        return best, best_viol, best_dist
    
    # Get an initial coloring
    # Start with a simple pattern
    initial = [i % 4 for i in range(N)]
    if not is_valid(initial):
        # Try different patterns
        initial = [i % 5 for i in range(N)]
        if not is_valid(initial):
            initial = [i % 6 for i in range(N)]
            if not is_valid(initial):
                # Use greedy
                initial = None
                for k in range(4, 12):
                    colors = [-1] * N
                    valid = True
                    for pos in range(N):
                        forbidden = set()
                        stuck = False
                        for ti in pos_taps[pos]:
                            a, b, c, d = all_taps[ti]
                            others = [p for p in [a, b, c, d] if p != pos]
                            assigned = []
                            all_assigned = True
                            for p in others:
                                if colors[p] == -1:
                                    all_assigned = False
                                    break
                                assigned.append(colors[p])
                            if not all_assigned:
                                continue
                            s = set(assigned)
                            if len(s) == 1:
                                stuck = True
                                break
                            elif len(s) == 2:
                                forbidden.update(s)
                        if stuck:
                            valid = False
                            break
                        for c in range(k):
                            if c not in forbidden:
                                colors[pos] = c
                                break
                        else:
                            valid = False
                            break
                    if valid and is_valid(colors):
                        initial = colors
                        break
                if initial is None:
                    initial = list(range(N))
    
    # Run simulated annealing
    best_result, best_viol, best_dist = simulated_annealing(initial, max_iter=300000, time_limit=50.0)
    
    # If we found a valid coloring, try to polish it
    if best_viol == 0 and time.time() < deadline - 10:
        # Try to reduce colors further with a focused search
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            # Try eliminating the least frequent color
            color_counts = {}
            for c in best_result:
                color_counts[c] = color_counts.get(c, 0) + 1
            
            if len(color_counts) <= 3:
                break
            
            # Find least frequent color
            min_count = min(color_counts.values())
            target_colors = [c for c, cnt in color_counts.items() if cnt == min_count]
            target_color = random.choice(target_colors)
            positions = [i for i, c in enumerate(best_result) if c == target_color]
            
            # Try to recolor each position to another color
            new_colors = best_result[:]
            success = True
            
            for pos in positions:
                found = False
                other_colors = [c for c in color_counts if c != target_color]
                random.shuffle(other_colors)
                
                for c in other_colors:
                    new_colors[pos] = c
                    if count_violations(new_colors) == 0:
                        found = True
                        break
                    new_colors[pos] = target_color
                
                if not found:
                    success = False
                    break
            
            if success:
                new_distinct = distinct_count(new_colors)
                if new_distinct < best_dist:
                    best_result = new_colors
                    best_dist = new_distinct
    
    return best_result
# EVOLVE-BLOCK-END

