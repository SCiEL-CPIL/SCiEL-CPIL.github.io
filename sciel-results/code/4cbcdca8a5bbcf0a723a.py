import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n = 16
    # Precompute all K4s and their edge indices
    k4s = []
    edge_index = {}
    edge_ij = []  # map edge index -> (i, j)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[(i,j)] = idx
            edge_ij.append((i, j))
            idx += 1
            
    k4_count = 0
    for a,b,c,d in combinations(range(n), 4):
        edges = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
        edge_idxs = [edge_index[e] for e in edges]
        k4s.append(edge_idxs)
        k4_count += 1
        
    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, edge_idxs in enumerate(k4s):
        for ei in edge_idxs:
            edge_to_k4s[ei].append(k4_idx)
            
    def get_violations_and_colors(colors):
        """
        Returns (violation_count, distinct_colors_count).
        violation_count is sum of max(0, 5 - distinct_colors) for each K4.
        """
        violations = 0
        color_set = set()
        for c in colors:
            color_set.add(c)
        distinct_colors = len(color_set)
        
        for edge_idxs in k4s:
            # Count distinct colors in this K4
            # Optimization: if we have 5 distinct colors, we can stop early?
            # No, we need to know if it's < 5.
            # A set of 6 elements is fast enough.
            k4_colors = set(colors[ei] for ei in edge_idxs)
            if len(k4_colors) < 5:
                violations += (5 - len(k4_colors))
                
        return violations, distinct_colors

    def get_local_violation_delta(edge_idx, old_color, new_color, colors):
        """
        Calculate change in violation count if edge_idx changes from old_color to new_color.
        Only checks K4s containing edge_idx.
        Returns delta_violations.
        """
        delta = 0
        for k4_idx in edge_to_k4s[edge_idx]:
            edge_idxs = k4s[k4_idx]
            
            # Calculate distinct colors before
            # We can optimize by just checking the 6 edges
            old_set = set()
            new_set = set()
            for ei in edge_idxs:
                if ei == edge_idx:
                    old_set.add(old_color)
                    new_set.add(new_color)
                else:
                    c = colors[ei]
                    old_set.add(c)
                    new_set.add(c)
            
            old_len = len(old_set)
            new_len = len(new_set)
            
            old_viol = max(0, 5 - old_len)
            new_viol = max(0, 5 - new_len)
            
            delta += (new_viol - old_viol)
            
        return delta

    # Initial coloring: Try to start with a low number of colors
    # (i+j) % 17 is valid but uses 17 colors.
    # Let's try (i+j) % 15? Not necessarily valid.
    # Let's use a random initialization with 15 colors and fix it.
    
    best_colors_count = 17
    best_violations = 0
    best_col_list = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    # Verify initial
    v, c = get_violations_and_colors(best_col_list)
    if v == 0:
        best_violations = 0
        best_colors_count = c
        
    random.seed(42)
    
    # Simulated Annealing / Hill Climbing
    # Objective: Minimize (Violations * 1000 + DistinctColors)
    
    # Try multiple random restarts with different initial color counts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # Initialize with a random number of colors between 13 and 15
        num_colors_init = random.choice([13, 14, 15])
        colors = [random.randint(0, num_colors_init - 1) for _ in range(120)]
        
        # Initial cost
        curr_viol, curr_dist = get_violations_and_colors(colors)
        curr_cost = curr_viol * 1000 + curr_dist
        
        # Local Search
        T = 10.0 # Temperature
        T_min = 0.1
        cooling = 0.99
        
        steps = 0
        max_steps = 50000
        
        while steps < max_steps and time.time() < deadline - 5:
            steps += 1
            
            # Pick a random edge
            edge_idx = random.randint(0, 119)
            old_color = colors[edge_idx]
            
            # Pick a new color
            # Strategy: mostly pick from existing colors to reduce distinct count
            # Sometimes pick a new color if we are stuck in high violations?
            # Let's stick to existing colors to force merging.
            # Get unique colors
            unique_colors = list(set(colors))
            
            if not unique_colors:
                break
                
            # If we have many violations, maybe we need more colors?
            # But the problem asks to minimize colors.
            # If we start with 13 colors and have violations, we might need to increase colors to fix violations?
            # No, the constraint is "at least 5 distinct colors".
            # If a K4 has < 5, we can fix it by changing an edge to a *new* color or a different existing color.
            # So we should allow new colors if it helps reduce violations.
            
            # Probability of picking a new color vs existing
            if curr_viol > 0:
                # If we have violations, try to introduce diversity
                if random.random() < 0.3:
                    new_color = max(unique_colors) + 1
                else:
                    new_color = random.choice(unique_colors)
            else:
                # If no violations, try to reduce colors
                new_color = random.choice(unique_colors)
                
            if new_color == old_color:
                continue
                
            # Calculate delta
            delta_v = get_local_violation_delta(edge_idx, old_color, new_color, colors)
            
            # Calculate delta in distinct colors
            # This is tricky to do locally without checking the whole array.
            # Approximation:
            # If old_color is no longer used, distinct_colors decreases by 1.
            # If new_color was not used, distinct_colors increases by 1.
            
            # To be precise and fast, we can just accept based on violations first,
            # and then do a second phase to reduce colors.
            
            # Let's use a two-phase approach in the loop:
            # Phase 1: Minimize Violations
            # Phase 2: Minimize Distinct Colors
            
            if curr_viol > 0:
                # Accept if violations decrease
                if delta_v < 0:
                    colors[edge_idx] = new_color
                    curr_viol += delta_v
                elif delta_v == 0:
                    # Accept with probability exp(-delta_cost / T) ?
                    # For now, accept to explore
                    if random.random() < 0.5:
                        colors[edge_idx] = new_color
                # else reject
            else:
                # Violations are 0. Try to reduce distinct colors.
                # Check if new_color is already in use
                # We need to know if old_color will be removed and if new_color is new.
                
                # Count usage of old_color and new_color
                # This is O(120) per step. 120 * 50000 = 6M ops. Might be slow in Python.
                # Optimization: Maintain counts.
                
                # Let's switch to maintaining counts.
                pass # Placeholder for logic below

        # Refine the logic with maintained counts for speed
        
    # Re-implementing the search loop with maintained counts for efficiency
    
    def run_search(init_colors, max_time):
        colors = init_colors[:]
        color_counts = {}
        for c in colors:
            color_counts[c] = color_counts.get(c, 0) + 1
            
        curr_viol, curr_dist = get_violations_and_colors(colors)
        curr_cost = curr_viol * 1000 + curr_dist
        
        best_local_viol = curr_viol
        best_local_dist = curr_dist
        best_local_colors = colors[:]
        
        T = 5.0
        T_min = 0.1
        cooling = 0.995
        steps = 0
        
        while time.time() < max_time:
            steps += 1
            if steps % 1000 == 0:
                T *= cooling
                if T < T_min:
                    T = T_min
                    
            edge_idx = random.randint(0, 119)
            old_color = colors[edge_idx]
            
            # Choose new color
            # 80% chance existing color, 20% chance new color (if viol > 0)
            unique_colors = [c for c in color_counts if color_counts[c] > 0]
            
            if curr_viol > 0 and random.random() < 0.2:
                new_color = max(unique_colors) + 1 if unique_colors else 0
            else:
                if not unique_colors:
                    break
                new_color = random.choice(unique_colors)
                
            if new_color == old_color:
                continue
                
            delta_v = get_local_violation_delta(edge_idx, old_color, new_color, colors)
            
            # Determine delta_dist
            delta_dist = 0
            # If old color count becomes 0, dist decreases by 1
            if color_counts[old_color] == 1:
                delta_dist -= 1
            # If new color count was 0, dist increases by 1
            if new_color not in color_counts or color_counts[new_color] == 0:
                delta_dist += 1
                
            delta_cost = delta_v * 1000 + delta_dist
            
            # Acceptance
            if delta_cost <= 0:
                # Accept
                colors[edge_idx] = new_color
                color_counts[old_color] -= 1
                if color_counts[old_color] == 0:
                    del color_counts[old_color]
                color_counts[new_color] = color_counts.get(new_color, 0) + 1
                
                curr_viol += delta_v
                curr_dist += delta_dist
                curr_cost += delta_cost
                
                if curr_viol < best_local_viol or (curr_viol == best_local_viol and curr_dist < best_local_dist):
                    best_local_viol = curr_viol
                    best_local_dist = curr_dist
                    best_local_colors = colors[:]
            else:
                # Metropolis criterion
                prob = 2.71828 ** (-delta_cost / max(T, 0.1))
                if random.random() < prob:
                    colors[edge_idx] = new_color
                    color_counts[old_color] -= 1
                    if color_counts[old_color] == 0:
                        del color_counts[old_color]
                    color_counts[new_color] = color_counts.get(new_color, 0) + 1
                    
                    curr_viol += delta_v
                    curr_dist += delta_dist
                    curr_cost += delta_cost
                    
                    if curr_viol < best_local_viol or (curr_viol == best_local_viol and curr_dist < best_local_dist):
                        best_local_viol = curr_viol
                        best_local_dist = curr_dist
                        best_local_colors = colors[:]
                        
        return best_local_viol, best_local_dist, best_local_colors

    # Run multiple searches with different initializations
    for restart in range(50):
        if time.time() > deadline - 5:
            break
            
        # Initialization strategies
        strategy = restart % 5
        
        if strategy == 0:
            # Modular 17
            init = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        elif strategy == 1:
            # Random 15
            init = [random.randint(0, 14) for _ in range(120)]
        elif strategy == 2:
            # Random 14
            init = [random.randint(0, 13) for _ in range(120)]
        elif strategy == 3:
            # Random 13
            init = [random.randint(0, 12) for _ in range(120)]
        else:
            # Greedy assignment?
            init = [0] * 120
            
        max_time = deadline - 2 - (time.time() - start_time)
        if max_time < 1:
            break
            
        v, d, cols = run_search(init, max_time)
        
        if v == 0 and d < best_colors_count:
            best_colors_count = d
            best_col_list = cols
            best_violations = 0
        elif v == 0 and d == best_colors_count:
            # Keep the first one or random? Keep best found.
            pass
            
        # If we found 13 or 14, we can stop early if we are confident?
        # But let's keep searching to see if we can get 13.
        
    # Final verification
    if best_violations > 0:
        # Fallback to a known valid solution if search failed
        # (i+j)%17 is valid
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        
    return best_col_list
