import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = 16
    # Generate all pairs (i, j) with i < j
    edges = []
    edge_index = {}
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = len(edges)
            edges.append((i, j))
    
    num_edges = len(edges)  # 120
    
    # Generate all K4s (sets of 4 vertices)
    k4_edges_list = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e1 = edge_index[(a, b)]
                    e2 = edge_index[(a, c)]
                    e3 = edge_index[(a, d)]
                    e4 = edge_index[(b, c)]
                    e5 = edge_index[(b, d)]
                    e6 = edge_index[(c, d)]
                    k4_edges_list.append([e1, e2, e3, e4, e5, e6])
    
    num_k4s = len(k4_edges_list)  # 1820
    
    # Map edge index to list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges_list):
        for e in e_list:
            edge_to_k4s[e].append(k4_idx)
    
    # Base solution: (i + j) % 17
    base_colors = [0] * num_edges
    for idx, (i, j) in enumerate(edges):
        base_colors[idx] = (i + j) % 17
    
    def compact(colors):
        """Map colors to 0..k-1 preserving distinct count."""
        unique = sorted(set(colors))
        mapping = {c: i for i, c in enumerate(unique)}
        return [mapping[c] for c in colors]
    
    def validate(colors):
        """Check if all K4s have >= 5 distinct colors."""
        for e_list in k4_edges_list:
            if len(set(colors[e] for e in e_list)) < 5:
                return False
        return True
    
    def count_bad_k4s(colors):
        """Count K4s with < 5 distinct colors."""
        bad = 0
        for e_list in k4_edges_list:
            if len(set(colors[e] for e in e_list)) < 5:
                bad += 1
        return bad
    
    # Simulated annealing with count-based state
    def solve_with_sa(k, time_limit):
        """Try to find a valid k-coloring using SA with count arrays."""
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Random init with k colors
            colours = [random.randint(0, k - 1) for _ in range(num_edges)]
            
            # Build count arrays: k4_counts[k4_idx][color] = count of edges with that color
            k4_counts = [[0] * k for _ in range(num_k4s)]
            for k4_idx, e_list in enumerate(k4_edges_list):
                for e in e_list:
                    k4_counts[k4_idx][colours[e]] += 1
            
            # Track distinct color count per K4
            k4_distinct = [0] * num_k4s
            for k4_idx in range(num_k4s):
                cnt = k4_counts[k4_idx]
                k4_distinct[k4_idx] = sum(1 for c in cnt if c > 0)
            
            bad_count = sum(1 for d in k4_distinct if d < 5)
            
            if bad_count == 0:
                return colours
            
            best_colours = colours[:]
            best_bad = bad_count
            
            T = 2.0
            cool_rate = 0.9995
            min_T = 0.05
            
            no_improve = 0
            max_no_improve = 5000
            
            while time.time() < end_time and bad_count > 0:
                if T < min_T:
                    break
                
                # Pick a random bad K4
                # Find a bad K4 index
                bad_k4s = [i for i in range(num_k4s) if k4_distinct[i] < 5]
                if not bad_k4s:
                    break
                k4_idx = random.choice(bad_k4s)
                e_list = k4_edges_list[k4_idx]
                e_idx = random.choice(e_list)
                
                old_c = colours[e_idx]
                new_c = random.randint(0, k - 1)
                if new_c == old_c:
                    continue
                
                # Evaluate delta: for each affected K4, check if distinct count changes
                affected = edge_to_k4s[e_idx]
                delta_bad = 0  # change in number of bad K4s (positive = worse)
                
                # Check each affected K4
                for ak in affected:
                    cnt = k4_counts[ak]
                    old_cnt = cnt[old_c]
                    new_cnt = cnt[new_c]
                    
                    # After change: old_c count decreases by 1, new_c count increases by 1
                    # Distinct count change:
                    old_distinct = k4_distinct[ak]
                    
                    # If old_c was the last of that color, distinct decreases by 1
                    # If new_c was zero before, distinct increases by 1
                    new_distinct = old_distinct
                    if old_cnt == 1:
                        new_distinct -= 1
                    if new_cnt == 0:
                        new_distinct += 1
                    
                    old_is_bad = 1 if old_distinct < 5 else 0
                    new_is_bad = 1 if new_distinct < 5 else 0
                    delta_bad += (new_is_bad - old_is_bad)
                
                new_bad = bad_count + delta_bad
                
                # Acceptance criterion
                if delta_bad <= 0 or random.random() < pow(2.718281828, -delta_bad / T):
                    # Apply the move
                    colours[e_idx] = new_c
                    for ak in affected:
                        cnt = k4_counts[ak]
                        cnt[old_c] -= 1
                        cnt[new_c] += 1
                        # Update distinct
                        old_distinct = k4_distinct[ak]
                        new_distinct = old_distinct
                        if cnt[old_c] == 0:
                            new_distinct -= 1
                        if cnt[new_c] == 1:
                            new_distinct += 1
                        k4_distinct[ak] = new_distinct
                    
                    bad_count = new_bad
                    if bad_count < best_bad:
                        best_bad = bad_count
                        best_colours = colours[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    T *= cool_rate
                else:
                    no_improve += 1
                    T *= cool_rate
                
                if no_improve > max_no_improve:
                    break
            
            if best_bad == 0:
                return best_colours
    
    # Strategy: Try decreasing k from 15 down to 13
    # First, verify base is valid
    if not validate(base_colors):
        # Fallback: return base
        return compact(base_colors)
    
    best_result = compact(base_colors)
    best_k = len(set(base_colors))  # 17
    
    # Try k = 15, 14, 13 with time budget
    time_spent = time.time() - start_time
    remaining = deadline - start_time - 5.0  # 5s safety
    
    for k in [15, 14, 13]:
        if time.time() > deadline - 10.0:
            break
        alloc = min(remaining * 0.4, 40.0)
        result = solve_with_sa(k, alloc)
        if result is not None:
            # Validate
            if validate(result):
                best_result = compact(result)
                best_k = k
                break  # Found a valid one, stop
        remaining = deadline - start_time - 5.0
    
    # Final polish: try to reduce colors in best_result via local search
    if time.time() < deadline - 5.0:
        colors = best_result[:]
        num_colors = len(set(colors))
        
        # Try to merge colors: pick two colors and try to map one to the other
        used_colors = sorted(set(colors))
        max_polish_time = time.time() + min(20.0, deadline - start_time - 5.0)
        
        improved = True
        while improved and time.time() < max_polish_time:
            improved = False
            for c1 in used_colors[:]:
                for c2 in used_colors[:]:
                    if c1 == c2 or c2 not in used_colors:
                        continue
                    # Try mapping c1 -> c2
                    test_colors = [c2 if c == c1 else c for c in colors]
                    if validate(test_colors):
                        colors = test_colors
                        used_colors.remove(c1)
                        improved = True
                        break
                if improved:
                    break
        
        if len(set(colors)) < len(set(best_result)):
            best_result = compact(colors)
    
    return best_result
# EVOLVE-BLOCK-END
