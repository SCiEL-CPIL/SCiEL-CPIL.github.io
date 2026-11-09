import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Precompute edges
    edges = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    n_edges = 120

    # Map edge (u, v) to index
    edge_idx_map = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx_map[(i, j)] = idx
        edge_idx_map[(j, i)] = idx

    # Generate all K4s
    k4s_all = []
    for quad in combinations(range(16), 4):
        a, b, c, d = quad
        e_indices = [
            edge_idx_map[(a, b)],
            edge_idx_map[(a, c)],
            edge_idx_map[(a, d)],
            edge_idx_map[(b, c)],
            edge_idx_map[(b, d)],
            edge_idx_map[(c, d)]
        ]
        k4s_all.append(e_indices)
    
    n_k4s = len(k4s_all)

    # For each edge, the list of K4 indices it belongs to
    edge_k4s = [[] for _ in range(n_edges)]
    for k4_idx, e_indices in enumerate(k4s_all):
        for ei in e_indices:
            edge_k4s[ei].append(k4_idx)

    def count_distinct_colors_in_k4(k4_idx, cols):
        e_indices = k4s_all[k4_idx]
        c0 = cols[e_indices[0]]
        c1 = cols[e_indices[1]]
        c2 = cols[e_indices[2]]
        c3 = cols[e_indices[3]]
        c4 = cols[e_indices[4]]
        c5 = cols[e_indices[5]]
        
        # Efficient distinct count for 6 elements
        # Using a set is okay for 6 elements, but let's try to be faster
        # Actually, set() is implemented in C and is very fast for small sizes.
        return len({c0, c1, c2, c3, c4, c5})

    def is_valid_k4(k4_idx, cols):
        return count_distinct_colors_in_k4(k4_idx, cols) >= 5

    # Initial colors: (i+j) % 17
    colors = [0] * n_edges
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 17

    # Initial state
    invalid_count = 0
    for k4_idx in range(n_k4s):
        if not is_valid_k4(k4_idx, colors):
            invalid_count += 1
            
    color_freq = {}
    for c in colors:
        color_freq[c] = color_freq.get(c, 0) + 1
        
    distinct_colors = len(color_freq)
    
    best_colors = colors[:]
    best_distinct = distinct_colors
    best_invalid = invalid_count

    # Simulated Annealing Parameters
    T = 10.0
    T_min = 0.01
    alpha = 0.999  # Cooling rate
    
    # We will interleave SA steps with compression attempts
    # Compression is deterministic and powerful if it succeeds
    
    steps = 0
    last_compress_attempt = 0
    
    while time.time() < deadline:
        # 1. SA Step
        e_idx = random.randint(0, n_edges - 1)
        old_color = colors[e_idx]
        
        # Choose a new color. 
        # We want to explore colors that might help reduce distinct count or fix invalid K4s.
        # Picking from [0, 20] is reasonable.
        new_color = random.randint(0, 20)
        
        if new_color == old_color:
            continue
            
        # Calculate Delta Invalid Count
        delta_invalid = 0
        affected_k4s = edge_k4s[e_idx]
        
        # We need to check the validity of affected K4s before and after
        # To avoid modifying `colors` until we decide to accept, 
        # we can simulate the change for the affected K4s.
        
        # Store old validity status of affected K4s
        # Actually, we just need to know how many of them become invalid/valid.
        
        # Let's calculate the change in distinct colors for each affected K4.
        # If a K4 was valid (>=5) and becomes invalid (<5), invalid_count increases.
        # If a K4 was invalid (<5) and becomes valid (>=5), invalid_count decreases.
        
        # Optimization: 
        # We only need to check the 91 affected K4s.
        
        # Temporarily apply change? No, let's just compute.
        
        # For each affected K4, get the 6 edge indices.
        # One of them is e_idx.
        # We can pre-compute the other 5 edge indices for each (edge, K4) pair?
        # Or just fetch them.
        
        # Let's fetch and compute.
        old_invalids = 0
        new_invalids = 0
        
        for k4_idx in affected_k4s:
            e_indices = k4s_all[k4_idx]
            
            # Get colors of the 6 edges
            # One is old_color, others are colors[e_i]
            # We need to check if the set size changes.
            
            # Get the 5 other colors
            other_colors = []
            for ei in e_indices:
                if ei != e_idx:
                    other_colors.append(colors[ei])
            
            # Old set
            old_set = set(other_colors)
            old_set.add(old_color)
            old_distinct = len(old_set)
            
            # New set
            new_set = set(other_colors)
            new_set.add(new_color)
            new_distinct = len(new_set)
            
            if old_distinct < 5:
                old_invalids += 1
            if new_distinct < 5:
                new_invalids += 1
                
        delta_invalid = new_invalids - old_invalids
        
        # Calculate Delta Distinct Colors (global)
        # This affects the global distinct color count, but SA cost is usually:
        # Cost = M * invalid_count + distinct_colors
        # Delta Cost = M * delta_invalid + delta_distinct_global
        
        # Delta distinct global:
        # If old_color count was 1, it disappears.
        # If new_color count was 0, it appears.
        
        old_freq = color_freq.get(old_color, 0)
        new_freq = color_freq.get(new_color, 0)
        
        delta_distinct_global = 0
        if old_freq == 1:
            delta_distinct_global -= 1
        if new_freq == 0:
            delta_distinct_global += 1
            
        M = 1000
        delta_cost = M * delta_invalid + delta_distinct_global
        
        # Acceptance probability
        if delta_cost <= 0:
            accept = True
        else:
            # T > 0
            prob = math.exp(-delta_cost / T)
            accept = random.random() < prob
            
        if accept:
            # Apply change
            colors[e_idx] = new_color
            
            # Update global state
            invalid_count += delta_invalid
            
            # Update color freq
            if old_freq == 1:
                del color_freq[old_color]
            else:
                color_freq[old_color] = old_freq - 1
                
            color_freq[new_color] = new_freq + 1
            
            distinct_colors += delta_distinct_global
            
            # Update best
            if invalid_count < best_invalid or (invalid_count == best_invalid and distinct_colors < best_distinct):
                best_invalid = invalid_count
                best_distinct = distinct_colors
                best_colors = colors[:]
                
        # Cooling
        T *= alpha
        if T < T_min:
            T = T_min
            
        steps += 1
        
        # 2. Compression Attempt
        # Try to compress if we have a valid coloring (invalid_count == 0)
        # Or even if not, but it's safer to compress valid states.
        # Let's attempt compression every 100 steps if invalid_count == 0.
        
        if invalid_count == 0 and steps - last_compress_attempt > 50:
            last_compress_attempt = steps
            # Try to merge two colors
            used_colors = list(color_freq.keys())
            if len(used_colors) > 1:
                # Pick two random colors to merge
                # Bias towards merging a low frequency color into a high frequency one?
                # Or just random.
                c1, c2 = random.sample(used_colors, 2)
                
                # Try to merge c2 into c1
                # This means replacing all c2 with c1.
                
                # Check if this merge is valid.
                # This is a batch change. We need to check all K4s affected by any edge with color c2.
                
                # Identify edges with color c2
                edges_with_c2 = [i for i, c in enumerate(colors) if c == c2]
                
                # Identify all K4s affected
                affected_k4_set = set()
                for e_idx in edges_with_c2:
                    for k4_idx in edge_k4s[e_idx]:
                        affected_k4_set.add(k4_idx)
                        
                # Check validity of these K4s after merge
                # We simulate the merge by checking if any K4 becomes invalid.
                
                merge_valid = True
                for k4_idx in affected_k4_set:
                    e_indices = k4s_all[k4_idx]
                    # Calculate distinct colors in this K4 after merge
                    # Any color c2 becomes c1.
                    distincts = set()
                    for ei in e_indices:
                        c = colors[ei]
                        if c == c2:
                            c = c1
                        distincts.add(c)
                    if len(distincts) < 5:
                        merge_valid = False
                        break
                        
                if merge_valid:
                    # Apply merge
                    for e_idx in edges_with_c2:
                        colors[e_idx] = c1
                        color_freq[c1] += 1
                    del color_freq[c2]
                    distinct_colors -= 1
                    
                    # Update best
                    if invalid_count < best_invalid or (invalid_count == best_invalid and distinct_colors < best_distinct):
                        best_invalid = invalid_count
                        best_distinct = distinct_colors
                        best_colors = colors[:]
                        
                    # Restart temperature? Or just continue?
                    # Compression is a big win. Let's cool down less or reset T slightly?
                    # Let's just continue.
                    
    # Final Compression Pass
    # If best_colors is valid, try to compress it further.
    # We can run a deterministic compression loop on best_colors.
    
    # Copy best_colors to work on
    final_colors = best_colors[:]
    freq = {}
    for c in final_colors:
        freq[c] = freq.get(c, 0) + 1
        
    improved = True
    while improved:
        improved = False
        used_colors = list(freq.keys())
        if len(used_colors) <= 1:
            break
            
        # Try all pairs? Or random?
        # Since this is the final pass, we can be exhaustive if the number of colors is small.
        # If distinct colors is ~15, pairs are 105.
        # Checking validity of a merge is O(N_k4s * 6).
        # 1820 * 6 = 10920 ops.
        # 105 * 10920 = 1.1M ops. This is fast.
        
        for i in range(len(used_colors)):
            for j in range(i + 1, len(used_colors)):
                c1 = used_colors[i]
                c2 = used_colors[j]
                
                # Try merge c2 into c1
                # Identify edges with c2
                edges_with_c2 = [idx for idx, c in enumerate(final_colors) if c == c2]
                
                # Identify affected K4s
                affected_k4_set = set()
                for e_idx in edges_with_c2:
                    for k4_idx in edge_k4s[e_idx]:
                        affected_k4_set.add(k4_idx)
                        
                merge_valid = True
                for k4_idx in affected_k4_set:
                    e_indices = k4s_all[k4_idx]
                    distincts = set()
                    for ei in e_indices:
                        c = final_colors[ei]
                        if c == c2:
                            c = c1
                        distincts.add(c)
                    if len(distincts) < 5:
                        merge_valid = False
                        break
                        
                if merge_valid:
                    # Apply merge
                    for e_idx in edges_with_c2:
                        final_colors[e_idx] = c1
                    freq[c1] += freq[c2]
                    del freq[c2]
                    improved = True
                    # Restart loop as colors changed
                    break
            if improved:
                break
                
    # Compress final color labels to 0..k-1
    unique_colors = sorted(list(set(final_colors)))
    mapping = {c: i for i, c in enumerate(unique_colors)}
    return [mapping[c] for c in final_colors]
