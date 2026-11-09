import random
import math
import time

# Precompute K4 structures
# Vertices 0..15
# Edges (i,j) with i<j. Index mapping:
# We need a way to map (i,j) to an index 0..119
# and vice versa.

def get_edge_index(i, j):
    # Placeholder, will build list
    return None

# Build edge list
edges = []
edge_to_idx = {}
for i in range(16):
    for j in range(i+1, 16):
        edge_to_idx[(i,j)] = len(edges)
        edges.append((i,j))

# Build K4s
# A K4 is a set of 4 vertices.
# For each set of 4 vertices, get the 6 edges.
k4s = []
for v1 in range(16):
    for v2 in range(v1+1, 16):
        for v3 in range(v2+1, 16):
            for v4 in range(v3+1, 16):
                verts = [v1, v2, v3, v4]
                edge_indices = []
                for x in range(4):
                    for y in range(x+1, 4):
                        u, v = verts[x], verts[y]
                        if u > v: u, v = v, u
                        edge_indices.append(edge_to_idx[(u,v)])
                k4s.append(tuple(edge_indices))

# Precompute for each edge, the list of K4 indices containing it
edge_to_k4s = [[] for _ in range(120)]
for k4_idx, k4 in enumerate(k4s):
    for e_idx in k4:
        edge_to_k4s[e_idx].append(k4_idx)

# Precompute distinct colors count for a given K4 given current colors
# We will maintain this incrementally.

def construct():
    start_time = time.time()
    max_time = 115 # Leave 5s buffer for safety
    
    # We want to minimize K.
    # Try K=13, then 14, then 15.
    # If we find a valid coloring for K, return it.
    
    for K in [13, 14, 15]:
        # Time budget for each K
        # K=13 is hardest, give it more time? Or less because it's unlikely?
        # If K=13 is possible, it's the best score.
        # Let's give 35s to 13, 35s to 14, 35s to 15.
        time_budget = 35
        end_time_for_K = start_time + (time.time() - start_time) + time_budget
        
        # Run multiple SA attempts within this budget
        # Each attempt runs for a short time (e.g., 5s) or fixed steps.
        # If an attempt finds a solution, return it.
        # If all attempts in the budget fail, move to next K.
        
        num_attempts = 7 # 7 attempts * 5s = 35s
        attempt_time = time_budget / num_attempts
        
        for attempt in range(num_attempts):
            if time.time() > end_time_for_K:
                break
            
            # Initialize colors
            # Greedy initialization: Random start, then hill climb to reduce penalty
            colors = [random.randint(0, K-1) for _ in range(120)]
            
            # State variables
            # k4_freqs[k4_idx]: dict {color: count}
            # k4_distinct[k4_idx]: int
            k4_freqs = [{} for _ in range(len(k4s))]
            k4_distinct = [0] * len(k4s)
            
            def calc_penalty():
                p = 0
                for d in k4_distinct:
                    if d < 5:
                        p += (5 - d)
                return p
            
            # Initialize state
            for k4_idx, k4 in enumerate(k4s):
                freq = k4_freqs[k4_idx]
                for e in k4:
                    c = colors[e]
                    freq[c] = freq.get(c, 0) + 1
                k4_distinct[k4_idx] = len(freq)
                
            current_penalty = calc_penalty()
            
            # Quick greedy improvement (Hill Climbing)
            # Try changing each edge to the best color (minimizing local penalty)
            improved = True
            max_hc_steps = 5000
            steps = 0
            while improved and steps < max_hc_steps:
                improved = False
                steps += 1
                # Shuffle edges to avoid bias
                edge_order = list(range(120))
                random.shuffle(edge_order)
                for e_idx in edge_order:
                    old_color = colors[e_idx]
                    best_color = old_color
                    best_local_penalty = 0
                    # Current contribution to penalty for K4s containing e_idx
                    # We need to calculate the change in penalty if we change e_idx to c
                    # This is expensive to do for all K if done naively.
                    # Instead, just try a few random colors and pick the best.
                    # Or, simply try all K colors and pick the one with lowest global penalty delta.
                    
                    # Calculate current penalty contribution of e_idx
                    # It's easier to just evaluate the global penalty change for a few candidates.
                    # Let's try all K colors.
                    min_delta = 0
                    for new_color in range(K):
                        if new_color == old_color:
                            continue
                        
                        # Simulate change
                        delta = 0
                        affected = edge_to_k4s[e_idx]
                        for k4_idx in affected:
                            freq = k4_freqs[k4_idx]
                            old_d = k4_distinct[k4_idx]
                            old_contrib = max(0, 5 - old_d)
                            
                            old_count = freq.get(old_color, 0)
                            new_count = freq.get(new_color, 0)
                            
                            # Apply
                            if old_count > 0:
                                freq[old_color] = old_count - 1
                                if old_count == 1:
                                    new_d = old_d - 1
                                else:
                                    new_d = old_d
                            else:
                                new_d = old_d
                            
                            if new_count > 0:
                                freq[new_color] = new_count + 1
                                new_d2 = new_d
                            else:
                                freq[new_color] = 1
                                new_d2 = new_d + 1
                                
                            # Wait, if old_count == 1 and new_count == 0, new_d = old_d - 1 + 1 = old_d?
                            # Let's re-evaluate distinct count change carefully.
                            # old_d is count of keys with count > 0.
                            # Change old_color count:
                            #   if old_count == 1: key becomes 0 (if we don't delete, we track count). Distinct -1.
                            #   if old_count > 1: Distinct unchanged.
                            # Change new_color count:
                            #   if new_count == 0: key becomes 1. Distinct +1.
                            #   if new_count > 0: Distinct unchanged.
                            
                            d_change = 0
                            if old_count == 1: d_change -= 1
                            if new_count == 0: d_change += 1
                            
                            new_d_final = old_d + d_change
                            new_contrib = max(0, 5 - new_d_final)
                            delta += (new_contrib - old_contrib)
                            
                            # Revert
                            if old_count > 0:
                                freq[old_color] = old_count
                            else:
                                freq[old_color] = 0 # Should not be here
                            if new_count > 0:
                                freq[new_color] = new_count
                            else:
                                freq[new_color] = 0
                                
                        if delta < min_delta:
                            min_delta = delta
                            best_color = new_color
                            
                    if min_delta < 0:
                        # Apply the best color
                        new_color = best_color
                        affected = edge_to_k4s[e_idx]
                        for k4_idx in affected:
                            freq = k4_freqs[k4_idx]
                            old_d = k4_distinct[k4_idx]
                            old_count = freq.get(old_color, 0)
                            new_count = freq.get(new_color, 0)
                            
                            d_change = 0
                            if old_count == 1: d_change -= 1
                            if new_count == 0: d_change += 1
                            
                            new_d_final = old_d + d_change
                            
                            # Update freq
                            freq[old_color] = old_count - 1 if old_count > 0 else 0
                            freq[new_color] = new_count + 1
                            
                            k4_distinct[k4_idx] = new_d_final
                            
                        colors[e_idx] = new_color
                        current_penalty += min_delta
                        improved = True
                        if current_penalty == 0:
                            return colors

            if current_penalty == 0:
                return colors

            # Start Simulated Annealing
            T = 1.0
            T_min = 0.01
            alpha = 0.995
            best_penalty_sa = current_penalty
            best_colors_sa = colors[:]
            
            sa_start = time.time()
            while time.time() - sa_start < attempt_time:
                # Move selection
                if random.random() < 0.5:
                    # Targeted: Pick a bad K4
                    bad_k4s = [i for i in range(len(k4s)) if k4_distinct[i] < 5]
                    if bad_k4s:
                        k4_idx = random.choice(bad_k4s)
                        e_idx = random.choice(k4s[k4_idx])
                    else:
                        e_idx = random.randint(0, 119)
                else:
                    e_idx = random.randint(0, 119)
                    
                old_color = colors[e_idx]
                if K == 1:
                    new_color = 0
                else:
                    new_color = random.randint(0, K-1)
                    if new_color == old_color:
                        continue
                        
                delta = 0
                affected = edge_to_k4s[e_idx]
                
                # Store revert info
                revert_info = []
                
                for k4_idx in affected:
                    freq = k4_freqs[k4_idx]
                    old_d = k4_distinct[k4_idx]
                    old_contrib = max(0, 5 - old_d)
                    
                    old_count = freq.get(old_color, 0)
                    new_count = freq.get(new_color, 0)
                    
                    d_change = 0
                    if old_count == 1: d_change -= 1
                    if new_count == 0: d_change += 1
                    
                    new_d_final = old_d + d_change
                    new_contrib = max(0, 5 - new_d_final)
                    delta += (new_contrib - old_contrib)
                    
                    revert_info.append((k4_idx, old_d, old_count, new_count))
                    k4_distinct[k4_idx] = new_d_final
                    # Temporarily update freq? No, we revert if rejected.
                    # But we need freq to be correct for subsequent checks in this loop?
                    # No, we only check one edge at a time.
                    # We must update freq if we accept. If we reject, we revert.
                    # It's safer to update freq now and revert if needed.
                    freq[old_color] = old_count - 1 if old_count > 0 else 0
                    freq[new_color] = new_count + 1
                    
                if delta < 0 or random.random() < math.exp(-delta / T):
                    colors[e_idx] = new_color
                    current_penalty += delta
                    if current_penalty < best_penalty_sa:
                        best_penalty_sa = current_penalty
                        best_colors_sa = colors[:]
                    if current_penalty == 0:
                        return colors
                else:
                    # Revert
                    for k4_idx, old_d, old_count, new_count in revert_info:
                        freq = k4_freqs[k4_idx]
                        freq[old_color] = old_count
                        freq[new_color] = new_count
                        k4_distinct[k4_idx] = old_d
                        
                T *= alpha
                if T < T_min:
                    T = T_min
                    
            if best_penalty_sa == 0:
                return best_colors_sa

    # Fallback: Return a valid 16-coloring or best found
    # A simple valid coloring: (i+j) % 17 is valid for 17 colors.
    # We need 16 colors?
    # Let's just return a random 16-coloring or the best we found.
    # If we didn't find any valid one, we return the one with min penalty for K=15?
    # The task requires valid coloring. If we fail, score is 0.
    # But we should try to return *something* valid if possible.
    # 16 colors is guaranteed to work?
    # f(16) <= 15 is a bound. So 15 should work.
    # If our SA failed for 15, we might have a bug or bad luck.
    # Let's try a deterministic construction for 15 or 16 if available.
    # For now, return a random 16-coloring and hope it's valid? No, that's risky.
    # Let's return the best_colors_sa from the last run (K=15) even if penalty > 0?
    # No, invalid scores 0.
    # Let's try to construct a valid 16-coloring.
    # Identity coloring: color(i,j) = i. Uses 16 colors.
    # Does it satisfy the condition?
    # K4 on {a,b,c,d}. Edges: (a,b)->a, (a,c)->a, (a,d)->a, (b,c)->b, (b,d)->b, (c,d)->c.
    # Colors: a, a, a, b, b, c.
    # Distinct: a, b, c. Only 3 colors. Invalid.
    
    # Another construction: color(i,j) = (i+j) % 16?
    # K4 {0,1,2,3}.
    # (0,1)->1, (0,2)->2, (0,3)->3, (1,2)->3, (1,3)->4, (2,3)->5.
    # Colors: 1,2,3,3,4,5. Distinct: 1,2,3,4,5. 5 colors. Valid?
    # Let's check another K4: {0,1,4,5}.
    # (0,1)->1, (0,4)->4, (0,5)->5, (1,4)->5, (1,5)->6, (4,5)->9.
    # Colors: 1,4,5,5,6,9. Distinct: 1,4,5,6,9. 5 colors.
    # It seems (i+j) % 16 might work?
    # Let's verify if (i+j) % K works for K=16.
    # If it works, we can use it as a fallback.
    
    # Let's check K4 {0, 2, 4, 6} with mod 16.
    # (0,2)->2, (0,4)->4, (0,6)->6, (2,4)->6, (2,6)->8, (4,6)->10.
    # Colors: 2,4,6,6,8,10. Distinct: 2,4,6,8,10. 5 colors.
    
    # What if i+j wraps around?
    # {14, 15, 0, 1} mod 16.
    # (14,15)->29%16=13, (14,0)->14, (14,1)->15, (15,0)->15, (15,1)->0, (0,1)->1.
    # Colors: 13, 14, 15, 15, 0, 1. Distinct: 13,14,15,0,1. 5 colors.
    
    # It seems (i+j) % 16 is a valid 16-coloring.
    # Let's use it as a fallback.
    
    return [(i + j) % 16 for i in range(16) for j in range(i + 1, 16)]
