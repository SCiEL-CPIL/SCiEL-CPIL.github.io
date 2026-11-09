import random
import math
import time

# Precompute K4 structures
# Vertices 0..15
# Edges (i,j) with i<j. Index mapping:
# We need a way to map (i,j) to an index 0..119
# and vice versa.

def get_edge_index(i, j):
    # Lexicographic order
    # (0,1), (0,2), ..., (0,15), (1,2), ...
    # Count edges before i:
    # For vertex k < i, edges (k, k+1)...(k, 15)
    # Number of edges for vertex k is 15 - k
    # Sum_{k=0}^{i-1} (15 - k)
    # Then for vertex i, edges (i, i+1)...(i, j)
    # Number is j - i
    
    # Let's just build the list once
    return None # Placeholder, will build list

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

# Helper to calculate penalty for a given coloring
def calc_penalty(colors):
    # colors is a list of 120 ints
    # For each K4, count distinct colors
    penalty = 0
    for k4 in k4s:
        # 6 edges
        c0, c1, c2, c3, c4, c5 = (colors[e] for e in k4)
        # Use a set or manual check
        # Since colors are small, we can use a small array or set
        distinct = len({c0, c1, c2, cc3, c4, c5}) # Typo in variable name in thought, fix in code
        if distinct < 5:
            penalty += 1
    return penalty

# This is O(1820 * 6) which is ~10k ops. Very fast.
# We can use this for initial check and if incremental is too complex.
# Given 120s limit, even 10,000 steps * 10k ops = 100M ops. Python might be slow for 100M ops (approx 10-50s).
# So incremental is preferred for many steps, or we limit steps.
# Let's try incremental.

def construct():
    start_time = time.time()
    max_time = 110 # Leave 10s buffer
    
    # Try K from 13 to 16
    for K in [13, 14, 15, 16]:
        # Initialize
        colors = [random.randint(0, K-1) for _ in range(120)]
        
        # Initial penalty
        penalty = calc_penalty(colors)
        
        if penalty == 0:
            return colors
            
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        alpha = 0.999 # Cooling rate
        best_colors = colors[:]
        best_penalty = penalty
        
        steps = 0
        # Run until time limit for this K
        while time.time() - start_time < max_time:
            # Check if we should move to next K
            # If we've spent too long on this K, break
            # Let's allocate 20s per K
            if time.time() - start_time > 20 * (K - 12): # 20s for 13, 40s for 14, etc? No, just fixed 20s per K
                break
            
            # Pick random edge
            e_idx = random.randint(0, 119)
            old_color = colors[e_idx]
            if K == 1:
                new_color = 0
            else:
                new_color = random.randint(0, K-1)
                if new_color == old_color:
                    continue
                    
            # Calculate delta penalty
            # We need to update penalty for all K4s containing e_idx
            # We can do this by recalculating the distinct count for those K4s
            # To do this efficiently, we need to know the current colors of the other 5 edges in each K4.
            
            delta = 0
            for k4_idx in edge_to_k4s[e_idx]:
                k4 = k4s[k4_idx]
                # Get colors of all 6 edges
                # One is e_idx, others are the rest
                # We can just get all 6 from colors array
                # But we need to simulate the change for delta calculation?
                # Actually, we can just calculate the distinct count for the K4 with old_color and with new_color.
                
                # Get the 6 edge indices
                e0, e1, e2, e3, e4, e5 = k4
                
                # Current colors
                c0 = colors[e0]
                c1 = colors[e1]
                c2 = colors[e2]
                c3 = colors[e3]
                c4 = colors[e4]
                c5 = colors[e5]
                
                # Find which one is e_idx
                # This is inefficient.
                # Better: precompute for each K4, the position of each edge?
                # Or just replace in a tuple.
                
                # Let's just create a list of colors for this K4
                k4_colors = [colors[e] for e in k4]
                
                # Count distinct with old_color (current state)
                old_distinct = len(set(k4_colors))
                
                # Simulate change
                # Find index of e_idx in k4
                # This is O(6)
                pos = k4.index(e_idx)
                k4_colors[pos] = new_color
                new_distinct = len(set(k4_colors))
                
                # Penalty contribution change
                # Old contribution: 1 if old_distinct < 5 else 0
                # New contribution: 1 if new_distinct < 5 else 0
                old_contrib = 1 if old_distinct < 5 else 0
                new_contrib = 1 if new_distinct < 5 else 0
                delta += (new_contrib - old_contrib)
                
            # Acceptance
            if delta < 0 or random.random() < math.exp(-delta / T):
                colors[e_idx] = new_color
                penalty += delta
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_colors = colors[:]
                    
                if penalty == 0:
                    return colors
                    
            T *= alpha
            if T < T_min:
                T = T_min
                
            steps += 1
            # Prevent infinite loop if T gets too small and no moves accepted
            # But we have time limit.
            
        # If we finished the time for this K without finding penalty 0
        # Check if best_penalty is 0
        if best_penalty == 0:
            return best_colors
            
    # If no valid coloring found for K < 17, use the known valid 17-color solution
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

# Wait, the inner loop `k4.index(e_idx)` is O(6).
# And we do it for 91 K4s.
# So 91 * 6 = 546 ops per step.
# 10,000 steps = 5.46M ops. Very fast.
# The bottleneck is `set(k4_colors)` which is O(6).
# So total ops per step is roughly 91 * (6 + 6) = ~1000 ops.
# 100,000 steps = 100M ops. Might be ~10-20s in Python.
# So we can run many steps.

# One issue: `k4.index(e_idx)` is slow if k4 is a tuple.
# We can precompute for each K4, a map from edge_index to position in the tuple.
# Or just store k4 as a list? No, tuple is fine, index is fast for small size.

# Let's optimize the delta calculation.
# Instead of recalculating set every time, we can maintain a frequency dict for each K4?
# That would be 1820 dicts.
# Update:
#   freq[old_color] -= 1
#   if freq[old_color] == 0: distinct_count -= 1
#   freq[new_color] += 1
#   if freq[new_color] == 1: distinct_count += 1
# This is O(1) per K4.
# Total O(91) per step.
# Much faster.

# Let's implement this optimized version.

# We need to store the current distinct count for each K4.
# And a frequency dict for each K4.

def construct():
    start_time = time.time()
    max_time = 110
    
    for K in [13, 14, 15, 16]:
        colors = [random.randint(0, K-1) for _ in range(120)]
        
        # Initialize frequency dicts and distinct counts
        k4_freqs = [{} for _ in range(len(k4s))]
        k4_distinct = [0] * len(k4s)
        penalty = 0
        
        for k4_idx, k4 in enumerate(k4s):
            freq = k4_freqs[k4_idx]
            for e in k4:
                c = colors[e]
                freq[c] = freq.get(c, 0) + 1
            k4_distinct[k4_idx] = len(freq)
            if k4_distinct[k4_idx] < 5:
                penalty += 1
                
        if penalty == 0:
            return colors
            
        T = 1.0
        T_min = 0.01
        alpha = 0.999
        best_colors = colors[:]
        best_penalty = penalty
        
        # Time limit per K: 20s
        limit_time = start_time + 20
        
        while time.time() < limit_time:
            e_idx = random.randint(0, 119)
            old_color = colors[e_idx]
            if K == 1:
                new_color = 0
            else:
                new_color = random.randint(0, K-1)
                if new_color == old_color:
                    continue
                    
            delta = 0
            for k4_idx in edge_to_k4s[e_idx]:
                freq = k4_freqs[k4_idx]
                d = k4_distinct[k4_idx]
                
                # Update old color
                freq[old_color] -= 1
                if freq[old_color] == 0:
                    d -= 1
                # Update new color
                if new_color in freq:
                    freq[new_color] += 1
                else:
                    freq[new_color] = 1
                    d += 1
                    
                # Calculate change in penalty contribution
                old_contrib = 1 if (d + (1 if freq[old_color]==0 else 0) - (1 if (new_color in freq and freq[new_color]==1) else 0)) < 5 else 0 
                # This logic is getting messy. Let's just compute old and new distinct counts clearly.
                
                # Let's revert and do it cleanly.
                # Save old state
                old_d = k4_distinct[k4_idx]
                old_contrib = 1 if old_d < 5 else 0
                
                # Apply change
                freq[old_color] -= 1
                if freq[old_color] == 0:
                    new_d = old_d - 1
                else:
                    new_d = old_d
                    
                if new_color in freq:
                    freq[new_color] += 1
                    # distinct count doesn't change if color was already present
                else:
                    freq[new_color] = 1
                    new_d += 1
                    
                new_contrib = 1 if new_d < 5 else 0
                delta += (new_contrib - old_contrib)
                
                # Store new state
                k4_distinct[k4_idx] = new_d
                
            if delta < 0 or random.random() < math.exp(-delta / T):
                colors[e_idx] = new_color
                penalty += delta
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_colors = colors[:]
                if penalty == 0:
                    return colors
            else:
                # Revert the frequency changes
                for k4_idx in edge_to_k4s[e_idx]:
                    freq = k4_freqs[k4_idx]
                    # Revert old_color
                    if old_color in freq:
                        freq[old_color] += 1
                    else:
                        freq[old_color] = 1
                    # Revert new_color
                    if new_color in freq:
                        freq[new_color] -= 1
                        if freq[new_color] == 0:
                            pass # distinct count was increased, so decrease it
                    else:
                        pass
                    
                    # This revert logic is error-prone.
                    # It's better to just recalculate the delta and if rejected, 
                    # we must undo the changes to freq and distinct.
                    
                    # Let's simplify: 
                    # If rejected, we must restore the state.
                    # We can just store the old values of distinct and freq changes?
                    # Or just recalculate the distinct count for these K4s from scratch?
                    # That's O(6) per K4. 91*6 = 546.
                    # If rejected, we do 546 ops.
                    # If accepted, we do 91 ops.
                    # This is safe and simple.
                    
                    pass # Placeholder for revert logic
                    
            T *= alpha
            if T < T_min:
                T = T_min

        if best_penalty == 0:
            return best_colors

    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

# The revert logic is tricky.
# Let's use the "recalculate if rejected" approach.
# If accepted, we have already updated freq and distinct.
# If rejected, we need to undo.
# Undoing freq is easy:
#   freq[old_color] += 1
#   freq[new_color] -= 1 (if it became 0, remove it? No, just set to 0 or delete)
#   distinct count: 
#     if old_color count became 0, distinct was decremented. So increment back.
#     if new_color count was 1 (before adding), distinct was incremented. So decrement back.
# We can track these conditions.

# Let's refine the loop.

def construct():
    start_time = time.time()
    
    for K in [13, 14, 15, 16]:
        colors = [random.randint(0, K-1) for _ in range(120)]
        
        k4_freqs = [{} for _ in range(len(k4s))]
        k4_distinct = [0] * len(k4s)
        penalty = 0
        
        for k4_idx, k4 in enumerate(k4s):
            freq = k4_freqs[k4_idx]
            for e in k4:
                c = colors[e]
                freq[c] = freq.get(c, 0) + 1
            k4_distinct[k4_idx] = len(freq)
            if k4_distinct[k4_idx] < 5:
                penalty += 1
                
        if penalty == 0:
            return colors
            
        T = 1.0
        T_min = 0.01
        alpha = 0.999
        best_colors = colors[:]
        best_penalty = penalty
        
        limit_time = start_time + 20
        
        while time.time() < limit_time:
            e_idx = random.randint(0, 119)
            old_color = colors[e_idx]
            if K == 1:
                new_color = 0
            else:
                new_color = random.randint(0, K-1)
                if new_color == old_color:
                    continue
                    
            delta = 0
            affected_k4s = edge_to_k4s[e_idx]
            
            # Store info to revert if necessary
            revert_info = []
            
            for k4_idx in affected_k4s:
                freq = k4_freqs[k4_idx]
                old_d = k4_distinct[k4_idx]
                old_contrib = 1 if old_d < 5 else 0
                
                # Check conditions before modifying
                old_count = freq.get(old_color, 0)
                new_count = freq.get(new_color, 0)
                
                # Apply modification
                if old_count > 0:
                    freq[old_color] = old_count - 1
                # new_color
                if new_count > 0:
                    freq[new_color] = new_count + 1
                else:
                    freq[new_color] = 1
                    
                # Calculate new distinct
                new_d = old_d
                if old_count == 1: # old_color removed
                    new_d -= 1
                if new_count == 0: # new_color added
                    new_d += 1
                    
                new_contrib = 1 if new_d < 5 else 0
                delta += (new_contrib - old_contrib)
                
                # Store revert info
                revert_info.append((k4_idx, old_d, old_count, new_count))
                
                # Update state (tentatively)
                k4_distinct[k4_idx] = new_d
                
            if delta < 0 or random.random() < math.exp(-delta / T):
                colors[e_idx] = new_color
                penalty += delta
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_colors = colors[:]
                if penalty == 0:
                    return colors
            else:
                # Revert
                for k4_idx, old_d, old_count, new_count in revert_info:
                    freq = k4_freqs[k4_idx]
                    # Revert freq
                    if old_count > 0:
                        freq[old_color] = old_count
                    else:
                        # old_color was not present? Impossible, it was the edge color.
                        pass
                    if new_count > 0:
                        freq[new_color] = new_count
                    else:
                        # new_color was not present, we added it, so remove it
                        if new_color in freq:
                            del freq[new_color]
                            
                    k4_distinct[k4_idx] = old_d
                    
            T *= alpha
            if T < T_min:
                T = T_min

        if best_penalty == 0:
            return best_colors

    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

# This looks correct.
# One small detail: `freq[old_color]` might not exist if old_count was 0? No, old_color is the current color of the edge, so it must be in freq.
# `freq[new_color]` might not exist if new_count was 0. We handled that.

# Let's double check the revert logic.
# If rejected:
#   We need to restore freq and distinct to the state BEFORE the tentative change.
#   For each affected K4:
#     freq[old_color] should be old_count.
#     freq[new_color] should be new_count. If new_count was 0, it should not be in freq (or be 0).
#     distinct should be old_d.
#
# In the tentative change:
#   We did:
#     if old_count > 0: freq[old_color] = old_count - 1
#     if new_count > 0: freq[new_color] = new_count + 1
#     else: freq[new_color] = 1
#
# To revert:
#   if old_count > 0: freq[old_color] = old_count
#   if new_count > 0: freq[new_color] = new_count
#   else: del freq[new_color] (since we added it)
#
# This matches my code.

# The time limit is 20s per K.
# Total 80s for 4 Ks.
# Plus initial setup.
# Should be well within 120s.

# One optimization: if K=13 is very hard, it might take the full 20s.
# If K=14 is easier, it might find a solution faster.
# This is fine.

# Final check on the output format.
# "120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16, in lexicographic order."
# My `colors` list is indexed by the edge index derived from lexicographic order.
# So `colors` is exactly the required format.

# The fallback solution is also in the correct format.

# I will add a seed for reproducibility? No, random is fine.
# But to ensure it doesn't fail on first try, the fallback is good.

# One potential issue: `math.exp(-delta / T)` might underflow if delta is large and T is small.
# If delta > 0 and T is small, exp is 0. That's fine.
# If delta is very negative, exp is 1. That's fine.

# Let's run it.
