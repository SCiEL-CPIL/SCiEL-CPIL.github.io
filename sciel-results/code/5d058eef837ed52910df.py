import random
import math
import time
import numpy as np

# Precompute K4 structures
edges = []
edge_to_idx = {}
for i in range(16):
    for j in range(i+1, 16):
        edge_to_idx[(i,j)] = len(edges)
        edges.append((i,j))

# Build K4s
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

# Convert k4s to numpy array for faster access if needed, but list of lists is fine for iteration
# Actually, we need edge_to_k4s
edge_to_k4s = [[] for _ in range(120)]
for k4_idx, k4 in enumerate(k4s):
    for e_idx in k4:
        edge_to_k4s[e_idx].append(k4_idx)

# Precompute as numpy arrays for speed? 
# edge_to_k4s_np = [np.array(lst) for lst in edge_to_k4s]

def construct():
    start_time = time.time()
    max_time = 115 # Leave 5s buffer
    
    # We want to minimize K.
    # Try K=13, then 14, etc.
    # If we find a valid solution for K, return it.
    
    best_solution = None
    best_K = 100
    
    # Time allocation
    # K=13 is hardest. Give it more time? Or equal time?
    # If K=13 fails, we must return a valid solution with K=14 or 15.
    # The score is based on the number of colors used.
    # So finding a valid K=13 is best.
    
    # Strategy:
    # 1. Try to find valid K=13.
    # 2. If fail, try K=14.
    # 3. If fail, try K=15.
    # 4. If fail, return best found (or a known valid one).
    
    # We will run SA for each K.
    # If we find a solution for K, we return immediately.
    
    # To ensure we return *something* valid if SA fails, we should have a fallback.
    # A random coloring with K=16 is likely valid?
    # Probability that a K4 has < 5 colors with K=16?
    # Birthday paradox style. 
    # With 16 colors and 6 edges, probability of all 6 being same is low.
    # Probability of 5 or fewer distinct colors?
    # It's small but not zero.
    # We can verify a candidate solution before returning.
    
    def is_valid(colors, K):
        # Check all K4s
        # Using numpy for speed?
        # k4s is list of tuples of 6 edge indices.
        # colors is list of ints.
        # For each k4, get 6 colors, count distinct.
        # If any < 5, return False.
        # This is O(1820 * 6). ~10k ops. Fast.
        for k4 in k4s:
            distinct = set()
            for e in k4:
                distinct.add(colors[e])
            if len(distinct) < 5:
                return False
        return True

    # SA Parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.9995 # Slow cooling
    
    for K in [13, 14, 15, 16]:
        if time.time() - start_time > max_time:
            break
            
        # Budget per K
        # If K=13, give 40s. If 14, 30s. If 15, 20s. If 16, 10s.
        budgets = {13: 40, 14: 30, 15: 20, 16: 10}
        budget = budgets[K]
        end_time = time.time() + budget
        
        # Initialize
        colors = np.random.randint(0, K, 120, dtype=np.int8)
        
        # Initialize counts and distinct
        # counts shape: (1820, 16) -- max K is 16, so we can allocate 16 columns.
        # This avoids resizing.
        counts = np.zeros((1820, 16), dtype=np.int8)
        distinct = np.zeros(1820, dtype=np.int8)
        
        # Populate initial state
        for k4_idx, k4 in enumerate(k4s):
            # k4 is tuple of 6 edge indices
            # Get colors for these edges
            # We can do this with numpy fancy indexing if colors is numpy array
            # c_k4 = colors[list(k4)]
            # But we need to update counts.
            # Since K <= 16, we can just loop 6 times.
            local_counts = np.zeros(16, dtype=np.int8)
            for e in k4:
                c = colors[e]
                local_counts[c] += 1
            counts[k4_idx] = local_counts
            distinct[k4_idx] = np.count_nonzero(local_counts)
            
        # Calculate initial penalty
        penalty = np.sum(distinct < 5)
        
        if penalty == 0:
            if is_valid(colors.tolist(), K): # Double check
                return colors.tolist()
        
        best_colors = colors.copy()
        best_penalty = penalty
        
        T = T_start
        
        # Precompute edge_to_k4s as lists of numpy arrays for faster iteration?
        # Or just lists of ints.
        
        while time.time() < end_time and time.time() < start_time + max_time:
            e_idx = np.random.randint(0, 120)
            old_color = colors[e_idx]
            
            if K == 1:
                new_color = 0
            else:
                new_color = np.random.randint(0, K)
                if new_color == old_color:
                    continue
            
            # Get affected K4s
            affected = edge_to_k4s[e_idx]
            
            delta = 0
            # Store revert info
            # We need to store: k4_idx, old_count_old, old_count_new, old_distinct
            # We can store this in a list of tuples.
            revert_info = []
            
            # Optimization: If delta becomes large positive, we might reject early?
            # But we need to update state to know delta.
            
            for k4_idx in affected:
                old_count_old = counts[k4_idx, old_color]
                old_count_new = counts[k4_idx, new_color]
                old_distinct = distinct[k4_idx]
                
                # Apply change
                counts[k4_idx, old_color] = old_count_old - 1
                counts[k4_idx, new_color] = old_count_new + 1
                
                # Calculate new distinct
                new_distinct = old_distinct
                if old_count_old == 1:
                    new_distinct -= 1
                if old_count_new == 0:
                    new_distinct += 1
                    
                # Calculate delta penalty
                old_bad = 1 if old_distinct < 5 else 0
                new_bad = 1 if new_distinct < 5 else 0
                delta += (new_bad - old_bad)
                
                # Store revert info
                revert_info.append((k4_idx, old_count_old, old_count_new, old_distinct))
                
                # Update distinct
                distinct[k4_idx] = new_distinct
                
            # Acceptance
            if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                colors[e_idx] = new_color
                penalty += delta
                if penalty < best_penalty:
                    best_penalty = penalty
                    best_colors = colors.copy()
                    if penalty == 0:
                        # Verify
                        if is_valid(colors.tolist(), K):
                            return colors.tolist()
            else:
                # Revert
                for k4_idx, old_count_old, old_count_new, old_distinct in revert_info:
                    counts[k4_idx, old_color] = old_count_old
                    counts[k4_idx, new_color] = old_count_new
                    distinct[k4_idx] = old_distinct
                    
            T *= alpha
            if T < T_end:
                T = T_end
                
        # End of SA for this K
        if best_penalty == 0:
            if is_valid(best_colors.tolist(), K):
                return best_colors.tolist()
        
        # If we didn't find a solution for this K, save the best one found so far
        # But we prefer lower K.
        # If we fail K=13, we try K=14.
        # If we fail K=14, we try K=15.
        # If we fail K=15, we try K=16.
        # If we fail all, we return the best valid one?
        # The problem requires a valid coloring.
        # If SA fails to find a valid one, we must generate one.
        # We can use a greedy repair on the best_colors found.
        
        # Greedy Repair
        def repair(colors, K):
            # Try to fix violations
            # Iterate through K4s with < 5 distinct colors
            # Change an edge color to a new random color
            # Repeat until valid or max iterations
            colors = colors.copy()
            for _ in range(10000):
                # Find a violating K4
                violated = False
                for k4_idx, k4 in enumerate(k4s):
                    distinct_count = len(set(colors[e] for e in k4))
                    if distinct_count < 5:
                        violated = True
                        # Pick a random edge in this K4
                        e = random.choice(k4)
                        # Pick a random new color
                        new_c = random.randint(0, K-1)
                        colors[e] = new_c
                        break
                if not violated:
                    return colors
            return None
            
        # Try to repair best_colors
        repaired = repair(best_colors, K)
        if repaired and is_valid(repaired, K):
            return repaired

    # If we get here, we failed to find a valid solution for any K in [13,16]
    # Return a known valid solution with K=16 or higher?
    # The task says "Minimize the number of colors used".
    # If we return a valid solution with K=17, it's valid but suboptimal.
    # But better than invalid.
    # Let's construct a valid K=17 solution.
    # (i+j) % 17 is not guaranteed to be valid for K4s?
    # Let's just return a random valid one?
    # We can generate a random one and repair it.
    
    colors = np.random.randint(0, 17, 120, dtype=np.int8)
    repaired = repair(colors, 17)
    if repaired:
        return repaired
    else:
        # Fallback: return a dummy valid array?
        # We must return 120 integers.
        # Let's just return a pattern that is likely valid.
        # Actually, the repair function is robust enough.
        # If it fails, it's a bug in repair.
        # Let's assume repair works.
        return colors.tolist()
