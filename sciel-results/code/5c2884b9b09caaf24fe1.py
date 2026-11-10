import time
import random

def construct():
    start_time = time.time()
    deadline = 120.0  # Keep safety margin
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Map (i, j) to index
    edge_idx = {}
    edges = []
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            edges.append((i, j))
            idx += 1
            
    # Initialize with seed coloring: (i + j) % 17
    colors = [0] * num_edges
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i + j) % 17
        
    # Precompute K4s containing each edge
    # k4s_containing_edge[e] = list of tuples of 4 vertices
    k4s_containing_edge = [[] for _ in range(num_edges)]
    all_k4s = []
    k4_indices = {}
    k4_idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    k4 = (a, b, c, d)
                    k4_indices[k4] = k4_idx
                    all_k4s.append(k4)
                    # Edges in this K4: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
                    e_ab = edge_idx[(a, b)]
                    e_ac = edge_idx[(a, c)]
                    e_ad = edge_idx[(a, d)]
                    e_bc = edge_idx[(b, c)]
                    e_bd = edge_idx[(b, d)]
                    e_cd = edge_idx[(c, d)]
                    k4s_containing_edge[e_ab].append(k4_idx)
                    k4s_containing_edge[e_ac].append(k4_idx)
                    k4s_containing_edge[e_ad].append(k4_idx)
                    k4s_containing_edge[e_bc].append(k4_idx)
                    k4s_containing_edge[e_bd].append(k4_idx)
                    k4s_containing_edge[e_cd].append(k4_idx)
                    k4_idx += 1
                    
    num_k4s = len(all_k4s)
    
    # Helper to count distinct colors in a K4
    def count_distinct_colors(k4_idx, cols):
        k4 = all_k4s[k4_idx]
        a, b, c, d = k4
        e1 = edge_idx[(a, b)]
        e2 = edge_idx[(a, c)]
        e3 = edge_idx[(a, d)]
        e4 = edge_idx[(b, c)]
        e5 = edge_idx[(b, d)]
        e6 = edge_idx[(c, d)]
        c_set = {cols[e1], cols[e2], cols[e3], cols[e4], cols[e5], cols[e6]}
        return len(c_set)
        
    # Calculate initial violations
    violations = 0
    for k in range(num_k4s):
        if count_distinct_colors(k, colors) < 5:
            violations += 1
            
    # Current best
    best_colors = colors.copy()
    best_violations = violations
    best_max_color = max(colors)
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # We want to minimize violations first, then max color.
    # Energy = violations * 1000 + max_color
    
    def get_energy(cols, max_c):
        # Recalculate violations is expensive if done globally.
        # We will track violations incrementally.
        pass
        
    # Incremental update strategy
    # Current state: colors, violations
    # Move: change edge e from c_old to c_new
    
    current_colors = colors.copy()
    current_violations = violations
    current_max_color = max(current_colors)
    
    # Count of each color to know if max color can be reduced
    color_counts = [0] * 17
    for c in current_colors:
        color_counts[c] += 1
        
    while time.time() - start_time < deadline:
        if T < min_T:
            break
            
        # Pick a random edge
        e = random.randint(0, num_edges - 1)
        c_old = current_colors[e]
        
        # Pick a new color
        # Bias towards lower colors to reduce max color
        # If we want to reduce max color, pick from 0..current_max_color-1
        # Otherwise, pick from 0..current_max_color
        if current_max_color > 0:
            # Try to reduce max color if possible
            if random.random() < 0.7:
                c_new = random.randint(0, current_max_color - 1)
            else:
                c_new = random.randint(0, current_max_color)
        else:
            c_new = 0
            
        if c_new == c_old:
            continue
            
        # Calculate delta violations
        # Affected K4s
        affected_k4s = k4s_containing_edge[e]
        
        old_violations_in_affected = 0
        new_violations_in_affected = 0
        
        # Temporarily change color to calculate new states
        # We can just calculate counts with c_old and c_new
        
        for k4_idx in affected_k4s:
            k4 = all_k4s[k4_idx]
            a, b, c, d = k4
            # Get indices of the 6 edges
            # One of them is e
            # We need the other 5
            # To avoid overhead, we can precompute the other 5 edges for each (k4_idx, e) pair?
            # Or just calculate.
            
            # Let's just use the count function logic but inline for speed
            # This part is the bottleneck.
            
            # Old state count
            # We need to check if this K4 was violating before.
            # And if it will be violating after.
            
            # Optimization: Precompute for each K4 the set of edge indices.
            # k4_edges[k4_idx] = list of 6 edge indices
            
        # Re-implement with precomputed k4_edges for speed
        # k4_edges = []
        # for k in range(num_k4s):
        #    ...
        
        # Let's add this precomputation
        # (Mental step: add precomputation in code)
        
        # For now, let's assume we can afford the calculation.
        # 91 K4s per move.
        # Each K4 check: 6 edge lookups, set creation, len.
        # ~10-20 ops.
        # 91 * 20 = 1800 ops per move.
        # 100k moves = 180M ops.
        # Python might take 10-30 seconds.
        # This is acceptable.
        
        # Calculate old violations for affected K4s
        for k4_idx in affected_k4s:
            if count_distinct_colors(k4_idx, current_colors) < 5:
                old_violations_in_affected += 1
                
        # Apply change
        current_colors[e] = c_new
        
        # Calculate new violations for affected K4s
        for k4_idx in affected_k4s:
            if count_distinct_colors(k4_idx, current_colors) < 5:
                new_violations_in_affected += 1
                
        # Revert change for delta calculation
        current_colors[e] = c_old
        
        delta_violations = new_violations_in_affected - old_violations_in_affected
        
        # Calculate delta max color
        delta_max_color = 0
        if c_new > current_max_color:
            delta_max_color = c_new - current_max_color
        elif c_old == current_max_color and color_counts[c_old] == 1:
            # If we remove the only instance of max color, max color might drop
            # Find new max
            # This is expensive to do exactly every time.
            # Approximation: if we remove max color, assume it drops to next highest present.
            # Or just check if c_new > c_old, etc.
            # For simplicity, let's just use violations as primary, and max color as secondary.
            # If violations don't change, prefer lower max color.
            pass
            
        # Energy delta
        # E = violations * 100 + max_color
        # We need to know new max color.
        # If c_new > c_old, new_max = max(current_max_color, c_new)
        # If c_old == current_max_color and color_counts[c_old] == 1, new_max = next highest
        # Else new_max = current_max_color
        
        new_max_color = current_max_color
        if c_new > current_max_color:
            new_max_color = c_new
        elif c_old == current_max_color and color_counts[c_old] == 1:
            # Find next highest color with count > 0
            for cc in range(current_max_color - 1, -1, -1):
                if color_counts[cc] > 0:
                    new_max_color = cc
                    break
            else:
                new_max_color = 0 # Should not happen if num_edges > 0
                
        delta_max_color = new_max_color - current_max_color
        
        delta_E = delta_violations * 100 + delta_max_color
        
        # Acceptance probability
        if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
            # Accept move
            current_colors[e] = c_new
            current_violations += delta_violations
            current_max_color = new_max_color
            color_counts[c_old] -= 1
            color_counts[c_new] += 1
            
            # Update best
            if current_violations == 0:
                if current_max_color < best_max_color or (current_max_color == best_max_color and current_violations < best_violations):
                    best_colors = current_colors.copy()
                    best_max_color = current_max_color
                    best_violations = 0
            elif current_violations < best_violations:
                best_colors = current_colors.copy()
                best_max_color = current_max_color
                best_violations = current_violations
                # If we find a valid solution, we can try to reduce colors further
                # But best_violations == 0 is the target
                
        T *= cooling
        
    # Return best valid solution
    # If best_violations > 0, we need to return a valid one.
    # The seed is valid.
    if best_violations > 0:
        # Return seed
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    else:
        return best_colors

# Need to add precomputation of k4_edges for speed
