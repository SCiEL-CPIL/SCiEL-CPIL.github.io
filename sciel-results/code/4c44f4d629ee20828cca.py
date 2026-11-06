import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    # Precompute K4s and edge mappings
    # Vertices 0..15
    # Edge index (i, j) -> i * 15 - i*(i-1)//2 + (j - i - 1) ? 
    # Simpler: 
    edge_to_idx = {}
    idx_to_edge = []
    idx = 0
    for i in range(16):
        for j in range(i + 1, 16):
            edge_to_idx[(i, j)] = idx
            idx_to_edge.append((i, j))
            idx += 1
    num_edges = 120
    
    k4s = []
    for v0 in range(16):
        for v1 in range(v0 + 1, 16):
            for v2 in range(v1 + 1, 16):
                for v3 in range(v2 + 1, 16):
                    # 6 edges
                    e0 = edge_to_idx[(v0, v1)]
                    e1 = edge_to_idx[(v0, v2)]
                    e2 = edge_to_idx[(v0, v3)]
                    e3 = edge_to_idx[(v1, v2)]
                    e4 = edge_to_idx[(v1, v3)]
                    e5 = edge_to_idx[(v2, v3)]
                    k4s.append((e0, e1, e2, e3, e4, e5))
    
    num_k4s = len(k4s)
    
    # Map edge to list of k4 indices containing it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, k4 in enumerate(k4s):
        for e in k4:
            edge_to_k4s[e].append(k4_idx)
    
    def is_bad_k4(colors, k4):
        c0, c1, c2, c3, c4, c5 = [colors[e] for e in k4]
        # Disjoint pairs: (0,3), (1,4), (2,5)
        m0 = (c0 == c3)
        m1 = (c1 == c4)
        m2 = (c2 == c5)
        return (m0 + m1 + m2) >= 2
    
    def count_bad(colors):
        bad_count = 0
        for k4 in k4s:
            if is_bad_k4(colors, k4):
                bad_count += 1
        return bad_count
        
    def local_search(K, time_limit):
        # Initialize random colors
        colors = [random.randint(0, K - 1) for _ in range(num_edges)]
        
        # Precompute bad status for each k4
        bad_status = [1 if is_bad_k4(colors, k4) else 0 for k4 in k4s]
        total_bad = sum(bad_status)
        
        best_colors = colors[:]
        best_bad = total_bad
        
        end_time = time.time() + time_limit
        while time.time() < end_time:
            if total_bad == 0:
                break
                
            # Pick a random bad k4
            bad_k4_indices = [i for i, b in enumerate(bad_status) if b == 1]
            if not bad_k4_indices:
                break
            k4_idx = random.choice(bad_k4_indices)
            k4 = k4s[k4_idx]
            
            # Pick a random edge in this k4 to change
            e_idx = random.choice(k4)
            old_color = colors[e_idx]
            
            # Try all other colors
            # Optimization: only try colors that might help
            # But K is small (13-16), so try all
            best_new_color = -1
            best_delta = 0
            best_new_total = total_bad
            
            for new_color in range(K):
                if new_color == old_color:
                    continue
                
                # Update status for k4s containing e_idx
                # We need to check how the change affects the badness of k4s containing e_idx
                # Only k4s containing e_idx can change their badness
                delta = 0
                for k4_i in edge_to_k4s[e_idx]:
                    old_bad = bad_status[k4_i]
                    # Temporarily change color to check new badness
                    colors[e_idx] = new_color
                    new_bad = 1 if is_bad_k4(colors, k4_i) else 0
                    delta += (new_bad - old_bad)
                    # Revert immediately for next trial? 
                    # No, we need to accumulate. 
                    # But we are changing colors[e_idx] in place.
                    # We should store the original bad status and restore?
                    # Actually, we just need the final state.
                    # But we are trying different new_colors.
                    # We must revert colors[e_idx] to old_color before trying the next new_color?
                    # Yes.
                
                # Revert
                colors[e_idx] = old_color
                
                if total_bad + delta < best_new_total:
                    best_new_total = total_bad + delta
                    best_new_color = new_color
                    best_delta = delta
            
            if best_new_color != -1:
                # Apply the best move
                colors[e_idx] = best_new_color
                # Update bad_status for affected k4s
                for k4_i in edge_to_k4s[e_idx]:
                    bad_status[k4_i] = 1 if is_bad_k4(colors, k4_i) else 0
                total_bad += best_delta
                
                if total_bad < best_bad:
                    best_bad = total_bad
                    best_colors = colors[:]
            else:
                # Random move to escape local minima
                new_color = random.randint(0, K - 1)
                if new_color != old_color:
                    colors[e_idx] = new_color
                    for k4_i in edge_to_k4s[e_idx]:
                        bad_status[k4_i] = 1 if is_bad_k4(colors, k4_i) else 0
                    # Recalculate total_bad
                    total_bad = sum(bad_status)
                    
                    if total_bad < best_bad:
                        best_bad = total_bad
                        best_colors = colors[:]
                    
        # Check if best solution is valid
        if best_bad == 0:
            return best_colors
        else:
            return None

    # Try K=13, 14, 15, 16
    for K in [13, 14, 15, 16]:
        if time.time() > deadline - 5:
            break
        time_limit = min(30, deadline - time.time() - 2)
        if time_limit <= 0:
            break
        result = local_search(K, time_limit)
        if result is not None:
            # Count distinct colors
            distinct_colors = len(set(result))
            # Map to 0..distinct_colors-1
            color_map = {}
            new_colors = []
            for c in result:
                if c not in color_map:
                    color_map[c] = len(color_map)
                new_colors.append(color_map[c])
            return new_colors
            
    # Fallback to seed solution (17 colors)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

