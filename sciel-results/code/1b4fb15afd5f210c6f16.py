import random
import time
import sys

def construct():
    n = 16
    num_edges = 120
    # Map (i, j) to index
    # Index = i * (n - i - 1) // 2 + (j - i - 1) is not quite right for simple loop
    # Let's just generate list of edges and map
    edges = []
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            edge_to_idx[(i, j)] = idx
            idx += 1
            
    # Precompute K4s
    # A K4 is a set of 4 vertices {v0, v1, v2, v3}
    # It contains 6 edges: (v0,v1), (v0,v2), (v0,v3), (v1,v2), (v1,v3), (v2,v3)
    k4s = []
    k4_edge_indices = []
    
    # Generate all combinations of 4 vertices
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            for v2 in range(v1 + 1, n):
                for v3 in range(v2 + 1, n):
                    verts = [v0, v1, v2, v3]
                    e_indices = []
                    for a in range(4):
                        for b in range(a + 1, 4):
                            e = (verts[a], verts[b])
                            e_indices.append(edge_to_idx[e])
                    k4s.append(verts)
                    k4_edge_indices.append(e_indices)
                    
    num_k4s = len(k4s)
    
    # Map edge index to list of k4 indices containing it
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx in range(num_k4s):
        for e_idx in k4_edge_indices[k4_idx]:
            edge_to_k4s[e_idx].append(k4_idx)

    def get_k4_colors(e_indices, colors):
        return [colors[e] for e in e_indices]

    def is_valid_k4(colors_6):
        # Check if at least 5 distinct colors
        return len(set(colors_6)) >= 5

    def count_bad_k4s(colors):
        bad = 0
        for k4_idx in range(num_k4s):
            e_indices = k4_edge_indices[k4_idx]
            c = [colors[e] for e in e_indices]
            if not is_valid_k4(c):
                bad += 1
        return bad

    def delta_cost(edge_idx, old_color, new_color, colors):
        # Calculate change in number of bad K4s
        # Only K4s containing edge_idx are affected
        delta = 0
        for k4_idx in edge_to_k4s[edge_idx]:
            e_indices = k4_edge_indices[k4_idx]
            # Get current colors for this K4
            c_old = [colors[e] for e in e_indices]
            # Get new colors for this K4
            c_new = c_old.copy()
            for e in e_indices:
                if e == edge_idx:
                    c_new[e_indices.index(e)] = new_color
            # Actually, we need to be careful with indices in c_new
            # Let's just build the list properly
            c_new_list = []
            for e in e_indices:
                if e == edge_idx:
                    c_new_list.append(new_color)
                else:
                    c_new_list.append(colors[e])
            
            old_valid = is_valid_k4(c_old)
            new_valid = is_valid_k4(c_new_list)
            
            if old_valid and not new_valid:
                delta += 1
            elif not old_valid and new_valid:
                delta -= 1
        return delta

    def local_search(K, time_limit=10):
        # Initialize random colors
        colors = [random.randint(0, K - 1) for _ in range(num_edges)]
        current_cost = count_bad_k4s(colors)
        
        if current_cost == 0:
            return colors
        
        start_time = time.time()
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() - start_time < time_limit:
            # Pick random edge
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
            
            # Pick new color
            new_color = random.randint(0, K - 1)
            if new_color == old_color:
                continue
            
            d_cost = delta_cost(e_idx, old_color, new_color, colors)
            
            if d_cost <= 0 or random.random() < pow(2.71828, -d_cost / T):
                colors[e_idx] = new_color
                current_cost += d_cost
                
                if current_cost == 0:
                    return colors
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return None

    # Try K = 15, 14, 13, 16
    # 15 is most likely to succeed quickly
    # We have 120s total.
    # Allocate time for each K.
    # If 15 fails, try 14.
    
    # Fallback solution (17 colors)
    fallback = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    # Strategy:
    # 1. Try K=15 for 30s
    # 2. Try K=14 for 30s
    # 3. Try K=13 for 30s
    # 4. Try K=16 for 20s
    # If any succeeds, return.
    # Else return fallback.
    
    # To increase success rate, do multiple restarts within the time limit.
    
    def try_K(K, time_limit):
        start = time.time()
        while time.time() - start < time_limit:
            res = local_search(K, time_limit=1) # 1s per run? 
            # Better: pass remaining time
            remaining = time_limit - (time.time() - start)
            if remaining <= 0:
                break
            res = local_search(K, time_limit=remaining)
            if res is not None:
                # Verify
                if count_bad_k4s(res) == 0:
                    return res
        return None

    # It's better to run local_search with the full remaining time in one go?
    # Simulated Annealing benefits from long runs.
    # Let's do 3 restarts of ~10s for K=15.
    
    result = None
    for K in [15, 14, 13, 16]:
        # Allocate time based on likelihood
        # 15 is very likely. 14 less. 13 rare. 16 easy.
        t_limit = 20 if K == 15 else 10
        result = try_K(K, t_limit)
        if result:
            break
            
    if result:
        return result
    else:
        return fallback
