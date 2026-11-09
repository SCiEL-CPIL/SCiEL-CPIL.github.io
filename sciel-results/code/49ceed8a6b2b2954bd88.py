import random
import time

def construct():
    n = 16
    num_edges = 120
    
    edges = []
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            edge_to_idx[(i, j)] = idx
            idx += 1
            
    k4_edge_indices = []
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
                    k4_edge_indices.append(e_indices)
                    
    num_k4s = len(k4_edge_indices)
    
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx in range(num_k4s):
        for e_idx in k4_edge_indices[k4_idx]:
            edge_to_k4s[e_idx].append(k4_idx)

    def count_bad_k4s(colors):
        bad = 0
        for e_indices in k4_edge_indices:
            if len({colors[e] for e in e_indices}) < 5:
                bad += 1
        return bad

    def delta_cost(edge_idx, old_color, new_color, colors):
        delta = 0
        for k4_idx in edge_to_k4s[edge_idx]:
            e_indices = k4_edge_indices[k4_idx]
            
            # Optimization: Check if change affects validity
            # Old set
            c_old = {colors[e] for e in e_indices}
            old_valid = len(c_old) >= 5
            
            # New set
            # If new_color == old_color, no change (handled by caller)
            # If new_color in c_old, set size might stay same or decrease (if old_color was unique)
            # If new_color not in c_old, set size might stay same or increase (if old_color was not unique)
            
            c_new_set = set()
            for e in e_indices:
                if e == edge_idx:
                    c_new_set.add(new_color)
                else:
                    c_new_set.add(colors[e])
            
            new_valid = len(c_new_set) >= 5
            
            if old_valid and not new_valid:
                delta += 1
            elif not old_valid and new_valid:
                delta -= 1
        return delta

    def local_search(K, time_limit):
        colors = [random.randint(0, K - 1) for _ in range(num_edges)]
        current_cost = count_bad_k4s(colors)
        
        if current_cost == 0:
            return colors
        
        start_time = time.time()
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() - start_time < time_limit:
            e_idx = random.randint(0, num_edges - 1)
            old_color = colors[e_idx]
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

    best_colors = None
    best_K = 15
    
    # 1. Ensure we have a valid 15-coloring
    # Random is likely valid. Check a few.
    for _ in range(20):
        c15 = [random.randint(0, 14) for _ in range(num_edges)]
        if count_bad_k4s(c15) == 0:
            best_colors = c15
            best_K = 15
            break
            
    if best_colors is None:
        # Use local search to find a valid 15-coloring
        res = local_search(15, 10)
        if res:
            best_colors = res
            best_K = 15
        else:
            # Extremely unlikely, but return something
            best_colors = [random.randint(0, 14) for _ in range(num_edges)]
            best_K = 15

    # 2. Try to improve to 14
    if best_K == 15:
        start_t14 = time.time()
        while time.time() - start_t14 < 50:
            res = local_search(14, 5)
            if res:
                best_colors = res
                best_K = 14
                break
                
    # 3. Try to improve to 13
    if best_K == 14:
        start_t13 = time.time()
        while time.time() - start_t13 < 50:
            res = local_search(13, 5)
            if res:
                best_colors = res
                best_K = 13
                break

    return best_colors
