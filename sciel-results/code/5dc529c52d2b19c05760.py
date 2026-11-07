import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    n = 256
    vertices = list(range(2, n + 1))
    N = len(vertices)
    
    vals = vertices
    
    # Build coprime graph adjacency masks
    adj_mask = [0] * N
    
    for i in range(N):
        vi = vals[i]
        mask = 0
        for j in range(N):
            if i != j:
                if math.gcd(vi, vals[j]) == 1:
                    mask |= (1 << j)
        adj_mask[i] = mask
        
    best_sum = 1
    best_set = {1}
    
    # Greedy heuristic to find initial solution and track the set
    def greedy_solve(candidates_mask):
        s = 0
        selected = set()
        c_mask = candidates_mask
        while c_mask:
            # Pick vertex with max weight
            best_v = -1
            best_w = -1
            temp = c_mask
            while temp:
                lb = temp & -temp
                idx = lb.bit_length() - 1
                if vals[idx] > best_w:
                    best_w = vals[idx]
                    best_v = idx
                temp ^= lb
            
            if best_v == -1:
                break
                
            s += best_w
            selected.add(vals[best_v])
            c_mask = c_mask & adj_mask[best_v]
            c_mask &= ~(1 << best_v)
            
        return s, selected

    # Initial greedy solution
    all_mask = (1 << N) - 1
    init_sum, init_set = greedy_solve(all_mask)
    if init_sum > best_sum:
        best_sum = init_sum
        best_set = init_set.copy()
    
    def bnb(candidates_mask, current_sum, current_set):
        nonlocal best_sum
        nonlocal best_set
        
        if time.time() > deadline:
            return
            
        if candidates_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            return
            
        # Upper bound calculation using greedy coloring
        cand_list = []
        temp = candidates_mask
        while temp:
            lb = temp & -temp
            idx = lb.bit_length() - 1
            cand_list.append(idx)
            temp ^= lb
            
        # Sort by weight descending for better coloring bound
        cand_list.sort(key=lambda x: vals[x], reverse=True)
        
        colors = {}
        max_weights_per_color = {}
        
        for v in cand_list:
            neigh_mask = adj_mask[v] & candidates_mask
            used_colors = set()
            temp_neigh = neigh_mask
            while temp_neigh:
                lb = temp_neigh & -temp_neigh
                n_idx = lb.bit_length() - 1
                if n_idx in colors:
                    used_colors.add(colors[n_idx])
                temp_neigh ^= lb
                
            c = 0
            while c in used_colors:
                c += 1
                
            colors[v] = c
            if c not in max_weights_per_color:
                max_weights_per_color[c] = vals[v]
            else:
                if vals[v] > max_weights_per_color[c]:
                    max_weights_per_color[c] = vals[v]
                    
        bound = sum(max_weights_per_color.values())
        
        if current_sum + bound <= best_sum:
            return
            
        # Branch on the highest weight vertex
        best_v = -1
        best_w = -1
        temp = candidates_mask
        while temp:
            lb = temp & -temp
            idx = lb.bit_length() - 1
            if vals[idx] > best_w:
                best_w = vals[idx]
                best_v = idx
            temp ^= lb
            
        if best_v == -1:
            return
            
        # Branch 1: Include best_v
        new_cand_include = candidates_mask & adj_mask[best_v]
        current_set.add(vals[best_v])
        bnb(new_cand_include, current_sum + vals[best_v], current_set)
        current_set.remove(vals[best_v])
        
        # Branch 2: Exclude best_v
        new_cand_exclude = candidates_mask & ~(1 << best_v)
        bnb(new_cand_exclude, current_sum, current_set)
        
    bnb(all_mask, 0, set())
    
    result = list(best_set)
    result.append(1)
    
    return [int(x) for x in result]
