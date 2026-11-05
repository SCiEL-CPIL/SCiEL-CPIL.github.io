import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute has_c6
    has_c6 = [False] * (1 << 15)
    k6_edges = [(i, j) for i in range(6) for j in range(i+1, 6)]
    local_edge_bit = {}
    for bit_pos, (u, v) in enumerate(k6_edges):
        local_edge_bit[(u, v)] = bit_pos
        local_edge_bit[(v, u)] = bit_pos
        
    for mask in range(1 << 15):
        adj = [0] * 6
        for k in range(15):
            if mask & (1 << k):
                u, v = k6_edges[k]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        c6_found = False
        for v in range(6):
            if bin(adj[v]).count('1') < 2:
                break
        else:
            from itertools import permutations
            for p in permutations(range(1, 6)):
                if (adj[0] & (1 << p[0])) and (adj[p[0]] & (1 << p[1])) and \
                   (adj[p[1]] & (1 << p[2])) and (adj[p[2]] & (1 << p[3])) and \
                   (adj[p[3]] & (1 << p[4])) and (adj[p[4]] & (1 << 0)):
                    c6_found = True
                    break
        has_c6[mask] = c6_found

    n = 17
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    m = len(edges)
    edge_index = {(i, j): idx for idx, (i, j) in enumerate(edges)}
    
    six_sets = list(combinations(range(n), 6))
    
    # Precompute edge_to_sets and set_edges
    edge_to_sets = [[] for _ in range(m)]
    set_edges = []
    
    for s_idx, s in enumerate(six_sets):
        s_edges = []
        for u_idx in range(6):
            for v_idx in range(u_idx + 1, 6):
                u = s[u_idx]
                v = s[v_idx]
                g_idx = edge_index[(u, v)]
                bit_pos = local_edge_bit[(u_idx, v_idx)]
                edge_to_sets[g_idx].append((s_idx, bit_pos))
                s_edges.append((g_idx, bit_pos))
        set_edges.append(s_edges)
        
    # Initialize random coloring
    colors = [random.randint(0, 3) for _ in range(m)]
    
    # Initialize masks
    masks = [[0]*4 for _ in range(len(six_sets))]
    for s_idx, s_edges_list in enumerate(set_edges):
        for g_idx, bit_pos in s_edges_list:
            c = colors[g_idx]
            masks[s_idx][c] |= (1 << bit_pos)
            
    # Initial violation count
    v_count = 0
    for s_idx in range(len(six_sets)):
        for c in range(4):
            if has_c6[masks[s_idx][c]]:
                v_count += 1
                
    best_v = v_count
    best_colors = colors[:]
    
    T = 1.0
    
    while time.time() < deadline:
        if v_count == 0:
            return {'n': n, 'colors': colors}
            
        e_idx = random.randint(0, m - 1)
        c_old = colors[e_idx]
        c_new = random.choice([c for c in range(4) if c != c_old])
        
        delta_e = 0
        
        for s_idx, bit_pos in edge_to_sets[e_idx]:
            # Update c_old
            old_mask_c_old = masks[s_idx][c_old]
            new_mask_c_old = old_mask_c_old & ~(1 << bit_pos)
            
            if has_c6[old_mask_c_old]:
                delta_e -= 1
            if has_c6[new_mask_c_old]:
                delta_e += 1
                
            # Update c_new
            old_mask_c_new = masks[s_idx][c_new]
            new_mask_c_new = old_mask_c_new | (1 << bit_pos)
            
            if has_c6[old_mask_c_new]:
                delta_e -= 1
            if has_c6[new_mask_c_new]:
                delta_e += 1
                
            # Apply updates to masks
            masks[s_idx][c_old] = new_mask_c_old
            masks[s_idx][c_new] = new_mask_c_new
            
        # Accept
        if delta_e <= 0 or random.random() < 2.71828 ** (-delta_e / T):
            colors[e_idx] = c_new
            v_count += delta_e
            if v_count < best_v:
                best_v = v_count
                best_colors = colors[:]
        else:
            # Revert
            for s_idx, bit_pos in edge_to_sets[e_idx]:
                masks[s_idx][c_old] |= (1 << bit_pos)
                masks[s_idx][c_new] &= ~(1 << bit_pos)
                
        T *= 0.99
        if T < 0.01:
            T = 0.01
            
    if best_v == 0:
        return {'n': n, 'colors': best_colors}
        
    # Return seed if no better found
    return {'n': 9, 'colors': [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]}
