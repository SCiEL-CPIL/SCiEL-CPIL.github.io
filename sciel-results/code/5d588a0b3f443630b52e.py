import time
import random

def _check_no_k222(n, colors):
    adj = [[0] * n for _ in range(2)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    for c in range(2):
        for a in range(n):
            mask_a = adj[c][a]
            for b in range(a + 1, n):
                if not (mask_a & (1 << b)):
                    continue
                mask_b = adj[c][b]
                common = mask_a & mask_b
                if common.bit_count() < 4:
                    continue
                
                # Check if G[common] contains K_{2,2}
                # Extract vertices
                verts = []
                mask_temp = common
                while mask_temp:
                    lsb = mask_temp & -mask_temp
                    verts.append(lsb.bit_length() - 1)
                    mask_temp ^= lsb
                    
                len_v = len(verts)
                for i in range(len_v):
                    c_v = verts[i]
                    mask_c = adj[c][c_v]
                    for j in range(i + 1, len_v):
                        d_v = verts[j]
                        common_cd = mask_c & adj[c][d_v] & common
                        if common_cd.bit_count() >= 2:
                            return False
    return True

def _paley_coloring(n):
    squares = {x * x % n for x in range(1, n)}
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            colors.append(0 if diff in squares else 1)
    return colors

def _count_k222(n, colors):
    adj = [[0] * n for _ in range(2)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    count = 0
    for c in range(2):
        for a in range(n):
            mask_a = adj[c][a]
            for b in range(a + 1, n):
                if not (mask_a & (1 << b)):
                    continue
                mask_b = adj[c][b]
                common = mask_a & mask_b
                if common.bit_count() < 4:
                    continue
                
                verts = []
                mask_temp = common
                while mask_temp:
                    lsb = mask_temp & -mask_temp
                    verts.append(lsb.bit_length() - 1)
                    mask_temp ^= lsb
                    
                len_v = len(verts)
                for i in range(len_v):
                    c_v = verts[i]
                    mask_c = adj[c][c_v]
                    for j in range(i + 1, len_v):
                        d_v = verts[j]
                        common_cd = mask_c & adj[c][d_v] & common
                        k = common_cd.bit_count()
                        if k >= 2:
                            count += k * (k - 1) // 2
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = 29
    colors = _paley_coloring(n)
    
    if not _check_no_k222(n, colors):
        # Fallback
        return {'n': n, 'colors': colors}
        
    n_new = 30
    colors_new = colors.copy()
    for i in range(n):
        colors_new.append(random.randint(0, 1))
        
    if _check_no_k222(n_new, colors_new):
        return {'n': n_new, 'colors': colors_new}
        
    current_cost = _count_k222(n_new, colors_new)
    best_cost = current_cost
    best_colors = colors_new.copy()
    
    T = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        u = random.randint(0, n_new - 1)
        v = random.randint(0, n_new - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        idx = u * (2 * n_new - u - 1) // 2 + (v - u - 1)
        
        colors_new[idx] = 1 - colors_new[idx]
        new_cost = _count_k222(n_new, colors_new)
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < pow(2.71828, -delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors_new.copy()
                if best_cost == 0:
                    return {'n': n_new, 'colors': best_colors}
        else:
            colors_new[idx] = 1 - colors_new[idx]
            
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    if best_cost == 0:
        return {'n': n_new, 'colors': best_colors}
    else:
        return {'n': n, 'colors': colors}
