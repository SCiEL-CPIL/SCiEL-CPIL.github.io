import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 128
    
    # Generate seed solution
    def get_seed():
        colors = []
        for x in range(1, n+1):
            for color in range(128):
                valid = True
                for d in range(1, (x-1)//3 + 1):
                    p1 = x-1-3*d
                    p2 = x-1-2*d
                    p3 = x-1-d
                    if p1 < 0:
                        continue
                    c1 = colors[p1]
                    c2 = colors[p2]
                    c3 = colors[p3]
                    if len({c1, c2, c3, color}) < 3:
                        valid = False
                        break
                if valid:
                    colors.append(color)
                    break
        return colors

    seed_colors = get_seed()
    seed_count = len(set(seed_colors))
    
    if seed_count == 3:
        return seed_colors

    # Precompute APs
    aps = []
    idx_to_aps = [[] for _ in range(n)]
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = b + d
            e = c + d
            ap_idx = len(aps)
            aps.append((a, b, c, e))
            idx_to_aps[a].append(ap_idx)
            idx_to_aps[b].append(ap_idx)
            idx_to_aps[c].append(ap_idx)
            idx_to_aps[e].append(ap_idx)
            
    num_aps = len(aps)
    
    def count_violations(colors):
        viol = 0
        for a, b, c, e in aps:
            if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                viol += 1
        return viol
        
    def local_search(k, time_limit):
        np.random.seed(int(time.time() * 1000) % 100000)
        current = np.random.randint(0, k, n)
        current_viol = count_violations(current)
        best = current.copy()
        best_viol = current_viol
        
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        
        while time.time() < time_limit:
            idx = np.random.randint(n)
            old_c = current[idx]
            new_c = np.random.randint(k)
            if new_c == old_c:
                continue
                
            delta = 0
            for ap_idx in idx_to_aps[idx]:
                a, b, c, e = aps[ap_idx]
                c_a = current[a]
                c_b = current[b]
                c_c = current[c]
                c_e = current[e]
                
                was_viol = len({c_a, c_b, c_c, c_e}) < 3
                
                if a == idx: c_a = new_c
                elif b == idx: c_b = new_c
                elif c == idx: c_c = new_c
                elif e == idx: c_e = new_c
                
                is_viol = len({c_a, c_b, c_c, c_e}) < 3
                
                if was_viol and not is_viol:
                    delta -= 1
                elif not was_viol and is_viol:
                    delta += 1
            
            if delta < 0:
                current[idx] = new_c
                current_viol += delta
                if current_viol < best_viol:
                    best = current.copy()
                    best_viol = current_viol
                    if best_viol == 0:
                        return best.tolist()
            else:
                if random.random() < np.exp(-delta / T):
                    current[idx] = new_c
                    current_viol += delta
            T = max(T_min, T * cooling)
            
        return None

    # Try to find better coloring
    remaining = deadline - time.time()
    
    # Try 3 colors
    if remaining > 5:
        t_limit = time.time() + remaining * 0.7
        res = local_search(3, t_limit)
        if res:
            return res
            
    remaining = deadline - time.time()
    if remaining > 3 and seed_count > 4:
        t_limit = time.time() + remaining * 0.8
        res = local_search(4, t_limit)
        if res:
            return res
            
    remaining = deadline - time.time()
    if remaining > 1 and seed_count > 5:
        t_limit = time.time() + remaining
        res = local_search(5, t_limit)
        if res:
            return res
            
    return seed_colors
