import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 140
    n = 16
    edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    m = len(edges)
    edge_idx = {e: idx for idx, e in enumerate(edges)}
    
    k4s = []
    for quad in combinations(range(n), 4):
        e_list = []
        for a, b in combinations(quad, 2):
            e_list.append(edge_idx[(a,b)])
        k4s.append(e_list)
    num_k4s = len(k4s)
    
    edge_to_k4s = [[] for _ in range(m)]
    for k4_idx, e_list in enumerate(k4s):
        for e in e_list:
            edge_to_k4s[e].append(k4_idx)
    
    def init_state(colors):
        k4_color_counts = [{} for _ in range(num_k4s)]
        k4_distinct = [0] * num_k4s
        for k4_idx, e_list in enumerate(k4s):
            cc = k4_color_counts[k4_idx]
            for e in e_list:
                c = colors[e]
                if c not in cc:
                    cc[c] = 1
                    k4_distinct[k4_idx] += 1
                else:
                    cc[c] += 1
        viol = sum(1 for d in k4_distinct if d < 5)
        return k4_color_counts, k4_distinct, viol
    
    k = 15
    random.seed(42)
    colors = [random.randrange(k) for _ in range(m)]
    k4_color_counts, k4_distinct, viol = init_state(colors)
    
    best_viol = viol
    best_colors = colors[:]
    
    for it in range(500000):
        if time.time() > deadline - 3:
            break
        if viol == 0:
            return colors
        
        viol_k4s = [i for i, d in enumerate(k4_distinct) if d < 5]
        if not viol_k4s:
            return colors
        k4_idx = random.choice(viol_k4s)
        e_list = k4s[k4_idx]
        e = random.choice(e_list)
        old_c = colors[e]
        new_c = random.randrange(k)
        
        # Apply change
        colors[e] = new_c
        delta_viol = 0
        for k4_i in edge_to_k4s[e]:
            cc = k4_color_counts[k4_i]
            old_d = k4_distinct[k4_i]
            old_viol = 1 if old_d < 5 else 0
            cc[old_c] -= 1
            if cc[old_c] == 0:
                del cc[old_c]
                old_d -= 1
            if new_c not in cc:
                cc[new_c] = 1
                old_d += 1
            else:
                cc[new_c] += 1
            k4_distinct[k4_i] = old_d
            new_viol = 1 if old_d < 5 else 0
            delta_viol += new_viol - old_viol
        
        new_viol_count = viol + delta_viol
        if new_viol_count <= viol or random.random() < 0.05:
            viol = new_viol_count
            if viol < best_viol:
                best_viol = viol
                best_colors = colors[:]
        else:
            # Revert
            colors[e] = old_c
            for k4_i in edge_to_k4s[e]:
                cc = k4_color_counts[k4_i]
                old_d = k4_distinct[k4_i]
                cc[new_c] -= 1
                if cc[new_c] == 0:
                    del cc[new_c]
                    old_d -= 1
                if old_c not in cc:
                    cc[old_c] = 1
                    old_d += 1
                else:
                    cc[old_c] += 1
                k4_distinct[k4_i] = old_d
    
    if viol == 0:
        return colors
    if best_viol == 0:
        return best_colors
    return [(i+j) % 17 for i in range(n) for j in range(i+1, n)]
