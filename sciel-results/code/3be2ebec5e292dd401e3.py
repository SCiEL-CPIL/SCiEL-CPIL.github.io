import time
import random
import numpy as np
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def generate_pairs(n):
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        return pairs

    def precompute_c6s(n):
        pairs = generate_pairs(n)
        pair_to_idx = {p: i for i, p in enumerate(pairs)}
        c6s = []
        for verts in combinations(range(n), 6):
            v0 = verts[0]
            rest = list(verts[1:])
            seen = set()
            for perm in permutations(rest):
                cycle = (v0,) + perm
                min_pos = cycle.index(min(cycle))
                norm = cycle[min_pos:] + cycle[:min_pos]
                rev = (norm[0],) + tuple(norm[1:][::-1])
                key = min(norm, rev)
                if key in seen:
                    continue
                seen.add(key)
                edges = []
                for k in range(6):
                    a, b = cycle[k], cycle[(k+1) % 6]
                    if a > b: a, b = b, a
                    edges.append(pair_to_idx[(a, b)])
                c6s.append(edges)
        return np.array(c6s, dtype=np.int32)

    def solve_n(n, time_limit):
        pairs = generate_pairs(n)
        num_edges = len(pairs)
        
        c6_array = precompute_c6s(n)
        num_c6s = len(c6_array)
        
        if num_c6s == 0:
            return {'n': n, 'colors': [0] * num_edges}
        
        best_result = None
        
        while time.time() < deadline - 5:
            colors = np.random.randint(0, 4, size=num_edges, dtype=np.int8)
            
            for step in range(2000):
                if time.time() > deadline - 5:
                    break
                
                c6_colors = colors[c6_array]
                mono = np.all(c6_colors == c6_colors[:, 0:1], axis=1)
                num_mono = int(np.sum(mono))
                
                if num_mono == 0:
                    best_result = {'n': n, 'colors': colors.tolist()}
                    break
                
                e = np.random.randint(num_edges)
                c_old = int(colors[e])
                
                mask = np.any(c6_array == e, axis=1)
                if not np.any(mask):
                    continue
                
                best_c = c_old
                best_mono = num_mono
                
                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    colors[e] = c_new
                    c6_colors_new = colors[c6_array]
                    total_mono = int(np.sum(np.all(c6_colors_new == c6_colors_new[:, 0:1], axis=1)))
                    if total_mono < best_mono:
                        best_mono = total_mono
                        best_c = c_new
                colors[e] = best_c
            if best_result:
                break
        
        return best_result

    # Try n=17 first (known feasible), then try larger
    for n in [17, 18, 19, 20]:
        if time.time() > deadline - 10:
            break
        res = solve_n(n, deadline)
        if res:
            return res
    
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
