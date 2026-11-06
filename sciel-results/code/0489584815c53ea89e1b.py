import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140
    n = 16
    num_edges = 120
    
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * n - i * (i + 1) // 2 + (j - i - 1)
    
    k4s = list(combinations(range(n), 4))
    num_k4s = len(k4s)
    
    k4_edge_indices = []
    for quad in k4s:
        edges = list(combinations(quad, 2))
        k4_edge_indices.append([edge_idx(i, j) for i, j in edges])
    
    def count_violations(colours):
        viol = 0
        for eidx in k4_edge_indices:
            c0 = colours[eidx[0]]
            c1 = colours[eidx[1]]
            c2 = colours[eidx[2]]
            c3 = colours[eidx[3]]
            c4 = colours[eidx[4]]
            c5 = colours[eidx[5]]
            d = 1
            if c1 != c0: d += 1
            if c2 != c0 and c2 != c1: d += 1
            if c3 != c0 and c3 != c1 and c3 != c2: d += 1
            if c4 != c0 and c4 != c1 and c4 != c2 and c4 != c3: d += 1
            if c5 != c0 and c5 != c1 and c5 != c2 and c5 != c3 and c5 != c4: d += 1
            if d < 5:
                viol += 1
        return viol
    
    best_valid = None
    best_valid_dc = float('inf')
    
    for k in [13, 14, 15]:
        if time.time() > deadline:
            break
        
        for restart in range(5):
            if time.time() > deadline:
                break
            
            colours = [random.randint(0, k-1) for _ in range(num_edges)]
            current_viol = count_violations(colours)
            
            if current_viol == 0:
                dc = len(set(colours))
                if dc < best_valid_dc:
                    best_valid_dc = dc
                    best_valid = colours[:]
                continue
            
            for step in range(3000):
                if time.time() > deadline:
                    break
                
                if current_viol == 0:
                    dc = len(set(colours))
                    if dc < best_valid_dc:
                        best_valid_dc = dc
                        best_valid = colours[:]
                    break
                
                e = random.randint(0, num_edges - 1)
                old_c = colours[e]
                
                best_v = current_viol
                best_c = old_c
                for _ in range(10):
                    new_c = random.randint(0, k-1)
                    if new_c == old_c:
                        continue
                    colours[e] = new_c
                    v = count_violations(colours)
                    if v < best_v:
                        best_v = v
                        best_c = new_c
                colours[e] = best_c
                current_viol = best_v
    
    if best_valid is not None:
        return best_valid
    return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
