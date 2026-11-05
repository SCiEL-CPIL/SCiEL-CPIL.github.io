import time
import random
import math
import itertools
import numpy as np

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if n != 7:
        return _standard_construct(n)
        
    # 1. Generate permutations of 1..6
    perms = list(itertools.permutations(range(1, 7)))
    num_perms = len(perms)  # 720
    
    # 2. Compute Overlap Matrix M
    M = np.zeros((num_perms, num_perms), dtype=np.int8)
    
    for i in range(num_perms):
        p1 = perms[i]
        for j in range(num_perms):
            p2 = perms[j]
            ov = 0
            if p1[1] == p2[0] and p1[2] == p2[1] and p1[3] == p2[2] and p1[4] == p2[3] and p1[5] == p2[4]:
                ov = 5
            elif p1[2] == p2[0] and p1[3] == p2[1] and p1[4] == p2[2] and p1[5] == p2[3]:
                ov = 4
            elif p1[3] == p2[0] and p1[4] == p2[1] and p1[5] == p2[2]:
                ov = 3
            elif p1[4] == p2[0] and p1[5] == p2[1]:
                ov = 2
            elif p1[5] == p2[0]:
                ov = 1
            M[i, j] = ov
    
    M_list = M.tolist()
    
    # 3. Greedy Initialization (Nearest Neighbor)
    order = [0]
    used = [False] * num_perms
    used[0] = True
    for _ in range(num_perms - 1):
        last = order[-1]
        best_next = -1
        best_ov = -1
        for j in range(num_perms):
            if not used[j]:
                ov = M_list[last][j]
                if ov > best_ov:
                    best_ov = ov
                    best_next = j
        order.append(best_next)
        used[best_next] = True
    
    def calc_total_overlap(ord_list):
        s = 0
        for i in range(len(ord_list) - 1):
            s += M_list[ord_list[i]][ord_list[i + 1]]
        return s
    
    current_overlap = calc_total_overlap(order)
    best_order = order[:]
    best_overlap = current_overlap
    
    # 4. Fast 2-Opt Local Search using NumPy
    # Precompute row slices for faster access
    M_rows = [M[i] for i in range(num_perms)]  # List of 1D arrays
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        for i in range(num_perms - 2):
            if time.time() > deadline - 5.0:
                break
            
            u = order[i]
            v = order[i + 1]
            old_uv = M_list[u][v]
            
            best_j = -1
            best_delta = 0
            
            # Vectorized: compute delta for all j in [i+2, num_perms-2]
            if i + 2 < num_perms - 1:
                js = list(range(i + 2, num_perms - 1))
                x_vals = np.array(order[js], dtype=np.int32)
                y_vals = np.array(order[js + 1], dtype=np.int32)
                
                # M[u, x] for all x in x_vals
                M_u_x = M_rows[u][x_vals]
                # M[v, y] for all y in y_vals  
                M_v_y = M_rows[v][y_vals]
                # M[x, y] for all pairs
                M_x_y = np.array([M_list[x][y] for x, y in zip(x_vals, y_vals)], dtype=np.int32)
                
                d = M_u_x + M_v_y - old_uv - M_x_y
                
                max_d = d.max()
                if max_d > 0:
                    best_idx = d.argmax()
                    best_delta = max_d
                    best_j = js[best_idx]
            
            if best_j > 0 and best_delta > 0:
                order[i + 1:best_j + 1] = reversed(order[i + 1:best_j + 1])
                current_overlap += best_delta
                improved = True
                if current_overlap > best_overlap:
                    best_overlap = current_overlap
                    best_order = order[:]
        
        # Adjacent swap polish
        if not improved and time.time() < deadline - 3.0:
            for i in range(num_perms - 1):
                if time.time() > deadline - 3.0:
                    break
                a = order[i]
                b = order[i + 1]
                old_ab = M_list[a][b]
                new_ba = M_list[b][a]
                
                if new_ba > old_ab:
                    delta = new_ba - old_ab
                    if i > 0:
                        delta += M_list[order[i - 1]][b] - M_list[order[i - 1]][a]
                    if i + 1 < num_perms - 1:
                        delta += M_list[b][order[i + 2]] - M_list[a][order[i + 2]]
                    
                    if delta > 0:
                        order[i], order[i + 1] = b, a
                        current_overlap += delta
                        improved = True
                        if current_overlap > best_overlap:
                            best_overlap = current_overlap
                            best_order = order[:]
    
    # 5. Construct Word
    word = []
    for idx, p_idx in enumerate(best_order):
        p = perms[p_idx]
        block = list(p) + [7] + list(p)
        if idx == 0:
            word = block
        else:
            ov = M_list[best_order[idx - 1]][best_order[idx]]
            word += block[ov:]
    
    return "".join(map(str, word))

def _standard_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
