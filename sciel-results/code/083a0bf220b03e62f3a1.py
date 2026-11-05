import time
import random
import math

def construct():
    # Greenwood-Gleason colouring of K16
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    def get_seed_coloring_32():
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return colors

    def get_idx_map(n):
        idx_map = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                idx_map[i][j] = idx
                idx_map[j][i] = idx
                idx += 1
        return idx_map

    def count_triangles(n, C, idx_map):
        M = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = C[idx_map[i][j]]
                M[c][i] |= 1 << j
                M[c][j] |= 1 << i
        
        total = 0
        for c in range(4):
            Mc = M[c]
            for i in range(n):
                for j in range(i + 1, n):
                    total += (Mc[i] & Mc[j]).bit_count()
        return total // 3

    # Initialize
    best_n = 32
    best_C = get_seed_coloring_32()
    
    current_n = 33
    # Extend best_C to current_n
    C = best_C[:]
    new_v = current_n - 1
    for i in range(new_v):
        C.append(random.randint(0, 3))
    
    idx_map = get_idx_map(current_n)
    T = count_triangles(current_n, C, idx_map)
    
    deadline = time.time() + 110.0
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.001
    cool_rate = 0.9995
    current_temp = T_start
    
    while time.time() < deadline:
        if T == 0:
            best_n = current_n
            best_C = C[:]
            current_n += 1
            if current_n > 62:
                break
            
            # Extend
            C = best_C[:]
            new_v = current_n - 1
            for i in range(new_v):
                C.append(random.randint(0, 3))
            
            idx_map = get_idx_map(current_n)
            T = count_triangles(current_n, C, idx_map)
            current_temp = T_start
            continue
        
        # SA Moves
        # Batch size
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            
            i = random.randint(0, current_n - 1)
            j = random.randint(0, current_n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            idx = idx_map[i][j]
            c_old = C[idx]
            
            c_new = random.randint(0, 3)
            if c_new == c_old:
                continue
            
            delta_T = 0
            for k in range(current_n):
                if k == i or k == j:
                    continue
                
                c_ik = C[idx_map[i][k]]
                c_jk = C[idx_map[j][k]]
                
                if c_ik == c_old and c_jk == c_old:
                    delta_T -= 1
                if c_ik == c_new and c_jk == c_new:
                    delta_T += 1
            
            if delta_T < 0:
                accept = True
            else:
                if delta_T == 0:
                    accept = True
                else:
                    # Avoid underflow/overflow
                    if current_temp > 0:
                        accept = random.random() < math.exp(-delta_T / current_temp)
                    else:
                        accept = False
            
            if accept:
                C[idx] = c_new
                T += delta_T
        
        current_temp *= cool_rate
        if current_temp < T_end:
            if T > 0:
                current_temp = T_start
            # else: solved, handled at top of loop

    return {'n': best_n, 'colors': best_C}
