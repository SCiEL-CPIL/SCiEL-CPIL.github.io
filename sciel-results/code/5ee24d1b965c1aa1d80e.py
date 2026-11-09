import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    
    def idx(xa, xb, ya, yb):
        return (xa * 7 + xb) * 7 * 7 + ya * 7 + yb
        
    def unidx(i):
        yb = i % 7
        i //= 7
        ya = i % 7
        i //= 7
        xb = i % 7
        i //= 7
        xa = i % 7
        return xa, xb, ya, yb

    num_points = 2401
    
    def f_add(e1, e2):
        a1, b1 = e1 % 7, e1 // 7
        a2, b2 = e2 % 7, e2 // 7
        return ((a1 + a2) % 7) + ((b1 + b2) % 7) * 7

    def f_mul(e1, e2):
        a1, b1 = e1 % 7, e1 // 7
        a2, b2 = e2 % 7, e2 // 7
        a = (a1 * a2 + 3 * b1 * b2) % 7
        b = (a1 * b2 + a2 * b1) % 7
        return a + b * 7

    dir_list = []
    for ma in range(p):
        for mb in range(p):
            m = ma + mb * 7
            vx = 1
            vy = m
            dir_list.append((vx, vy))
    dir_list.append((0, 1))
    
    lines = []
    point_to_lines = [[] for _ in range(num_points)]
    seen_lines = set()
    
    for i in range(num_points):
        xa, xb, ya, yb = unidx(i)
        x = (xa + xb * 7, ya + yb * 7)
        
        for d_idx, (vx, vy) in enumerate(dir_list):
            points_on_line = []
            for ta in range(p):
                for tb in range(p):
                    t = ta + tb * 7
                    tx = f_mul(t, vx)
                    ty = f_mul(t, vy)
                    nx = f_add(x[0], tx)
                    ny = f_add(x[1], ty)
                    na, nb = nx % 7, nx // 7
                    pa, pb = ny % 7, ny // 7
                    p_idx = idx(na, nb, pa, pb)
                    points_on_line.append(p_idx)
            
            line_key = tuple(sorted(points_on_line))
            if line_key not in seen_lines:
                seen_lines.add(line_key)
                line_idx = len(lines)
                lines.append(line_key)
                for p_idx in line_key:
                    point_to_lines[p_idx].append(line_idx)
    
    lookup = [dict() for _ in range(num_points)]
    for y in range(num_points):
        for bit_pos, L_idx in enumerate(point_to_lines[y]):
            lookup[y][L_idx] = bit_pos
    
    impact = [None] * num_points
    for x in range(num_points):
        imp_list = []
        for L_idx in point_to_lines[x]:
            for y in lines[L_idx]:
                if y == x:
                    continue
                bit_pos = lookup[y][L_idx]
                imp_list.append((y, bit_pos))
        impact[x] = imp_list

    def run_greedy(order_seed):
        random.seed(order_seed)
        in_N = np.ones(num_points, dtype=bool)
        witnesses = [(1 << 50) - 1] * num_points
        
        order = list(range(num_points))
        random.shuffle(order)
        
        removed = 0
        
        for x in order:
            if not in_N[x]:
                continue
            
            safe = True
            changes = []
            
            for y, bit_pos in impact[x]:
                if witnesses[y] & (1 << bit_pos):
                    new_w = witnesses[y] & ~(1 << bit_pos)
                    if new_w == 0:
                        safe = False
                        break
                    changes.append((y, bit_pos, new_w))
            
            if safe:
                in_N[x] = False
                removed += 1
                for y, bit_pos, new_w in changes:
                    witnesses[y] = new_w
            
        pts = []
        for i in range(num_points):
            if in_N[i]:
                xa, xb, ya, yb = unidx(i)
                pts.append([[xa, xb], [ya, yb]])
        
        return np.array(pts, dtype=np.int64) if pts else np.zeros((1, 2, 2), dtype=np.int64)

    start_time = time.time()
    best_score = -1
    best_result = None
    
    seed = 42
    while time.time() - start_time < 590:
        res = run_greedy(seed)
        score = 2401 - len(res)
        if score > best_score:
            best_score = score
            best_result = res
        seed += 1
        
    return best_result
