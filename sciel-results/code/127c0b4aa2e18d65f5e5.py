import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3
    
    def mul_f49(a, b, c, d):
        return ((a*c + b*d*k) % p, (a*d + b*c) % p)
    
    def add_f49(a, b, c, d):
        return ((a+c) % p, (b+d) % p)
        
    points = []
    point_to_idx = {}
    idx_to_point = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    idx = len(points)
                    pt = [[xa, xb], [ya, yb]]
                    points.append(pt)
                    point_to_idx[(xa, xb, ya, yb)] = idx
                    idx_to_point.append((xa, xb, ya, yb))
    
    M = len(points)
    
    lines = []
    line_to_points = []
    point_to_lines = [[] for _ in range(M)]
    
    for xa in range(p):
        for xb in range(p):
            c = (xa, xb)
            l_idx = len(lines)
            pts_on_line = []
            for ya in range(p):
                for yb in range(p):
                    pt_idx = point_to_idx[(xa, xb, ya, yb)]
                    pts_on_line.append(pt_idx)
                    point_to_lines[pt_idx].append(l_idx)
            lines.append((0, c))
            line_to_points.append(pts_on_line)
            
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ba in range(p):
                for bb in range(p):
                    b = (ba, bb)
                    l_idx = len(lines)
                    pts_on_line = []
                    for xa in range(p):
                        for xb in range(p):
                            ya, yb = mul_f49(ma, mb, xa, xb)
                            ya, yb = add_f49(ya, yb, ba, bb)
                            pt_idx = point_to_idx[(xa, xb, ya, yb)]
                            pts_on_line.append(pt_idx)
                            point_to_lines[pt_idx].append(l_idx)
                    lines.append((1, m, b))
                    line_to_points.append(pts_on_line)
                    
    L = len(lines)
    
    start_time = time.time()
    time_limit = 620.0 # Leave 10s margin
    best_C_size = 0
    best_C = set()
    
    while time.time() - start_time < time_limit:
        random.shuffle(idx_to_point)
        
        C = set()
        W = {}
        Count = [0] * L
        
        for x_idx in idx_to_point:
            D_x = set()
            for l_idx in point_to_lines[x_idx]:
                if Count[l_idx] == 0:
                    D_x.add(l_idx)
            
            if not D_x:
                continue
                
            lines_x_set = set(point_to_lines[x_idx])
            
            valid = True
            for y_idx in C:
                if W[y_idx].issubset(lines_x_set):
                    valid = False
                    break
                    
            if not valid:
                continue
                
            C.add(x_idx)
            W[x_idx] = D_x
            
            for l_idx in point_to_lines[x_idx]:
                Count[l_idx] += 1
                
            for y_idx in C:
                if y_idx == x_idx:
                    continue
                W[y_idx].difference_update(lines_x_set)
                    
        if len(C) > best_C_size:
            best_C_size = len(C)
            best_C = set(C)
            
    N_points = []
    for i in range(M):
        if i not in best_C:
            N_points.append(points[i])
            
    if not N_points:
        N_points = [points[0]]
        
    return np.array(N_points, dtype=np.int64)
