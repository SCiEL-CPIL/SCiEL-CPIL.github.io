"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from itertools import combinations

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = math.sqrt(3.0)
    
    def hex_verts(cx, cy, theta, s=1.0):
        a = np.array([math.radians(60*k) + theta for k in range(6)])
        return np.column_stack([cx + s*np.cos(a), cy + s*np.sin(a)])
    
    def req_L(cx, cy, theta, ot=0.0):
        v = hex_verts(cx, cy, theta)
        na = np.array([math.radians(30 + 60*j) + ot for j in range(6)])
        projs = v[:,0]*np.cos(na)[np.newaxis,:] + v[:,1]*np.sin(na)[np.newaxis,:]
        return np.max(projs) / (sqrt3/2.0)
    
    # 19 grid positions
    grid = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q+r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q/2.0)
                grid.append((x, y, q, r))
    
    def hex_dist(q1, r1, q2, r2):
        dq, dr = q1-q2, r1-r2
        return max(abs(dq), abs(dr), abs(dq+dr))
    
    rings = {0: [], 1: [], 2: []}
    for i, (x, y, q, r) in enumerate(grid):
        d = hex_dist(q, r, 0, 0)
        if d in rings:
            rings[d].append(i)
    
    center_idx = rings[0][0]
    ring1 = rings[1]
    ring2 = rings[2]
    
    # Try all C(12,5)=792 subsets of ring2
    best_combo = None
    best_L = float('inf')
    
    for combo in combinations(range(len(ring2)), 5):
        subset = [center_idx] + ring1 + [ring2[i] for i in combo]
        max_L = 0.0
        for idx in subset:
            x, y, _, _ = grid[idx]
            L = req_L(x, y, 0.0, 0.0)
            max_L = max(max_L, L)
        if max_L < best_L:
            best_L = max_L
            best_combo = combo
    
    best_subset = [center_idx] + ring1 + [ring2[i] for i in best_combo]
    
    # Optimize outer rotation
    def total_L(subset, thetas, ot):
        m = 0.0
        for i, idx in enumerate(subset):
            x, y, _, _ = grid[idx]
            m = max(m, req_L(x, y, thetas[i], ot))
        return m
    
    thetas = np.zeros(N)
    best_ot = 0.0
    best_Lv = total_L(best_subset, thetas, 0.0)
    for ot_d in np.arange(0, 60, 0.5):
        ot = math.radians(ot_d)
        L = total_L(best_subset, thetas, ot)
        if L < best_Lv:
            best_Lv = L
            best_ot = ot
    
    for st in [0.1, 0.01, 0.001]:
        for _ in range(15):
            f = False
            for dd in np.arange(-st*5, st*5+st, st):
                ot = best_ot + math.radians(dd)
                L = total_L(best_subset, thetas, ot)
                if L < best_Lv - 1e-13:
                    best_Lv = L
                    best_ot = ot
                    f = True
            if not f: break
    
    # Optimize individual thetas
    for _ in range(3):
        changed = False
        for i in range(N):
            idx = best_subset[i]
            x, y, _, _ = grid[idx]
            bt = thetas[i]
            bL = req_L(x, y, bt, best_ot)
            for td in np.arange(0, 60, 1.0):
                t = math.radians(td)
                L = req_L(x, y, t, best_ot)
                if L < bL - 1e-13:
                    bL = L; bt = t
            for st in [0.1, 0.01]:
                for dd in np.arange(-st*5, st*5+st, st):
                    t = bt + math.radians(dd)
                    L = req_L(x, y, t, best_ot)
                    if L < bL - 1e-13:
                        bL = L; bt = t
            if bL < req_L(x, y, thetas[i], best_ot) - 1e-13:
                thetas[i] = bt
                changed = True
        best_Lv = total_L(best_subset, thetas, best_ot)
        if not changed: break
    
    inner = []
    for i, idx in enumerate(best_subset):
        x, y, _, _ = grid[idx]
        inner.append([x, y, math.degrees(thetas[i])])
    
    return inner, (0.0, 0.0), best_Lv, best_ot
