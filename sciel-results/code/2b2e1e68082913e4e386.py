import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    def get_hex_vertices(cx, cy):
        return [(cx + np.cos(k * np.pi / 3), cy + np.sin(k * np.pi / 3)) for k in range(6)]
    
    def min_enclosing_hex_side(verts):
        best_L = float('inf')
        best_phi = 0
        best_c = (0.0, 0.0)
        for pd in range(0, 61, 3):
            phi = np.radians(pd)
            ns = [(np.cos(phi + k * np.pi / 3), np.sin(phi + k * np.pi / 3)) for k in range(6)]
            for tc in [(0.0, 0.0)]:
                mp = max((px - tc[0]) * nx + (py - tc[1]) * ny for (px, py) in verts for (nx, ny) in ns)
                L = mp * 2.0 / np.sqrt(3)
                if L < best_L:
                    best_L, best_phi, best_c = L, phi, tc
        return best_L, np.degrees(best_phi), best_c
    
    d1 = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    d2 = [(2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-1),(-2,0),(-2,1),(-2,2),(-1,2),(0,2),(1,1)]
    
    bl = float('inf')
    bc = None
    bo = None
    for om in range(12):
        sel = [d2[i] for i in range(12) if i != om][:5]
        pos = [(0,0)] + d1 + sel
        cs = [axial_to_cartesian(q, r) for q, r in pos]
        av = []
        for cx, cy in cs:
            av.extend(get_hex_vertices(cx, cy))
        L, pd, c = min_enclosing_hex_side(av)
        if L < bl:
            bl, bc, bo = L, cs, (pd, c)
        if time.time() > deadline:
            break
    
    if bc is None:
        pos = [(0,0)] + d1 + d2[:5]
        bc = [axial_to_cartesian(q, r) for q, r in pos]
        av = []
        for cx, cy in bc:
            av.extend(get_hex_vertices(cx, cy))
        bl, pd, c = min_enclosing_hex_side(av)
        bo = (pd, c)
    
    inner = [[x, y, 0.0] for x, y in bc]
    pd, c = bo
    return inner, c, bl, pd
