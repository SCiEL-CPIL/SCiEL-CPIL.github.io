import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice basis
    # For unit hexagons packed edge-to-edge, center spacing = sqrt(3)
    # Basis: v1 = (sqrt(3), 0), v2 = (sqrt(3)/2, 3/2)
    
    # Compact arrangement of 11 hexagons in hexagonal coordinates
    # (col, row) -> x = sqrt(3)*(col + row/2), y = 1.5*row
    hex_coords = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    
    positions = np.array([[sqrt3 * (i + j/2.0), 1.5 * j] for (i, j) in hex_coords])
    
    def get_vertices(pos):
        verts = []
        for i in range(11):
            px, py = pos[i]
            for k in range(6):
                a = k * np.pi / 3
                verts.append([px + np.cos(a), py + np.sin(a)])
        return np.array(verts)
    
    def outer_size(pos, cx, cy, theta):
        verts = get_vertices(pos)
        mp = 0.0
        for k in range(6):
            a = theta + k * np.pi / 3
            proj = (verts[:,0]-cx)*np.cos(a) + (verts[:,1]-cy)*np.sin(a)
            mp = max(mp, np.max(proj))
        return 2.0*mp/sqrt3
    
    def optimal_outer(pos):
        verts = get_vertices(pos)
        cx0, cy0 = np.mean(verts[:,0]), np.mean(verts[:,1])
        
        def obj(p):
            return outer_size(pos, p[0], p[1], p[2])
        
        bv, bx = float('inf'), np.array([cx0, cy0, 0.0])
        for t0 in np.linspace(0, np.pi/3, 7):
            try:
                r = minimize(obj, [cx0, cy0, t0], method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-9})
                if r.fun < bv:
                    bv, bx = r.fun, r.x
            except:
                pass
        return bv, bx
    
    def overlap_penalty(pos):
        pen = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                d = np.linalg.norm(pos[i]-pos[j])
                if d < sqrt3:
                    pen += 100*(sqrt3-d)**2
        return pen
    
    def obj_inner(params):
        if time.time() > deadline:
            return float('inf')
        pos = params.reshape(11, 2)
        pen = overlap_penalty(pos)
        if pen > 0.5:
            return 1e6 + pen
        s, _ = optimal_outer(pos)
        return s + pen
    
    x0 = positions.flatten()
    best_s, best_p = float('inf'), x0.copy()
    
    try:
        r = minimize(obj_inner, x0, method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-7, 'fatol': 1e-7})
        if r.fun < best_s:
            best_s, best_p = r.fun, r.x
    except:
        pass
    
    # Random restarts
    if time.time() < deadline - 20:
        rng = np.random.RandomState(0)
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            xp = x0 + rng.normal(0, 0.3, x0.shape)
            try:
                r = minimize(obj_inner, xp, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7})
                if r.fun < best_s:
                    best_s, best_p = r.fun, r.x
            except:
                pass
    
    fp = best_p.reshape(11, 2)
    s, op = optimal_outer(fp)
    cx, cy, theta = op
    inner = [[fp[i,0], fp[i,1], 0.0] for i in range(11)]
    return inner, (cx, cy), s, np.degrees(theta) % 360.0
# EVOLVE-BLOCK-END
