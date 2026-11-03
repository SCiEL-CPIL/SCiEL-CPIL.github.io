import numpy as np
from scipy.optimize import minimize

N = 12
sqrt3 = np.sqrt(3)

def hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    return np.array([[cx + side*np.cos(angle_rad + k*np.pi/3),
                      cy + side*np.sin(angle_rad + k*np.pi/3)] for k in range(6)])

def enclosing_hex_side(verts, cx, cy, theta_deg):
    theta_rad = np.radians(theta_deg)
    max_apothem = 0
    for k in range(6):
        a = theta_rad + k * np.pi / 3
        nx, ny = np.cos(a), np.sin(a)
        projections = (verts[:, 0] - cx) * nx + (verts[:, 1] - cy) * ny
        max_apothem = max(max_apothem, np.max(projections))
    return max_apothem * 2 / sqrt3

def construct():
    best_side = 1e10
    best_result = None
    
    arrangements = []
    
    # 3x4 honeycomb
    for j in range(4):
        for i in range(3):
            arrangements[-1] if arrangements else None
    # Let me build them properly
    arrangements = []
    
    # A: 3 wide, 4 tall
    a = [[i*sqrt3 + (j%2)*sqrt3/2, j*1.5, 0.0] for j in range(4) for i in range(3)]
    arrangements.append(a)
    
    # B: 4 wide, 3 tall
    b = [[i*sqrt3 + (j%2)*sqrt3/2, j*1.5, 0.0] for j in range(3) for i in range(4)]
    arrangements.append(b)
    
    # C: 3x4 rotated 30
    c = [[i*sqrt3 + (j%2)*sqrt3/2, j*1.5, 30.0] for j in range(4) for i in range(3)]
    arrangements.append(c)
    
    # D: 2 wide, 6 tall
    d = [[i*sqrt3 + (j%2)*sqrt3/2, j*1.5, 0.0] for j in range(6) for i in range(2)]
    arrangements.append(d)
    
    # E: 6 wide, 2 tall
    e = [[i*sqrt3 + (j%2)*sqrt3/2, j*1.5, 0.0] for j in range(2) for i in range(6)]
    arrangements.append(e)
    
    for inner in arrangements:
        xs = [p[0] for p in inner]
        ys = [p[1] for p in inner]
        cx0 = (min(xs) + max(xs)) / 2
        cy0 = (min(ys) + max(ys)) / 2
        inner_c = [[x-cx0, y-cy0, ang] for x, y, ang in inner]
        verts = np.vstack([hex_vertices(x, y, 1.0, a) for x, y, a in inner_c])
        
        def obj(p):
            return enclosing_hex_side(verts, p[0], p[1], p[2])
        
        best = 1e10
        best_p = None
        for t0 in [0, 15, 30, 45]:
            r = minimize(obj, [0, 0, t0], method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7})
            if r.fun < best:
                best = r.fun
                best_p = r.x
        
        if best < best_side:
            best_side = best
            best_result = (inner_c, (best_p[0], best_p[1]), best, best_p[2])
    
    return best_result

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
