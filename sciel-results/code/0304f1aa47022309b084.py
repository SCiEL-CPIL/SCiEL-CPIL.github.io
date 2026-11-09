"""Packing 12 unit hexagons in a regular hexagon using scipy optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hexagon_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def _project_on_axis(vertices, axis):
    dots = vertices @ axis
    return np.min(dots), np.max(dots)

def _min_separation(v1, v2):
    """Return minimum separation between two hexagons (positive = no overlap)."""
    edges1 = np.roll(v1, -1, axis=0) - v1
    edges2 = np.roll(v2, -1, axis=0) - v2
    min_sep = np.inf
    for e in list(edges1) + list(edges2):
        axis = np.array([-e[1], e[0]])
        norm = np.linalg.norm(axis)
        if norm < 1e-10:
            continue
        axis = axis / norm
        min1, max1 = _project_on_axis(v1, axis)
        min2, max2 = _project_on_axis(v2, axis)
        sep = min(max1 - min2, max2 - min1)
        if sep < min_sep:
            min_sep = sep
    return min_sep

def _containment_margin(inner_verts, outer_verts):
    """Return minimum margin of inner hexagon inside outer hexagon."""
    min_margin = np.inf
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        center = np.mean(outer_verts, axis=0)
        if np.dot(normal, center - p1) < 0:
            normal = -normal
        for v in inner_verts:
            dist = np.dot(normal, v - p1)
            if dist < min_margin:
                min_margin = dist
    return min_margin

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial arrangement: 3x4 grid in hexagonal lattice
    s = np.sqrt(3)
    a1 = np.array([s, 0.0])
    a2 = np.array([s / 2, s * np.sqrt(3) / 2])
    
    positions = []
    for i in range(3):
        for j in range(4):
            pos = i * a1 + j * a2
            positions.append(pos)
    
    positions = np.array(positions)
    positions -= np.mean(positions, axis=0)
    
    cx, cy, L_init, phi_init = 0.0, 0.0, 5.0, 0.0
    x0 = np.concatenate([positions.flatten(), [cx, cy, L_init, phi_init]])
    
    def objective(vars):
        if time.time() > deadline:
            return 1e10
        return vars[26]
    
    def make_constraints():
        cons = []
        for i in range(N):
            for j in range(i + 1, N):
                def non_overlap(vars, i=i, j=j):
                    if time.time() > deadline:
                        return 1.0
                    pi = vars[2*i:2*i+2]
                    pj = vars[2*j:2*j+2]
                    vi = _hexagon_vertices(pi[0], pi[1], 1.0, 0.0)
                    vj = _hexagon_vertices(pj[0], pj[1], 1.0, 0.0)
                    return _min_separation(vi, vj) - 1e-6
                cons.append({'type': 'ineq', 'fun': non_overlap})
        
        for i in range(N):
            def containment(vars, i=i):
                if time.time() > deadline:
                    return 1.0
                pi = vars[2*i:2*i+2]
                cx, cy, L, phi = vars[24], vars[25], vars[26], vars[27]
                vi = _hexagon_vertices(pi[0], pi[1], 1.0, 0.0)
                vo = _hexagon_vertices(cx, cy, L, phi)
                return _containment_margin(vi, vo)
            cons.append({'type': 'ineq', 'fun': containment})
        
        cons.append({'type': 'ineq', 'fun': lambda v: v[26] - 2.0})
        return cons
    
    cons = make_constraints()
    
    best_L = np.inf
    best_result = None
    
    try:
        result = minimize(objective, x0, method='SLSQP', constraints=cons,
                         options={'maxiter': 300, 'ftol': 1e-7, 'disp': False})
        if time.time() < deadline and result.fun < best_L:
            best_L = result.fun
            best_result = result.x
    except Exception:
        pass
    
    if time.time() < deadline - 15:
        try:
            result2 = minimize(objective, x0, method='COBYLA', constraints=cons,
                              options={'maxiter': 150, 'rhobeg': 0.05})
            if time.time() < deadline and result2.fun < best_L:
                best_L = result2.fun
                best_result = result2.x
        except Exception:
            pass
    
    if best_result is not None:
        positions_final = best_result[:24].reshape(12, 2)
        cx, cy, L, phi = best_result[24], best_result[25], best_result[26], best_result[27]
    else:
        positions_final = positions
        cx, cy, L, phi = 0.0, 0.0, L_init, 0.0
    
    L = max(L, 3.0)
    
    inner = []
    for pos in positions_final:
        inner.append([float(pos[0]), float(pos[1]), 0.0])
    
    return inner, (float(cx), float(cy)), float(L), float(phi)
# EVOLVE-BLOCK-END
