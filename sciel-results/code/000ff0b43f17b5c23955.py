import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600
    sqrt3 = np.sqrt(3)
    
    def get_vertices(cx, cy, s, angle_deg):
        angles = np.radians(angle_deg + 60 * np.arange(6))
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
    
    def are_separated(pos1, pos2):
        v1 = get_vertices(pos1[0], pos1[1], 1.0, pos1[2])
        v2 = get_vertices(pos2[0], pos2[1], 1.0, pos2[2])
        
        normals = []
        for verts in [v1, v2]:
            for i in range(6):
                p1 = verts[i]
                p2 = verts[(i+1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    normals.append(normal / norm)
        
        for n in normals:
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            if np.max(proj1) < np.min(proj2) - 1e-6 or np.max(proj2) < np.min(proj1) - 1e-6:
                return True
        return False
    
    def is_valid(positions):
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                if not are_separated(positions[i], positions[j]):
                    return False
        return True
    
    def min_outer_side(positions):
        all_verts = np.vstack([get_vertices(p[0], p[1], 1.0, p[2]) for p in positions])
        normal_angles = np.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        def obj(params):
            cx, cy = params
            center = np.array([cx, cy])
            max_projs = np.array([np.max(np.dot(all_verts, n)) for n in normals])
            center_projs = np.array([np.dot(center, n) for n in normals])
            return np.max(max_projs - center_projs) / (sqrt3/2)
        
        cx0, cy0 = np.mean(all_verts, axis=0)
        res = minimize(obj, [cx0, cy0], method='Nelder-Mead', options={'maxiter':1000,'xatol':1e-10,'fatol':1e-10})
        return obj(res.x)
    
    # Build compact config using lattice
    v1 = np.array([1.5, sqrt3/2])
    v2 = np.array([0, sqrt3])
    
    # Ring 0: center
    centers = [np.array([0.0, 0.0])]
    
    # Ring 1: 6 neighbors
    ring1 = [
        v1, v2, v2-v1, -v1, -v2, v1-v2
    ]
    for c in ring1:
        centers.append(c.copy())
    
    # Ring 2: add 4 more for total 11
    # Pick symmetric positions
    ring2_candidates = [
        2*v1, 2*v2, 2*(v2-v1), -2*v1, -2*v2, 2*(v1-v2),
        v1+v2, v2+(v2-v1), (v2-v1)+(-v1), -v1+(-v2), -v2+(v1-v2), (v1-v2)+v1
    ]
    # Simplify: just pick 4 symmetric ones
    # For 4-fold symmetry, pick: 2*v1, 2*v2, 2*(v2-v1), 2*(v1-v2) ... but that's not 4-fold
    # Let's pick: v1+v2, v2+(v2-v1), -v1+(-v2), -v2+(v1-v2) -> these are 4 of the 12
    
    # Actually, let me just pick 4 that give a nice shape
    # Pick: 2*v1, 2*v2, -2*v1, -2*v2 (opposite pairs)
    for c in [2*v1, 2*v2, -2*v1, -2*v2]:
        centers.append(c.copy())
    
    positions = np.array([[c[0], c[1], 0.0] for c in centers])
    
    # Verify
    if not is_valid(positions):
        # Fallback: spread out more
        positions = np.array([[0,0,0]])
        for i in range(1, 11):
            angle = 2*np.pi*(i-1)/10
            r = 2*sqrt3
            positions = np.vstack([positions, [r*np.cos(angle), r*np.sin(angle), 0]])
    
    # Optimize
    def cost(params):
        if time.time() - start_time > deadline:
            return 1e10
        pos = params.reshape(11, 3)
        for i in range(11):
            for j in range(i+1, 11):
                if not are_separated(pos[i], pos[j]):
                    return 1e10
        return min_outer_side(pos)
    
    best = positions.flatten().copy()
    best_c = cost(best)
    
    if time.time() - start_time < deadline - 30:
        res = minimize(cost, best, method='Nelder-Mead', options={'maxiter':2000,'xatol':1e-8,'fatol':1e-8})
        if res.fun < best_c:
            best = res.x
            best_c = res.fun
    
    final_pos = best.reshape(11, 3)
    s = min_outer_side(final_pos)
    all_verts = np.vstack([get_vertices(p[0], p[1], 1.0, p[2]) for p in final_pos])
    cx, cy = np.mean(all_verts, axis=0)
    
    return final_pos, np.array([cx, cy, 0.0]), float(s)
