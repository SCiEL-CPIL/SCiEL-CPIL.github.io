import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    apothem = np.sqrt(3) / 2.0

    def hex_vertices(cx, cy, angle, s=1.0):
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def compute_S_vec(cx, cy, angles, outer_angle=0.0):
        """Vectorized computation of minimum outer side length."""
        # cx, cy, angles are arrays of length N
        v_angles = angles[:, np.newaxis] + np.arange(6) * np.pi / 3
        vx = cx[:, np.newaxis] + np.cos(v_angles)
        vy = cy[:, np.newaxis] + np.sin(v_angles)
        
        ca = np.cos(outer_angle)
        sa = np.sin(outer_angle)
        rx = vx * ca + vy * sa
        ry = -vx * sa + vy * ca
        
        thetas = np.pi/6 + np.arange(6) * np.pi/3
        ct = np.cos(thetas)
        st = np.sin(thetas)
        
        # proj: (N, 6, 6)
        projs = rx[:, :, np.newaxis] * ct[np.newaxis, np.newaxis, :] + ry[:, :, np.newaxis] * st[np.newaxis, np.newaxis, :]
        
        max_proj = np.max(np.abs(projs))
        return max_proj / apothem

    def sat_overlap_depth(v1, v2):
        """Return overlap depth (0 if no overlap) between two convex polygons using SAT."""
        polys = [v1, v2]
        max_depth = 0.0
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                edge_len = np.linalg.norm(edge)
                if edge_len < 1e-12:
                    continue
                normal = np.array([-edge[1], edge[0]]) / edge_len
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
                if overlap < 0:
                    return 0.0
                max_depth = max(max_depth, overlap)
        return max_depth

    def objective(params):
        """
        params: [cx1, cy1, a1, ..., cxN, cyN, aN, outer_angle]
        """
        if time.time() > deadline:
            return 1e10
        
        n = N
        cx = params[0:3*n:3]
        cy = params[1:3*n:3]
        angles = params[2:3*n:3]
        outer_angle = params[-1]
        
        S = compute_S_vec(cx, cy, angles, outer_angle)
        
        # Overlap penalty
        penalty = 0.0
        # Precompute vertices for overlap check
        verts_list = []
        for i in range(n):
            verts_list.append(hex_vertices(cx[i], cy[i], angles[i]))
            
        for i in range(n):
            c1 = verts_list[i].mean(axis=0)
            for j in range(i + 1, n):
                c2 = verts_list[j].mean(axis=0)
                dist = np.linalg.norm(c1 - c2)
                if dist < 2.0:
                    depth = sat_overlap_depth(verts_list[i], verts_list[j])
                    if depth > 0:
                        penalty += 1000.0 * depth
        return S + penalty

    def make_initial_lattice():
        """Create initial hexagonal lattice placement."""
        params = np.zeros(3 * N + 1)
        # Center hexagon
        params[0:3] = [0, 0, 0]
        idx = 1
        # Ring 1: 6 hexagons
        for k in range(6):
            theta = k * np.pi / 3
            cx = np.sqrt(3) * np.cos(theta)
            cy = np.sqrt(3) * np.sin(theta)
            params[3 * idx] = cx
            params[3 * idx + 1] = cy
            params[3 * idx + 2] = 0
            idx += 1
        # Ring 2: 4 more hexagons
        # Place in gaps: 30, 90, 150, 210 degrees
        for k in range(4):
            theta = (k + 0.5) * np.pi / 3
            cx = 2 * np.sqrt(3) * np.cos(theta)
            cy = 2 * np.sqrt(3) * np.sin(theta)
            params[3 * idx] = cx
            params[3 * idx + 1] = cy
            params[3 * idx + 2] = 0
            idx += 1
        params[-1] = 0.0 # outer angle
        return params

    def optimize_from(params, maxiter=2000, maxfev=10000):
        if time.time() > deadline:
            return None, np.inf
        try:
            result = minimize(objective, params, method='Powell',
                              options={'maxiter': maxiter, 'maxfev': maxfev, 'xatol': 1e-9, 'fatol': 1e-9})
            return result.x, result.fun
        except Exception:
            return None, np.inf

    best_params = None
    best_score = np.inf

    # Start 1: Standard lattice
    p0 = make_initial_lattice()
    res_params, res_score = optimize_from(p0)
    if res_params is not None and res_score < best_score:
        best_params = res_params
        best_score = res_score

    # Start 2: Rotated outer hexagon
    if time.time() < deadline - 20:
        p1 = p0.copy()
        p1[-1] = np.pi / 12 # 15 degrees
        res_params, res_score = optimize_from(p1)
        if res_params is not None and res_score < best_score:
            best_params = res_params
            best_score = res_score

    # Start 3: Perturbed best
    if best_params is not None and time.time() < deadline - 15:
        p2 = best_params + np.random.normal(0, 0.05, size=best_params.size)
        res_params, res_score = optimize_from(p2)
        if res_params is not None and res_score < best_score:
            best_params = res_params
            best_score = res_score

    # Start 4: Different ring 2 placement (corners instead of gaps?)
    if time.time() < deadline - 15:
        p3 = make_initial_lattice()
        # Try placing ring 2 at 0, 60, 120, 180? No, that overlaps ring 1.
        # Try shifting ring 2 slightly inward?
        # Or just randomize angles of ring 2 hexagons
        for i in range(7, N):
            p3[3 * i + 2] = np.random.uniform(-np.pi/6, np.pi/6)
        res_params, res_score = optimize_from(p3)
        if res_params is not None and res_score < best_score:
            best_params = res_params
            best_score = res_score

    # Final refinement
    if best_params is not None and time.time() < deadline - 10:
        for trial in range(5):
            if time.time() > deadline - 5:
                break
            p = best_params + np.random.normal(0, 0.02, size=best_params.size)
            res_params, res_score = optimize_from(p, maxiter=1000, maxfev=5000)
            if res_params is not None and res_score < best_score:
                best_params = res_params
                best_score = res_score

    if best_params is None:
        best_params = make_initial_lattice()

    # Extract results
    cx = best_params[0:3*N:3]
    cy = best_params[1:3*N:3]
    angles = best_params[2:3*N:3]
    outer_angle = best_params[-1]
    
    inner = []
    for i in range(N):
        inner.append([cx[i], cy[i], np.degrees(angles[i])])

    # Compute final S accurately
    S = compute_S_vec(cx, cy, angles, outer_angle)
    
    # The evaluator expects: inner, center, side, angle
    # center is (0,0)
    # side is S
    # angle is outer_angle (in degrees? or radians? Contract says "outer hexagon center (x, y), side length and rotation". 
    # Usually rotation is in degrees if inner angles are in degrees. 
    # Let's check the current program: `return inner, (0.0, 0.0), S, 0.0`. 
    # The current program returns 0.0 for angle. 
    # The inspiration says "angle_degrees" for inner. 
    # It's safer to return outer angle in degrees as well, or radians? 
    # "inner rows (x, y, angle_degrees)". 
    # "outer hexagon center (x, y), side length and rotation". 
    # It doesn't specify units for outer rotation. 
    # However, standard is often radians in math, but degrees in "angle_degrees". 
    # Let's look at the current program: it returns 0.0. 
    # If I return np.degrees(outer_angle), it might be consistent. 
    # But if the evaluator expects radians, 0.0 is safe. 
    # If I optimize outer angle, I should return it. 
    # Let's assume radians for outer angle to be safe with standard scientific libraries, 
    # UNLESS the inner angles are clearly degrees. 
    # The prompt says "inner rows (x, y, angle_degrees)". 
    # It does NOT say "outer rotation_degrees". 
    # So I will return outer_angle in radians.
    
    return inner, (0.0, 0.0), S, outer_angle
# EVOLVE-BLOCK-END
