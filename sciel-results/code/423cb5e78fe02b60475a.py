import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 120s kill
    
    # Helper: check if point is inside outer hexagon
    # Outer hexagon centered at (cx, cy) with side s, rotation angle_deg
    # Vertices of regular hexagon with side s, center (cx,cy), angle a:
    # (cx + s*cos(a + k*60°), cy + s*sin(a + k*60°)) for k=0..5
    # A point (px,py) is inside iff for all 6 edges, the point is on the correct side
    
    def point_in_hex(px, py, cx, cy, s, angle_deg):
        a = np.radians(angle_deg)
        # The 6 outward normals of the hexagon (perpendicular to edges, pointing outward)
        # For a hexagon with vertices at angles a, a+60, ..., a+300,
        # the edge normals point at angles a+30, a+90, ... (midpoints of arcs)
        # Actually, the supporting lines: for each vertex direction, the constraint is
        # (p - c) · n_k <= s * cos(30°) = s * √3/2 where n_k are unit normals
        # The 6 normals point at angles a + 30°, a + 90°, ..., a + 330°
        normals = np.array([np.cos(a + np.radians(30 + 60*k)), np.sin(a + np.radians(30 + 60*k))] for k in range(6))
        rel = np.array([px - cx, py - cy])
        dists = normals @ rel
        return np.all(dists <= s * np.sqrt(3) / 2 + 1e-9)
    
    # Check if two hexagons overlap
    # Two regular hexagons with centers (x1,y1), (x2,y2), angles a1, a2, both side 1
    # They don't overlap iff the distance between centers is >= the minimum distance
    # for which two unit hexagons at those orientations just touch.
    # This is complex. Approximate: the inradius is √3/2, so if we use circles of radius √3/2,
    # non-overlap of circles implies non-overlap of hexagons (conservative).
    # Better: compute the actual minimum distance. For two hexagons with same orientation,
    # min center distance for non-overlap is √3 (touching side to side).
    # For different orientations, it's more complex.
    
    # For a robust check, let's use the fact that two convex polygons don't overlap
    # iff there exists a separating axis. For two hexagons, we check all 12 edge normals.
    
    def hex_vertices(cx, cy, side, angle_deg):
        a = np.radians(angle_deg)
        verts = []
        for k in range(6):
            ang = a + np.radians(60 * k)
            verts.append([cx + side * np.cos(ang), cy + side * np.sin(ang)])
        return np.array(verts)
    
    def polygons_disjoint(v1, v2):
        # v1, v2 are (6,2) arrays of vertices
        # Check separating axis theorem
        edges = []
        for v in (v1, v2):
            for i in range(6):
                e = v[(i+1)%6] - v[i]
                # Normal to edge
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm > 1e-12:
                    n = n / norm
                    edges.append(n)
        for n in edges:
            proj1 = v1 @ n
            proj2 = v2 @ n
            if proj1.max() < proj2.min() - 1e-9 or proj2.max() < proj1.min() - 1e-9:
                return True
        return False
    
    def check_valid(inner, center, side, angle_deg):
        cx, cy = center
        # Check all inner hexagons are inside outer
        for (ix, iy, iang) in inner:
            # Check all 6 vertices of inner hex are inside outer
            verts = hex_vertices(ix, iy, 1.0, iang)
            for (vx, vy) in verts:
                if not point_in_hex(vx, vy, cx, cy, side, angle_deg):
                    return False
        # Check no two inner hexagons overlap
        for i in range(N):
            for j in range(i+1, N):
                vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                if not polygons_disjoint(vi, vj):
                    return False
        return True
    
    # Initial guess: honeycomb-like arrangement
    # Rows of hexagons. For same-orientation hexagons in a row, spacing is √3
    # Between rows, vertical spacing is 1.5 (for pointy-top) or other depending on orientation
    
    # Let's use a structured initial guess and then optimize
    # Start with all hexagons at angle 0, arranged in a grid pattern
    
    # Better initial: place in a compact cluster
    # 12 hexagons: try 3-4-3-2 rows or similar
    
    # For flat-top hexagons (angle 0 means vertex at right), 
    # horizontal spacing between centers in same row: 2 * inradius = √3
    # Vertical spacing between rows: 1.5 (for the staggered pattern)
    # Actually for flat-top: width = 2, height = √3
    # Center-to-center horizontal: 1.5 * width/2... let me just use √3 for same-orientation touching
    
    # Let me use a simple initial placement and optimize
    sqrt3 = np.sqrt(3)
    
    # Initial: 4 rows of 3
    init_positions = []
    for row in range(4):
        for col in range(3):
            x = col * sqrt3
            y = row * 1.5
            init_positions.append([x, y, 0.0])
    
    # Center the arrangement
    xs = [p[0] for p in init_positions]
    ys = [p[1] for p in init_positions]
    cx0 = (min(xs) + max(xs)) / 2
    cy0 = (min(ys) + max(ys)) / 2
    for p in init_positions:
        p[0] -= cx0
        p[1] -= cy0
    
    # Initial outer hexagon
    # Estimate needed side length
    max_r = max(np.sqrt(p[0]**2 + p[1]**2) for p in init_positions) + 1
    side0 = max_r / (sqrt3/2) + 1
    
    best_result = (init_positions, (0.0, 0.0), side0, 0.0)
    best_score = side0
    
    # Objective function for optimization
    def objective(params):
        # params: x1,y1,a1, x2,y2,a2, ..., cx, cy, side
        # We'll optimize positions, angles, and outer hexagon
        inner = []
        for i in range(N):
            inner.append([params[3*i], params[3*i+1], params[3*i+2]])
        cx = params[3*N]
        cy = params[3*N+1]
        side = params[3*N+2]
        angle = params[3*N+3]
        
        # Penalty for violations
        penalty = 0.0
        
        # Containment: each vertex of each inner hex must be inside outer
        for i in range(N):
            verts = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            for (vx, vy) in verts:
                a = np.radians(angle)
                normals = np.array([np.cos(a + np.radians(30 + 60*k)), np.sin(a + np.radians(30 + 60*k))] for k in range(6))
                rel = np.array([vx - cx, vy - cy])
                dists = normals @ rel
                violations = np.maximum(0, dists - side * sqrt3 / 2)
                penalty += np.sum(violations**2)
        
        # Non-overlap: for each pair, compute minimum penetration
        for i in range(N):
            vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            for j in range(i+1, N):
                vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                # Compute max overlap using SAT
                max_pen = 0.0
                edges = []
                for v in (vi, vj):
                    for k in range(6):
                        e = v[(k+1)%6] - v[k]
                        n = np.array([-e[1], e[0]])
                        norm = np.linalg.norm(n)
                        if norm > 1e-12:
                            n = n / norm
                            edges.append(n)
                for n in edges:
                    proj1 = vi @ n
                    proj2 = vj @ n
                    overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
                    if overlap > 0:
                        max_pen = max(max_pen, overlap)
                penalty += max_pen**2
        
        return side + penalty
    
    # Try optimization with SLSQP
    x0 = []
    for p in init_positions:
        x0.extend([p[0], p[1], p[2]])
    x0.extend([0.0, 0.0, side0, 0.0])
    
    try:
        result = minimize(objective, x0, method='SLSQP', 
                          options={'maxiter': 500, 'ftol': 1e-6})
        if result.success or result.fun < best_score:
            inner_opt = []
            for i in range(N):
                inner_opt.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
            cx_opt = result.x[3*N]
            cy_opt = result.x[3*N+1]
            side_opt = result.x[3*N+2]
            angle_opt = result.x[3*N+3]
            
            if check_valid(inner_opt, (cx_opt, cy_opt), side_opt, angle_opt):
                if side_opt < best_score:
                    best_score = side_opt
                    best_result = (inner_opt, (cx_opt, cy_opt), side_opt, angle_opt)
    except Exception:
        pass
    
    # If still at initial, try a few random restarts
    if best_score > 8:  # still poor
        np.random.seed(42)
        for restart in range(5):
            if time.time() > deadline:
                break
            # Random perturbation of initial
            init_r = []
            for p in init_positions:
                init_r.append([p[0] + np.random.randn()*0.5, 
                              p[1] + np.random.randn()*0.5, 
                              np.random.randn()*0.1])
            side_r = best_score
            x0_r = []
            for p in init_r:
                x0_r.extend([p[0], p[1], p[2]])
            x0_r.extend([0.0, 0.0, side_r, 0.0])
            
            try:
                result_r = minimize(objective, x0_r, method='SLSQP',
                                   options={'maxiter': 200, 'ftol': 1e-5})
                if result_r.fun < best_score:
                    inner_r = []
                    for i in range(N):
                        inner_r.append([result_r.x[3*i], result_r.x[3*i+1], result_r.x[3*i+2]])
                    cx_r = result_r.x[3*N]
                    cy_r = result_r.x[3*N+1]
                    side_r = result_r.x[3*N+2]
                    angle_r = result_r.x[3*N+3]
                    if check_valid(inner_r, (cx_r, cy_r), side_r, angle_r):
                        best_score = side_r
                        best_result = (inner_r, (cx_r, cy_r), side_r, angle_r)
            except Exception:
                pass
    
    return best_result
# EVOLVE-BLOCK-END
