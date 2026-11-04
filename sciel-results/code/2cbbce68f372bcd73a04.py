import numpy as np
from scipy.optimize import differential_evolution
import time

N = 12

def get_hex_vertices(x, y, angle_deg, side=1.0):
    angles = np.deg2rad(angle_deg) + np.arange(6) * np.pi / 3
    return np.column_stack([x + side * np.cos(angles), y + side * np.sin(angles)])

def point_in_convex_polygon(p, poly):
    # Check if point p is inside convex polygon poly
    # Using cross products
    for i in range(len(poly)):
        p1 = poly[i]
        p2 = poly[(i + 1) % len(poly)]
        cross = np.cross(p2 - p1, p - p1)
        if cross < 0:
            return False
    return True

def segments_intersect(p1, p2, p3, p4):
    # Check if segment p1-p2 intersects p3-p4
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def hexagons_overlap(hex1, hex2):
    # Fast reject
    c1 = np.mean(hex1, axis=0)
    c2 = np.mean(hex2, axis=0)
    dist = np.linalg.norm(c1 - c2)
    if dist > 2.0:
        return False
    if dist < 1.732: # sqrt(3)
        return True
        
    # Check vertices
    for v in hex1:
        if point_in_convex_polygon(v, hex2):
            return True
    for v in hex2:
        if point_in_convex_polygon(v, hex1):
            return True
            
    # Check edges
    for i in range(6):
        p1 = hex1[i]
        p2 = hex1[(i + 1) % 6]
        for j in range(6):
            p3 = hex2[j]
            p4 = hex2[(j + 1) % 6]
            if segments_intersect(p1, p2, p3, p4):
                return True
    return False

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Parameters: x1, y1, a1, ..., x12, y12, a12, L
    # Total 37 parameters
    
    def objective(params):
        if time.time() > deadline:
            return float('inf')
            
        L = params[-1]
        if L < 3.0 or L > 10.0:
            return L + 100.0
            
        # Extract hexagon parameters
        hexes = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            verts = get_hex_vertices(x, y, a)
            hexes.append(verts)
            
        # Containment Penalty
        # Outer hexagon: Center (0,0), Side L, Angle 0
        # Normals for outer hexagon (flat top? or pointy top?)
        # Let's assume standard orientation: Vertex at (L, 0) is NOT standard for "flat top".
        # Standard "flat top" hexagon has horizontal top and bottom sides.
        # Vertices at 0, 60, 120, 180, 240, 300 degrees?
        # If vertex at 0 deg, it's pointy right.
        # Let's use the definition from get_hex_vertices:
        # angles = deg2rad(a) + k*pi/3.
        # If a=0, vertex at 0, 60, etc.
        # This is a "pointy" hexagon (vertex on x-axis).
        # The outer hexagon should match this orientation or be rotated.
        # The task says "outer hexagon ... rotation".
        # We can fix the outer hexagon rotation to 0 and allow inner ones to rotate.
        # Or allow outer rotation? The interface returns outer rotation.
        # But minimizing side length is independent of outer rotation if we can rotate inner ones.
        # So we can fix outer rotation to 0.
        
        # Normals for outer hexagon (pointy top, vertex at 0):
        # Sides are perpendicular to normals at 30, 90, 150, 210, 270, 330 degrees.
        normal_angles = np.deg2rad(30 + np.arange(6) * 60)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        apothem = L * np.sqrt(3) / 2
        
        containment_penalty = 0.0
        for verts in hexes:
            # Check all vertices of inner hex
            for v in verts:
                # Project v onto all normals
                dists = np.dot(v, normals.T)
                # Max distance must be <= apothem
                max_dist = np.max(dists)
                if max_dist > apothem:
                    containment_penalty += (max_dist - apothem) ** 2
        
        if containment_penalty > 0:
            return L + 10.0 * containment_penalty
            
        # Overlap Penalty
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                if hexagons_overlap(hexes[i], hexes[j]):
                    overlap_penalty += 10.0
                    
        if overlap_penalty > 0:
            return L + overlap_penalty
            
        return L

    # Bounds
    # x, y in [-5, 5]
    # angle in [0, 360]
    # L in [3.5, 6.0]
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))
        bounds.append((-5.0, 5.0))
        bounds.append((0.0, 360.0))
    bounds.append((3.5, 6.0))
    
    # Initial guess: Random or Heuristic
    # Let's use a heuristic start: 7 hexes in flower, 5 more random
    # But DE is global, so random is fine.
    
    # Run DE
    # maxiter=100, popsize=15
    # To respect deadline, we can check time in callback? DE doesn't support callback for early stop easily in all versions.
    # We'll just run it and hope it finishes. If not, we return best.
    # Actually, we can run multiple shorter DEs.
    
    best_result = None
    best_score = float('inf')
    
    # Run DE
    try:
        res = differential_evolution(objective, bounds, maxiter=50, popsize=10, seed=42, tol=1e-4)
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
    except Exception as e:
        pass
        
    # If time allows, run a local refinement on the best result
    if best_result is not None and time.time() < deadline - 10:
        # Use Nelder-Mead for polish
        from scipy.optimize import minimize
        try:
            res_local = minimize(objective, best_result, method='Nelder-Mead', options={'maxiter': 200})
            if res_local.fun < best_score:
                best_score = res_local.fun
                best_result = res_local.x
        except Exception:
            pass

    if best_result is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Extract result
    L = best_result[-1]
    inner = []
    for i in range(N):
        x = best_result[i * 3]
        y = best_result[i * 3 + 1]
        a = best_result[i * 3 + 2]
        inner.append([x, y, a])
        
    return inner, (0.0, 0.0), L, 0.0

# EVOLVE-BLOCK-START
# (The code above is the new design)
# EVOLVE-BLOCK-END
