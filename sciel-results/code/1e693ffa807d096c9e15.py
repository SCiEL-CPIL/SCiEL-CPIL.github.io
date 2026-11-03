import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def get_hex_vertices(cx, cy, angle):
        # Regular hexagon with side 1, center (cx, cy), rotated by angle
        # Vertices are at distance 1 from center
        verts = []
        for k in range(6):
            a = angle + k * math.pi / 3.0
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    def is_point_in_hexagon(px, py, S, phi):
        # Outer hexagon: center (0,0), side S, rotation phi
        # Vertices of outer hexagon
        # We check if point is inside by checking against the 6 half-planes defined by edges
        # Or simpler: transform point to hexagon local frame
        # Rotate point by -phi
        cos_p = math.cos(-phi)
        sin_p = math.sin(-phi)
        lx = px * cos_p - py * sin_p
        ly = px * sin_p + py * cos_p
        
        # In local frame, hexagon is flat-top or pointy-top?
        # Standard regular hexagon with side S:
        # Vertices at angles 0, 60, ... relative to center?
        # Let's define outer hexagon vertices at angles phi + k*pi/3.
        # The apothem is S * sqrt(3)/2.
        # The distance from center to any edge is S * sqrt(3)/2.
        # The edges are perpendicular to the directions phi + pi/6 + k*pi/3.
        
        apothem_outer = S * APOTHEM
        
        # Check distance to each of the 6 edges
        for k in range(6):
            # Normal to edge k
            ang = phi + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            # Distance from origin to line is apothem_outer
            # Point (lx, ly) must satisfy lx*nx + ly*ny <= apothem_outer
            # And lx*nx + ly*ny >= -apothem_outer
            proj = lx * nx + ly * ny
            if proj > apothem_outer + 1e-9:
                return False
            if proj < -apothem_outer - 1e-9:
                return False
        return True

    def calculate_overlap_penalty(params):
        # params: [x1, y1, th1, x2, y2, th2, ..., xN, yN, thN, S, phi]
        # Actually, let's fix phi=0 for simplicity and optimize S and inner params.
        # Due to symmetry, we can assume the outer hexagon is aligned.
        # If the optimal solution requires a rotated outer hexagon relative to the inner cluster,
        # it's equivalent to rotating the inner cluster. So fixing outer to 0 is WLOG.
        
        S = params[-1]
        if S < 1.0: S = 1.0 # Side length must be at least 1
        
        penalty = 0.0
        
        # 1. Containment penalty
        # If any vertex is outside, add penalty proportional to distance
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            th = params[3*i+2]
            verts = get_hex_vertices(x, y, th)
            for vx, vy in verts:
                if not is_point_in_hexagon(vx, vy, S, 0.0):
                    # Calculate how far outside
                    # Find max violation
                    max_viol = 0
                    for k in range(6):
                        ang = math.pi / 6.0 + k * math.pi / 3.0
                        nx = math.cos(ang)
                        ny = math.sin(ang)
                        proj = vx * nx + vy * ny
                        apothem = S * APOTHEM
                        if proj > apothem:
                            viol = proj - apothem
                            if viol > max_viol: max_viol = viol
                        if -proj > apothem:
                            viol = -proj - apothem
                            if viol > max_viol: max_viol = viol
                    penalty += max_viol * 100.0

        # 2. Non-overlap penalty
        # For each pair, check if they intersect.
        # Approximation: If distance between centers < 2*apothem, they likely overlap.
        # But with rotation, we need to be careful.
        # A better check: Use the Separating Axis Theorem (SAT).
        # For two convex polygons, they are disjoint if there is a separating axis.
        # Candidate axes: normals of edges of both polygons.
        
        hexes = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            th = params[3*i+2]
            hexes.append(get_hex_vertices(x, y, th))

        for i in range(N):
            for j in range(i+1, N):
                v1 = hexes[i]
                v2 = hexes[j]
                
                # Get axes from v1 and v2
                axes = []
                for k in range(6):
                    # Edge normal for v1
                    p1 = v1[k]
                    p2 = v1[(k+1)%6]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    # Normal
                    nx = -dy
                    ny = dx
                    norm = math.sqrt(nx*nx + ny*ny)
                    if norm > 1e-9:
                        axes.append((nx/norm, ny/norm))
                
                for k in range(6):
                    p1 = v2[k]
                    p2 = v2[(k+1)%6]
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    nx = -dy
                    ny = dx
                    norm = math.sqrt(nx*nx + ny*ny)
                    if norm > 1e-9:
                        axes.append((nx/norm, ny/norm))
                
                # Check for separation
                separated = False
                for ax, ay in axes:
                    # Project v1
                    proj1 = [p[0]*ax + p[1]*ay for p in v1]
                    min1 = min(proj1)
                    max1 = max(proj1)
                    # Project v2
                    proj2 = [p[0]*ax + p[1]*ay for p in v2]
                    min2 = min(proj2)
                    max2 = max(proj2)
                    
                    if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                        separated = True
                        break
                
                if not separated:
                    # They overlap. Estimate penalty.
                    # Use distance between centers as a proxy for depth of overlap
                    cx1 = np.mean(v1[:, 0])
                    cy1 = np.mean(v1[:, 1])
                    cx2 = np.mean(v2[:, 0])
                    cy2 = np.mean(v2[:, 1])
                    dist = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                    # Min dist for non-overlap is approx 2*apothem = sqrt(3) ~ 1.732
                    # If dist < 1.732, penalty
                    if dist < 2.0:
                        penalty += (2.0 - dist) * 10.0
                    else:
                        # Rare case: overlapping but centers far? Unlikely for convex hexagons
                        penalty += 10.0

        return S + penalty

    def objective(params):
        return calculate_overlap_penalty(params)

    # Initial guesses
    best_S = float('inf')
    best_params = None
    
    # Heuristic initialization: Central hexagon + Ring of 10
    # Center at (0,0)
    # Ring of 10 around it? No, a ring of 6 fits perfectly.
    # For 11, maybe 1 center, 6 ring, 4 extra?
    # Or 2 rows?
    
    # Let's try a few random initializations and a structured one.
    
    inits = []
    
    # Structured: 1 center, 6 around, 4 more
    c0 = [0, 0, 0]
    ring = []
    for k in range(6):
        a = k * math.pi / 3
        x = 2 * APOTHEM * math.cos(a) # Distance 1.732? No, distance between centers for touching is 2*APOTHEM = sqrt(3)
        # Wait, if they touch, distance is sqrt(3).
        x = math.sqrt(3) * math.cos(a)
        y = math.sqrt(3) * math.sin(a)
        ring.append([x, y, 0])
    
    # Add 4 more in the gaps? Or further out?
    # Let's just scatter them randomly near the center for the optimizer to sort out.
    
    for run in range(20):
        params = []
        if run == 0:
            # Structured
            params.extend([0, 0, 0])
            for k in range(6):
                a = k * math.pi / 3
                x = math.sqrt(3) * math.cos(a)
                y = math.sqrt(3) * math.sin(a)
                params.extend([x, y, 0])
            # 4 more: place them further out
            for k in range(4):
                a = k * math.pi / 2 + math.pi/4
                x = 3.0 * math.cos(a)
                y = 3.0 * math.sin(a)
                params.extend([x, y, 0])
            params.append(3.5) # Initial S
        else:
            # Random cluster
            for i in range(N):
                x = (np.random.rand() - 0.5) * 4.0
                y = (np.random.rand() - 0.5) * 4.0
                th = np.random.rand() * math.pi / 3
                params.extend([x, y, th])
            params.append(4.0) # Initial S
        
        inits.append(params)
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

    for init in inits:
        try:
            res = minimize(objective, init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Extract results
    S = best_params[-1]
    # Ensure S is not too small
    if S < 1.0: S = 1.0
    
    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        th = float(best_params[3*i+2])
        # Normalize angle to [0, 180) or similar
        th_deg = math.degrees(th) % 180.0
        inner.append([x, y, th_deg])
        
    center = [0.0, 0.0]
    side = float(S)
    rotation = 0.0 # We fixed outer rotation to 0
    
    return inner, center, side, rotation
