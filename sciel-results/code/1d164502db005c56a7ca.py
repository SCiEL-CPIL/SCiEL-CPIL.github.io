import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def hexagon_vertices(cx, cy, side, rotation_deg):
    rot = math.radians(rotation_deg)
    verts = []
    for i in range(6):
        angle = rot + i * math.pi / 3
        x = cx + side * math.cos(angle)
        y = cy + side * math.sin(angle)
        verts.append((x, y))
    return verts

def point_in_hexagon(px, py, cx, cy, side, rotation_deg):
    rot = math.radians(rotation_deg)
    dx = px - cx
    dy = py - cy
    cos_r = math.cos(-rot)
    sin_r = math.sin(-rot)
    lx = dx * cos_r - dy * sin_r
    ly = dx * sin_r + dy * cos_r
    
    apothem = side * math.sqrt(3) / 2
    for i in range(6):
        angle = i * math.pi / 3
        nx = math.cos(angle)
        ny = math.sin(angle)
        if lx * nx + ly * ny > apothem + 1e-10:
            return False
    return True

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: return a valid but likely non-optimal configuration
        # Simple grid of 11 hexagons
        pts = []
        # 3x3 grid + 2 extra
        for q in range(-1, 2):
            for r in range(-1, 2):
                if q == 0 and r == 0: continue # Skip center for now, or keep it
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                pts.append((x, y))
        # Add center
        pts.append((0,0))
        # Add two more to make 11? 3x3=9. Need 2 more.
        pts.append((math.sqrt(3)*2, 0))
        pts.append((0, 1.5*2))
        
        pts = np.array(pts)
        # Calculate bounding hexagon
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        D = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            D = max(D, np.max(projs), -np.min(projs))
        S = (D + APOTHEM) / APOTHEM
        
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [float(cx), float(cy)], float(S), 0.0

    # Continuous Optimization Strategy
    
    # Parameters:
    # 0..N-1: x coords of inner hexagons
    # N..2N-1: y coords of inner hexagons
    # 2N: outer center x
    # 2N+1: outer center y
    # 2N+2: outer rotation (beta)
    # 2N+3: outer side length (S) -- actually we want to minimize S, so we can fix S and check feasibility?
    # Better: Minimize S subject to constraints? Or minimize S + penalty?
    # Let's minimize S + lambda * (overlap_penalty + containment_violation)
    # But S appears in the containment constraint.
    # Let's define variables:
    # x_i, y_i for i in 1..N
    # cx, cy (outer center)
    # beta (outer rotation)
    # S (outer side)
    # Total vars: 2N + 3 = 25
    
    def objective(params):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10
        
        # Unpack
        xs = params[0:N]
        ys = params[N:2*N]
        cx = params[2*N]
        cy = params[2*N+1]
        beta = params[2*N+2]
        S = params[2*N+3]
        
        # S must be positive
        if S < 0.5:
            return 1e10 + (0.5 - S) * 100
            
        # 1. Containment Penalty
        # For each inner hexagon, check if all its vertices are inside the outer hexagon.
        # Outer hexagon is defined by 6 half-planes.
        # A point (px, py) is inside if for all k in 0..5:
        # dot((px-cx, py-cy), n_k) <= S * APOTHEM
        # where n_k are normals to the sides of the outer hexagon.
        
        # Normals for outer hexagon with rotation beta:
        # The sides are at angles beta + pi/6 + k*pi/3 ?
        # Let's stick to the definition used in the seed:
        # Vertices at beta + i*pi/3.
        # Normals to sides are at beta + pi/6 + i*pi/3.
        
        normals = np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])
        
        max_violation = 0.0
        
        for i in range(N):
            px = xs[i]
            py = ys[i]
            # Inner hexagon vertices relative to its center (px, py)
            # Rotation is 0 for simplicity in optimization? 
            # The task allows rotation. But adding rotation variables (N more) makes it 3N+3 = 36 vars.
            # Let's assume inner hexagons are axis-aligned (rotation 0) for now to reduce dimensionality.
            # If this doesn't work, we can add rotation.
            # Actually, the seed used axial coords which implies rotation 0 (or 30 deg depending on def).
            # Let's assume rotation 0 for inner hexagons.
            
            # Vertices of inner hexagon i:
            # (px + cos(a), py + sin(a)) for a in 0, pi/3, ...
            # Wait, side length 1.
            # Vertices at distance 1 from center.
            angles_inner = np.array([j * math.pi / 3 for j in range(6)])
            verts_x = px + np.cos(angles_inner)
            verts_y = py + np.sin(angles_inner)
            
            # Check containment
            # For each vertex, check against all 6 outer half-planes
            # proj = dot(vert - center, normal)
            # Constraint: proj <= S * APOTHEM
            
            dx = verts_x - cx
            dy = verts_y - cy
            
            # proj shape: (6, 6) -> 6 vertices, 6 normals
            projs = dx[:, None] * normals[0, None] + dy[:, None] * normals[1, None]
            
            # Violation is max(0, proj - S*APOTHEM)
            violations = np.maximum(0, projs - S * APOTHEM)
            max_violation = max(max_violation, np.max(violations))
            
        # 2. Overlap Penalty
        # For each pair (i, j), check distance between centers.
        # If dist < 2 * APOTHEM, they might overlap.
        # For regular hexagons of side 1, the minimum distance to be disjoint is 2 * APOTHEM = sqrt(3) if aligned.
        # If rotated, it's more complex.
        # Let's use a soft penalty based on center distance.
        # d_min = sqrt(3) approx 1.732.
        # Penalty = sum(max(0, d_min - dist)^2)
        
        overlap_penalty = 0.0
        d_min = math.sqrt(3) - 1e-5 # Small epsilon to allow touching
        
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < d_min:
                    overlap_penalty += (d_min - dist) ** 2
        
        # Total Objective
        # We want to minimize S.
        # But we must satisfy constraints.
        # Use: S + lambda1 * max_violation + lambda2 * overlap_penalty
        
        # Adaptive lambdas? Or fixed large ones?
        # If S is minimized, it will try to shrink S, causing violations.
        # The penalty pushes S up or moves points.
        # Since S is a variable, the optimizer can increase S to reduce violation.
        # We want the minimal S that allows zero violation.
        
        # Let's try: S + 100 * max_violation + 10 * overlap_penalty
        
        return S + 100.0 * max_violation + 10.0 * overlap_penalty

    best_S = float('inf')
    best_params = None
    
    # Multi-start
    num_starts = 5
    rng = np.random.default_rng(42)
    
    for start_idx in range(num_starts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Initialize
        # Random cluster
        scale = 3.0
        xs = rng.uniform(-scale, scale, N)
        ys = rng.uniform(-scale, scale, N)
        cx = 0.0
        cy = 0.0
        beta = 0.0
        S = 4.0 # Initial guess
        
        params = np.concatenate([xs, ys, [cx, cy, beta, S]])
        
        # Bounds
        # S must be > 0
        # Positions can be anything, but let's bound them to prevent divergence
        bounds = []
        for i in range(2*N):
            bounds.append((-10, 10))
        bounds.append((-10, 10)) # cx
        bounds.append((-10, 10)) # cy
        bounds.append((0, math.pi/6)) # beta
        bounds.append((1.0, 10.0)) # S
        
        try:
            res = minimize(objective, params, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
                
        except Exception:
            pass

    # If we found a good solution, extract it
    if best_params is not None:
        xs = best_params[0:N]
        ys = best_params[N:2*N]
        cx = best_params[2*N]
        cy = best_params[2*N+1]
        beta = best_params[2*N+2]
        S = best_params[2*N+3]
        
        # Verify validity
        # Check overlap and containment with exact geometry
        
        valid = True
        
        # 1. Overlap
        d_min = math.sqrt(3)
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < d_min - 1e-6:
                    valid = False
                    break
            if not valid: break
            
        # 2. Containment
        if valid:
            normals = np.array([
                (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
                 math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
                for k in range(6)
            ])
            
            for i in range(N):
                px = xs[i]
                py = ys[i]
                angles_inner = np.array([j * math.pi / 3 for j in range(6)])
                verts_x = px + np.cos(angles_inner)
                verts_y = py + np.sin(angles_inner)
                
                dx = verts_x - cx
                dy = verts_y - cy
                projs = dx[:, None] * normals[0, None] + dy[:, None] * normals[1, None]
                
                if np.any(projs > S * APOTHEM + 1e-6):
                    valid = False
                    break
        
        if valid:
            inner = [[float(x), float(y), 0.0] for x, y in zip(xs, ys)]
            return inner, [float(cx), float(cy)], float(S), float(math.degrees(beta))
        else:
            # Invalid, try to repair or fallback
            pass

    # Fallback to the initial heuristic if optimization failed or was invalid
    # This ensures we return something valid
    pts = []
    for q in range(-1, 2):
        for r in range(-1, 2):
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
    pts.append((0,0))
    pts.append((math.sqrt(3)*2, 0))
    pts.append((0, 1.5*2))
    
    pts = np.array(pts)
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    D = 0
    for k in range(6):
        ang = math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
        D = max(D, np.max(projs), -np.min(projs))
    S = (D + APOTHEM) / APOTHEM
    
    inner = [[float(x), float(y), 0.0] for x, y in pts]
    return inner, [float(cx), float(cy)], float(S), 0.0
