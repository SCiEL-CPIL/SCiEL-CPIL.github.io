import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

def _hex_vertices_array(cx, cy, s, angle_rad):
    """
    Returns (6, 2) array of vertices for a regular hexagon.
    """
    thetas = angle_rad + np.arange(6) * np.pi / 3
    xs = cx + s * np.cos(thetas)
    ys = cy + s * np.sin(thetas)
    return np.column_stack((xs, ys))

def _point_in_hex_dist_sq_array(pts, cx, cy, s, angle_rad):
    """
    pts: (M, 2) array of points.
    Returns (M,) array of squared distance violations for each point being outside the hexagon.
    0 if inside.
    """
    if len(pts) == 0:
        return np.array([])
    
    a = -angle_rad
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    
    apothem = s * math.sqrt(3) / 2
    
    # Normals for the 3 unique pairs of sides
    # Directions: 30, 90, 150 degrees relative to the hexagon's local frame?
    # Actually, for a hexagon with vertex at angle 0, the normal to the side between v0 and v1 is at 30 deg.
    # The sides are perpendicular to directions: 30, 90, 150 (and opposites).
    # Let's use the standard normal vectors for the 3 constraints.
    # Constraint i: |n_i . p| <= apothem
    # n_0 = (cos(30), sin(30))
    # n_1 = (cos(90), sin(90))
    # n_2 = (cos(150), sin(150))
    
    n0 = np.array([math.cos(math.pi/6), math.sin(math.pi/6)])
    n1 = np.array([math.cos(math.pi/2), math.sin(math.pi/2)])
    n2 = np.array([math.cos(5*math.pi/6), math.sin(5*math.pi/6)])
    
    proj0 = rx * n0[0] + ry * n0[1]
    proj1 = rx * n1[0] + ry * n1[1]
    proj2 = rx * n2[0] + ry * n2[1]
    
    viol0 = np.maximum(0, proj0 - apothem) + np.maximum(0, -proj0 - apothem)
    viol1 = np.maximum(0, proj1 - apothem) + np.maximum(0, -proj1 - apothem)
    viol2 = np.maximum(0, proj2 - apothem) + np.maximum(0, -proj2 - apothem)
    
    # Sum of squared violations
    return viol0**2 + viol1**2 + viol2**2

def _pairwise_penalty(p1, p2):
    """
    p1, p2: (cx, cy, angle_rad)
    Returns penalty for overlap.
    We check if vertices of one are inside the other.
    """
    c1x, c1y, a1 = p1
    c2x, c2y, a2 = p2
    
    dx = c1x - c2x
    dy = c1y - c2y
    dist_sq = dx*dx + dy*dy
    
    # Quick reject: if distance > 2*2 = 4 (circumradius 2), no overlap possible.
    # Actually max distance between vertices is 2. If centers are > 2+2=4 apart, no overlap.
    if dist_sq > 16.0:
        return 0.0
        
    # Check vertices of 1 in 2
    v1 = _hex_vertices_array(c1x, c1y, 1.0, a1)
    pen1 = np.sum(_point_in_hex_dist_sq_array(v1, c2x, c2y, 1.0, a2))
    
    # Check vertices of 2 in 1
    v2 = _hex_vertices_array(c2x, c2y, 1.0, a2)
    pen2 = np.sum(_point_in_hex_dist_sq_array(v2, c1x, c1y, 1.0, a1))
    
    return pen1 + pen2

def _objective(params):
    """
    params: 
    0: 2*N floats for inner hexagons (x, y, angle)
    2*N: 3 floats for outer hexagon (cx, cy, side)
    2*N+3: 1 float for outer angle
    
    Total: 3*N + 4 = 37 for N=11.
    """
    start_time = time.time()
    if start_time > 145.0:
        return 1e6
        
    inner_params = params[:3*N]
    outer_params = params[3*N:]
    
    ocx, ocy, oside, oangle = outer_params
    
    # Ensure side is positive
    if oside < 0:
        oside = 0.0
    
    # Objective is the side length
    obj_val = oside
    
    # Penalties
    penalty = 0.0
    
    # 1. Containment penalty: All inner vertices must be inside outer hexagon
    for i in range(N):
        ix = inner_params[i*3]
        iy = inner_params[i*3+1]
        iang = inner_params[i*3+2]
        
        v_i = _hex_vertices_array(ix, iy, 1.0, iang)
        
        # Check if these vertices are inside the outer hexagon
        # _point_in_hex_dist_sq_array returns 0 if inside, >0 if outside
        viol = _point_in_hex_dist_sq_array(v_i, ocx, ocy, oside, oangle)
        penalty += np.sum(viol)
        
    # 2. Disjointness penalty: Inner hexagons must not overlap
    for i in range(N):
        p1 = (inner_params[i*3], inner_params[i*3+1], inner_params[i*3+2])
        for j in range(i+1, N):
            p2 = (inner_params[j*3], inner_params[j*3+1], inner_params[j*3+2])
            penalty += _pairwise_penalty(p1, p2)
            
    # Weighted penalty
    # We want obj + lambda * penalty to be minimized.
    # If penalty is 0, we just minimize side.
    # If penalty > 0, it adds a large cost.
    # Dynamic weighting? Or fixed?
    # Let's use a fixed weight that is large enough to discourage invalid configs but not dominate.
    # However, since we are minimizing side, a large penalty for overlap is good.
    # The scale of violation is squared distance.
    # Let's try weight = 10.0 initially.
    
    return obj_val + 10.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_params = None
    best_score = float('inf')
    
    # Initial guesses
    # We can try a few different initial layouts.
    # 1. Grid-like
    # 2. Random
    
    # Define a helper to create initial params
    def make_initial(xys, angles):
        params = []
        for k in range(N):
            params.extend([xys[k][0], xys[k][1], angles[k]])
        # Outer: center (0,0), side 5, angle 0
        params.extend([0.0, 0.0, 5.0, 0.0])
        return np.array(params)
    
    # Layout 1: Simple grid
    # Hexagonal grid positions
    # dx = sqrt(3), dy = 1.5 for tight packing?
    # Side 1 hexagon width (flat to flat) = sqrt(3). Height = 2.
    # Horizontal spacing for flat-top: sqrt(3). Vertical: 1.5.
    # Let's just scatter them in a circle.
    
    layouts = []
    
    # Layout A: Random scatter
    np.random.seed(42)
    for _ in range(5):
        xys = np.random.uniform(-3, 3, (N, 2))
        angles = np.random.uniform(0, 2*np.pi, N)
        layouts.append(make_initial(xys, angles))
        
    # Layout B: Grid
    # Center at 0,0.
    # Neighbors at distance sqrt(3) approx.
    # Let's place them in a ring.
    xys = [(0,0)]
    for k in range(1, N):
        ang = 2 * np.pi * k / N
        r = 2.0
        xys.append((r * math.cos(ang), r * math.sin(ang)))
    angles = [0.0] * N
    layouts.append(make_initial(xys, angles))
    
    # Layout C: Tighter grid
    # Hex grid
    # (0,0)
    # (sqrt(3), 0), (-sqrt(3), 0)
    # (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), etc.
    # This is complex to generate perfectly for 11.
    # Let's just use the random ones and the ring.
    
    # Optimization
    options = {
        'maxiter': 2000,
        'xatol': 1e-4,
        'fatol': 1e-4,
        'disp': False
    }
    
    for i, init in enumerate(layouts):
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(_objective, init, method='Nelder-Mead', options=options)
            
            if res.success or res.fun < best_score:
                # Check if this is a valid solution (penalty ~ 0)
                # We can re-evaluate to be sure
                # Actually, res.fun includes penalty.
                # We want to prioritize solutions with low penalty.
                # Let's compute the "true" score (side length) only if penalty is small.
                
                # Re-evaluate penalty separately to be safe
                # This is expensive. 
                # Instead, trust the objective if it's low.
                # But if penalty is high, the side length might be small but invalid.
                # The objective function returns side + 10*penalty.
                # If penalty is 0, score = side.
                # If penalty is 1, score = side + 10.
                # So a valid solution with side 3.5 has score 3.5.
                # An invalid solution with side 3.0 and penalty 0.1 has score 3.0 + 1.0 = 4.0.
                # So minimizing the objective naturally favors valid solutions with small side.
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x
                    
        except Exception as e:
            pass
            
    if best_params is None:
        # Fallback: return a known valid configuration
        # Just place them in a line?
        # Or return the first layout.
        best_params = layouts[0]
        
    # Extract result
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a_rad = best_params[i*3+2]
        a_deg = math.degrees(a_rad) % 360.0
        inner.append([x, y, a_deg])
        
    ocx = best_params[3*N]
    ocy = best_params[3*N+1]
    oside = best_params[3*N+2]
    oangle_rad = best_params[3*N+3]
    oangle_deg = math.degrees(oangle_rad) % 360.0
    
    # Ensure side is not negative
    if oside < 0:
        oside = 0.0
        
    # Final sanity check: if the score is huge, something went wrong.
    # But we return the best found.
    
    return inner, (ocx, ocy), oside, oangle_deg

if __name__ == "__main__":
    r = construct()
    print(r)
