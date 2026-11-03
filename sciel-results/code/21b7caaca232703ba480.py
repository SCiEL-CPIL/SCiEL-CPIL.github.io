import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

N = 11
START_TIME = time.time()
DEADLINE = 145.0  # Safety margin before 150s limit

def time_remaining():
    return DEADLINE - time.time()

def hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append((cx + s * np.cos(theta), cy + s * np.sin(theta)))
    return np.array(verts)

def point_in_poly(p, poly):
    """Ray casting method for point in polygon."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        if ((poly[i][1] > p[1]) != (poly[j][1] > p[1])) and \
           (p[0] < (poly[j][0] - poly[i][0]) * (p[1] - poly[i][1]) / (poly[j][1] - poly[i][1]) + poly[i][0]):
            inside = not inside
        j = i
    return inside

def hex_inside_outer(verts_inner, outer_verts):
    """Check if all vertices of inner hex are inside or on the outer hex."""
    for v in verts_inner:
        if not point_in_poly(v, outer_verts):
            return False
    return True

def seg_seg_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

def hexes_overlap(v1, v2, eps=1e-9):
    """Check if two hexagons overlap. Returns True if they do."""
    # Check vertex containment
    for v in v1:
        if point_in_poly(v, v2):
            return True
    for v in v2:
        if point_in_poly(v, v1):
            return True
    # Check edge intersection
    for i in range(6):
        p1, p2 = v1[i], v1[(i+1)%6]
        for j in range(6):
            q1, q2 = v2[j], v2[(j+1)%6]
            if seg_seg_intersect(p1, p2, q1, q2):
                return True
    return False

def min_enclosing_hex_side(centers, angles, outer_angle=0, outer_center=(0,0)):
    """
    Find the minimum side length S of a regular hexagon with given angle and center
    that contains all inner unit hexagons at given centers and angles.
    Uses binary search on S.
    """
    ox, oy = outer_center
    
    def fits(S):
        outer_verts = hex_vertices(ox, oy, S, outer_angle)
        for i in range(len(centers)):
            inner_verts = hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
            for v in inner_verts:
                if not point_in_poly(v, outer_verts):
                    return False
        return True
    
    # Lower bound: max distance from outer center to any inner vertex
    # The hexagon with side S has inradius = S * sqrt(3)/2
    # For a point to be inside, its distance from center in the direction of the normal
    # to each face must be <= S * sqrt(3)/2
    # Simple lower bound: max distance to any vertex / (some factor)
    # Let's just start with a reasonable range
    lo = 2.0
    hi = 15.0
    
    # Ensure hi is large enough
    while not fits(hi):
        hi *= 1.5
        if hi > 100:
            break
    
    # Binary search
    for _ in range(40):
        if time_remaining() < 5.0:
            break
        mid = (lo + hi) / 2
        if fits(mid):
            hi = mid
        else:
            lo = mid
    
    return hi

def construct():
    global START_TIME
    if not hasattr(construct, '_start'):
        construct._start = time.time()
    
    # Initial configuration: Honeycomb pattern
    # Center at (0,0)
    # 6 neighbors at distance sqrt(3) * 2 = 2*sqrt(3) ? No, distance is 2*sqrt(3)/1?
    # For unit hexagons, distance between centers in honeycomb is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    # Wait, apothem of unit hex is sqrt(3)/2.
    # Distance between centers of adjacent hexes is 2 * apothem = sqrt(3).
    
    centers = np.zeros((N, 2))
    angles = np.zeros(N)  # All aligned initially
    
    # 7 cluster: center + 6 neighbors
    centers[0] = [0, 0]
    for i in range(6):
        ang = np.radians(30 + 60 * i)
        centers[1+i] = [np.sqrt(3) * np.cos(ang), np.sqrt(3) * np.sin(ang)]
    
    # 4 extras: Place them in the "gaps" or further out
    # Let's place 4 more at distance 3*sqrt(3)/2? 
    # Or just further out in the same directions or 30 deg offset?
    # Let's try placing them at distance 2*sqrt(3) in the 0, 90, 180, 270 directions?
    # Or better, extend the chain.
    # Let's place them at the vertices of a larger hexagon formed by the 6 neighbors.
    # The 6 neighbors are at radius sqrt(3).
    # The "corners" of the 7-cluster are at the vertices of the hexagon formed by connecting neighbor centers.
    # Let's place extras at distance 2*sqrt(3) in the 0, 60, 120, 180, 240, 300 directions?
    # We only have 4 extras.
    # Let's place them at (2*sqrt(3), 0), (-2*sqrt(3), 0), (0, 2*sqrt(3)), (0, -2*sqrt(3))?
    # This might cause overlap with the neighbors.
    
    # Let's use a more systematic initial guess.
    # Place the 4 extras at distance 3.5 in directions 0, 90, 180, 270.
    # Or 4.0.
    extra_dirs = [0, 90, 180, 270]
    for k in range(4):
        ang = np.radians(extra_dirs[k])
        centers[7+k] = [3.5 * np.cos(ang), 3.5 * np.sin(ang)]
    
    # Initial outer side length estimate
    S_init = 6.0
    
    # Objective function
    def objective(params):
        if time_remaining() < 5.0:
            return 1e6
        
        # params: 2*N centers + 2 outer center + 1 outer angle = 2N+3
        S = params[2*N+3] # We will optimize S too? No, we want to minimize S.
        # Actually, let's fix S and check constraints? Or minimize S subject to constraints?
        # Let's minimize S + penalty.
        # Params: 2N centers + 2 outer center + 1 outer angle + 1 S
        # Total: 2N + 4
        
        cx_inner = params[:2*N].reshape(N, 2)
        ox = params[2*N]
        oy = params[2*N+1]
        outer_angle = params[2*N+2]
        S = params[2*N+3]
        
        if S <= 0:
            return 1e6
        
        # Penalties
        penalty = 0.0
        
        # Check outer containment
        outer_verts = hex_vertices(ox, oy, S, outer_angle)
        for i in range(N):
            inner_verts = hex_vertices(cx_inner[i][0], cx_inner[i][1], 1.0, angles[i]) # Angles fixed to 0 for now?
            # We should allow rotation too, but let's start with fixed rotation for simplicity.
            for v in inner_verts:
                if not point_in_poly(v, outer_verts):
                    # Estimate how far outside
                    # Simple penalty: count of outside vertices
                    penalty += 10.0
        
        # Check pairwise disjointness
        for i in range(N):
            v1 = hex_vertices(cx_inner[i][0], cx_inner[i][1], 1.0, angles[i])
            for j in range(i+1, N):
                v2 = hex_vertices(cx_inner[j][0], cx_inner[j][1], 1.0, angles[j])
                if hexes_overlap(v1, v2):
                    penalty += 50.0
                else:
                    # Optional: add small penalty for close proximity to encourage separation
                    # But hexes_overlap is binary. We can check min distance.
                    # For speed, let's rely on the binary overlap check + strong penalty.
                    pass
        
        return S + penalty
    
    # Initial params
    x0 = np.concatenate([centers.flatten(), [0, 0, 0, S_init]])
    
    # Use Differential Evolution for robustness, then SLSQP for polish
    best_x = x0
    best_val = objective(x0)
    
    try:
        # DE
        res_de = differential_evolution(objective, bounds=[(-10,10)]*2*N + [(-5,5), (-5,5), (-180,180), (2,15)],
                                        maxiter=50, popsize=15, seed=42, tol=0.01, 
                                        init='sobol', polish=False, workers=1,
                                        update_interval=1, callback=lambda x, convergence: None)
        if res_de.success and res_de.fun < best_val:
            best_x = res_de.x
            best_val = res_de.fun
    except Exception:
        pass
    
    if time_remaining() > 20:
        try:
            # SLSQP polish
            res_sl = minimize(objective, best_x, method='SLSQP', options={'maxiter': 100, 'ftol': 1e-6})
            if res_sl.success and res_sl.fun < best_val:
                best_x = res_sl.x
                best_val = res_sl.fun
        except Exception:
            pass
    
    # Extract best parameters
    cx_inner = best_x[:2*N].reshape(N, 2)
    ox = best_x[2*N]
    oy = best_x[2*N+1]
    outer_angle = best_x[2*N+2]
    S_opt = best_x[2*N+3]
    
    # Final validation and tightening of S
    # Use the found centers and angles (0) and outer angle/center to find exact min S
    if time_remaining() > 10:
        # Check if current config is valid
        valid = True
        outer_verts = hex_vertices(ox, oy, S_opt, outer_angle)
        for i in range(N):
            inner_verts = hex_vertices(cx_inner[i][0], cx_inner[i][1], 1.0, angles[i])
            if not hex_inside_outer(inner_verts, outer_verts):
                valid = False
                break
            for j in range(i+1, N):
                v2 = hex_vertices(cx_inner[j][0], cx_inner[j][1], 1.0, angles[j])
                if hexes_overlap(inner_verts, v2):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Tighten S
            S_tight = min_enclosing_hex_side(cx_inner, angles, outer_angle, (ox, oy))
            if S_tight < S_opt:
                S_opt = S_tight
        else:
            # If invalid, fallback to initial guess or try to repair
            # For now, return initial guess
            cx_inner = centers
            ox, oy = 0, 0
            outer_angle = 0
            S_opt = S_init
    
    inner = [[float(cx_inner[i][0]), float(cx_inner[i][1]), float(angles[i])] for i in range(N)]
    return inner, (float(ox), float(oy)), float(S_opt), float(outer_angle)

