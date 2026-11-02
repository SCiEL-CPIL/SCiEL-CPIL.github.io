from scipy.optimize import minimize
import math
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

# Pre-compute normals for hexagon orientation (relative)
# A regular hexagon with side 1 has apothem sqrt(3)/2.
# The Minkowski sum of two such hexagons (same orientation) has apothem sqrt(3).
# For non-overlap, the distance between centers projected onto the 6 normals must be >= sqrt(3).
# The normals are perpendicular to the sides.
# If the hexagon is rotated by angle theta, the normals rotate by theta.
# We assume all inner hexagons have the same rotation relative to the outer one? 
# The task says "each inner hexagon may translate and rotate". 
# However, optimal packings of identical convex bodies usually align them or have specific symmetries.
# Given the complexity, we will assume all inner hexagons have the SAME rotation angle 'alpha' relative to the outer hexagon.
# This reduces the dimensionality significantly. The outer hexagon rotation is 'beta'.
# The relative rotation is 'delta = alpha - beta'.
# We can fix beta = 0 without loss of generality for the inner relative positions, 
# and then optimize the global rotation of the whole cluster if needed? 
# Actually, the score is the side length of the outer hexagon.
# The outer hexagon can be rotated freely. The inner hexagons can be rotated freely.
# Let's fix the outer hexagon to be "flat top" (or pointy top) and optimize the inner hexagons' positions and a common rotation angle for the inner hexagons.
# Or better: Optimize positions of inner hexagons and their individual rotations? 
# With N=11, individual rotations are 11 extra variables. Total 22 + 11 = 33 vars.
# But physics relaxation works best in 2D.
# Let's stick to the assumption: All inner hexagons have the same rotation angle 'phi' relative to the outer hexagon.
# We can set the outer hexagon rotation to 0. Then we optimize positions (2N) and phi (1).
# The outer hexagon side length S is determined by the max extent of the inner hexagons.

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: trivial solution
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def get_normals(phi):
        # Normals for a hexagon rotated by phi
        # Standard hexagon (pointy top) vertices at angles 30, 90, 150...
        # Normals are perpendicular to sides.
        # Side 1 is horizontal? No, let's define standard orientation.
        # Let's use the convention that the "up" normal is at 90 degrees (pi/2).
        # The 6 normals are at angles: pi/2 + k*pi/3
        angles = np.array([math.pi / 2 + k * math.pi / 3 for k in range(6)]) + phi
        return np.column_stack((np.cos(angles), np.sin(angles)))

    def calc_container_size(pts, phi):
        # pts: (N, 2) array of centers
        # phi: rotation angle of inner hexagons relative to outer (outer is fixed at 0)
        # We need to find the minimum side length S of a regular hexagon (fixed orientation, e.g., pointy top)
        # that contains all inner hexagons.
        # An inner hexagon with center c and rotation phi is contained in an outer hexagon (center O, side S, rotation 0)
        # iff for all 6 normals n_k of the OUTER hexagon, the support function of the inner hexagon in direction n_k
        # plus the projection of c-O onto n_k is <= S * apothem_outer.
        # Wait, the outer hexagon is defined by its side length S. Its apothem is S * sqrt(3)/2.
        # The condition for a point p to be in outer hexagon (center 0, side S) is:
        # |p . n_k| <= S * apothem_unit for all k? No.
        # The outer hexagon is the intersection of 6 half-planes: p . n_k <= S * apothem_unit.
        # For an inner hexagon (center c, rot phi, side 1), its support function in direction n_k is:
        # h(n_k) = c . n_k + h_inner(n_k)
        # where h_inner(n_k) is the support function of a unit hexagon rotated by phi in direction n_k.
        # h_inner(n_k) = max_{v in inner_hex} v . n_k
        # For a regular hexagon of side 1, apothem a = sqrt(3)/2.
        # The support function in direction n is a * cos(delta) where delta is the angle between n and the nearest normal of the hexagon?
        # Actually, for a regular polygon, h(n) = apothem / cos(angle_diff) is not quite right for all angles.
        # For a hexagon, h(n) = apothem * max(cos(angle_diff), ...)
        # Let's compute it explicitly.
        # The vertices of a unit hexagon (rot 0) are at distance 1 from center.
        # Vertices: (cos(30+60k), sin(30+60k))
        # The support function h(n) = max(v_i . n)
        # We can precompute this or compute on the fly.
        
        # Outer hexagon normals (fixed orientation, let's say pointy top, so one normal is vertical)
        outer_normals = np.array([[math.cos(math.pi / 2 + k * math.pi / 3), math.sin(math.pi / 2 + k * math.pi / 3)] for k in range(6)])
        
        # Inner hexagon support function in direction n
        # Inner hexagon vertices (relative to its center, rot phi):
        # v_j = (cos(phi + 30 + 60j), sin(phi + 30 + 60j))
        # h_inner(n) = max_j (v_j . n)
        
        max_required_apothem = 0.0
        for i in range(N):
            c = pts[i]
            # Support of inner hexagon in each outer normal direction
            # We can compute h_inner for all 6 outer normals at once
            # v_j are 6 vertices
            angles_v = phi + math.pi/6 + np.arange(6) * math.pi/3
            verts = np.column_stack((np.cos(angles_v), np.sin(angles_v)))
            # Projection of vertices onto outer_normals
            # proj[j, k] = v_j . n_k
            proj = verts @ outer_normals.T
            h_inner = np.max(proj, axis=0) # shape (6,)
            
            # Required apothem for this hexagon:
            # c . n_k + h_inner[k] <= S * apothem_unit
            # S * apothem_unit >= c . n_k + h_inner[k]
            # Also need to consider the other side? No, the hexagon is centered at 0.
            # The condition is |p| <= ... no, it's a polygon.
            # The polygon is defined by p . n_k <= A for all k, where A = S * apothem_unit.
            # So we need A >= max_k (c . n_k + h_inner[k])
            # AND A >= max_k (-c . n_k - h_inner(-n_k)) ?
            # Since the set of normals includes both n and -n (effectively, as k goes 0..5, we cover 360 deg),
            # we just need to check all 6.
            # Wait, outer_normals has 6 vectors 60 deg apart. They are not opposite pairs.
            # n_0 and n_3 are opposite. n_1 and n_4 are opposite. n_2 and n_5 are opposite.
            # So checking all 6 covers both sides.
            
            reqs = c @ outer_normals.T + h_inner
            max_req = np.max(reqs)
            if max_req > max_required_apothem:
                max_required_apothem = max_req
        
        # S * apothem_unit = max_required_apothem
        # S = max_required_apothem / apothem_unit
        return max_required_apothem / APOTHEM

    def relaxation_step(pts, phi, forces):
        # Apply forces to update positions
        # This is a simplified Verlet or Euler integration
        pass

    def optimize_cluster(pts_init, phi_init):
        # Use scipy minimize to find the tightest packing
        # Variables: x1, y1, ..., xN, yN, phi
        # Objective: calc_container_size
        # Constraints: Non-overlap.
        # Non-overlap constraint: For all pairs i,j, distance between centers projected on inner normals >= sqrt(3)
        # This is a non-convex constraint.
        # We will use a penalty method.
        
        def objective(params):
            pts = params[:2*N].reshape(N, 2)
            phi = params[2*N]
            
            # 1. Container size
            S = calc_container_size(pts, phi)
            
            # 2. Overlap penalty
            penalty = 0.0
            normals = get_normals(phi)
            for i in range(N):
                for j in range(i+1, N):
                    d = pts[i] - pts[j]
                    projs = np.abs(d @ normals.T)
                    min_proj = np.min(projs)
                    if min_proj < math.sqrt(3):
                        violation = math.sqrt(3) - min_proj
                        penalty += violation * 1000.0
            
            return S + penalty

        init_params = np.concatenate([pts_init.ravel(), [phi_init]])
        
        # Use Nelder-Mead for robustness with non-differentiable penalty
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.success or res.fun < 100:
                best_pts = res.x[:2*N].reshape(N, 2)
                best_phi = res.x[2*N]
                best_S = calc_container_size(best_pts, best_phi)
                # Verify no overlap
                normals = get_normals(best_phi)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        d = best_pts[i] - best_pts[j]
                        projs = np.abs(d @ normals.T)
                        if np.min(projs) < math.sqrt(3) - 1e-9:
                            valid = False
                if valid:
                    return best_S, best_pts, best_phi
        except Exception:
            pass
        return float('inf'), None, None

    # Generate diverse initial configurations
    inits = []
    
    # 1. Hexagonal spiral (standard)
    axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (1,-2)]
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        pts.append([x, y])
    inits.append((np.array(pts), 0.0))
    
    # 2. Random cluster
    np.random.seed(42)
    pts_rand = np.random.rand(N, 2) * 4 - 2
    inits.append((pts_rand, np.random.rand() * math.pi))
    
    # 3. Circle packing
    angles = np.linspace(0, 2*math.pi, N, endpoint=False)
    radii = np.zeros(N)
    radii[0] = 0
    for i in range(1, N):
        radii[i] = 2.0 # Rough estimate
    pts_circ = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    inits.append((pts_circ, 0.0))

    best_S = float('inf')
    best_pts = None
    best_phi = 0.0

    for pts_init, phi_init in inits:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        S, pts_opt, phi_opt = optimize_cluster(pts_init, phi_init)
        if S < best_S:
            best_S = S
            best_pts = pts_opt
            best_phi = phi_opt

    if best_pts is None:
        # Fallback
        best_S = 10.0
        best_pts = np.array([[0.0, 0.0]] * N)
        best_phi = 0.0

    # Final Polish: Refine the best solution with tighter optimization
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        def final_obj(params):
            pts = params[:2*N].reshape(N, 2)
            phi = params[2*N]
            S = calc_container_size(pts, phi)
            penalty = 0.0
            normals = get_normals(phi)
            for i in range(N):
                for j in range(i+1, N):
                    d = pts[i] - pts[j]
                    projs = np.abs(d @ normals.T)
                    min_proj = np.min(projs)
                    if min_proj < math.sqrt(3):
                        violation = math.sqrt(3) - min_proj
                        penalty += violation * 10000.0
            return S + penalty

        init_params = np.concatenate([best_pts.ravel(), [best_phi]])
        try:
            res = minimize(final_obj, init_params, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_S + 0.01:
                pts_final = res.x[:2*N].reshape(N, 2)
                phi_final = res.x[2*N]
                S_final = calc_container_size(pts_final, phi_final)
                # Verify
                normals = get_normals(phi_final)
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        d = pts_final[i] - pts_final[j]
                        projs = np.abs(d @ normals.T)
                        if np.min(projs) < math.sqrt(3) - 1e-9:
                            valid = False
                if valid and S_final < best_S:
                    best_S = S_final
                    best_pts = pts_final
                    best_phi = phi_final
        except Exception:
            pass

    # Construct output
    # The outer hexagon is centered at (0,0) and has rotation 0 (pointy top).
    # The inner hexagons are at best_pts with rotation best_phi.
    # We need to return:
    # inner: list of [x, y, angle_degrees]
    # center: [x, y] of outer hexagon
    # side: side length
    # angle: rotation of outer hexagon (0.0)
    
    inner = []
    for x, y in best_pts:
        inner.append([float(x), float(y), float(math.degrees(best_phi))])
        
    center = [0.0, 0.0]
    side = float(best_S)
    rotation = 0.0

    return inner, center, side, rotation
