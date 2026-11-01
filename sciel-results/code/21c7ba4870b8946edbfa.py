import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_degrees)
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    # Normals for a regular hexagon centered at origin, pointy-top or flat-top?
    # The formula 2 * max_proj / sqrt(3) works for any orientation if normals are correct.
    # For a hexagon with side S, the distance from center to edge (apothem) is S*sqrt(3)/2.
    # max_proj is the maximum projection of inner vertices onto the outer normals.
    # Side length S = 2 * max_proj / sqrt(3).
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    # Rotation matrix to transform outer normals to global frame? 
    # Or transform inner points to outer frame?
    # Let's transform inner points to outer local frame.
    # Outer local frame is rotated by outer_angle_deg relative to global.
    # R = [[cos, sin], [-sin, cos]] transforms global -> local if local is rotated by -angle?
    # Standard rotation: x' = x cos(a) + y sin(a), y' = -x sin(a) + y cos(a)
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    if max_proj < 0:
        max_proj = 0.0
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth (log-sum-exp) surrogate for the max projection.
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    all_projs = []
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        all_projs.append(projs)
    P = np.concatenate([p.ravel() for p in all_projs])
    if P.size == 0:
        return 0.0
    m = P.max()
    if m < -1e6:
        return 0.0
    smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
    if smooth_max < 0:
        smooth_max = 0.0
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons (lists of (x,y))."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]; p2 = poly[(i + 1) % len(poly)]
            e = (p2[0] - p1[0], p2[1] - p1[1])
            n = (-e[1], e[0])
            ln = math.hypot(n[0], n[1])
            if ln > 1e-12:
                axes.append((n[0] / ln, n[1] / ln))
    for ax in axes:
        min1 = min(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        max1 = max(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        min2 = min(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        max2 = max(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT check that all inner hexagons are disjoint."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def get_cluster_positions():
    """
    Returns the local positions of the 11 hexagons in a compact cluster.
    Configuration: 1 center, 6 surrounding, 4 more (forming a 3-4-3 or similar shape).
    Using axial coordinates (q, r) with x = 1.5*q, y = sqrt(3)*(r + q/2).
    
    Standard dense packing for 11:
    Row 0: -1, 0, 1 (3 hexes) -> q=-1,0,1 r=0
    Row 1: -1, 0, 1, 2 (4 hexes) -> q=-1,0,1,2 r=-1
    Row 2: -1, 0, 1 (3 hexes) -> q=-1,0,1 r=-2
    Total: 3+4+3 = 10. Need 11.
    
    Let's use the "flower" plus extras:
    Center (0,0)
    Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    Ring 2 (partial): (2,-1), (1,1), (-2,1), (-1,-1) -> 4 hexes
    Total: 1 + 6 + 4 = 11.
    """
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    positions = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        positions.append([x, y])
    return np.array(positions)

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin

    cluster_local = get_cluster_positions()
    # Center the cluster locally so it's centered at origin
    centroid = np.mean(cluster_local, axis=0)
    cluster_local = cluster_local - centroid

    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Define the objective function for the rigid composite
    # Params: [X, Y, Theta, Cx, Cy, Phi]
    # X, Y, Theta: Composite translation and rotation
    # Cx, Cy, Phi: Outer hexagon center and rotation
    
    def objective(params):
        X, Y, Theta, Cx, Cy, Phi = params
        theta_rad = math.radians(Theta)
        c_t = math.cos(theta_rad)
        s_t = math.sin(theta_rad)
        R_cluster = np.array([[c_t, -s_t], [s_t, c_t]])
        
        # Transform local cluster positions to global
        # p_global = R_cluster * p_local + [X, Y]
        global_positions = (cluster_local @ R_cluster.T) + np.array([X, Y])
        
        # The inner hexagons are also rotated by Theta (rigid body)
        # We need to pass inner hexagons to compute_required_side... 
        # Wait, compute_required_side expects a list of (x,y,angle).
        # Since all hexagons in the cluster are rotated by the same amount Theta,
        # we can just pass that angle.
        
        inner_list = []
        for i in range(N):
            gx, gy = global_positions[i]
            inner_list.append((gx, gy, Theta))
        
        # Compute side length
        side = compute_required_side_fast(inner_list, (Cx, Cy), Phi)
        
        # Check disjointness? 
        # In a rigid lattice with unit spacing, they are disjoint by construction 
        # (distance between centers is >= sqrt(3) for neighbors? No, distance is sqrt(3)*1 for neighbors).
        # Distance between centers of adjacent unit hexagons in a lattice is sqrt(3).
        # The "radius" of a unit hexagon (apothem) is sqrt(3)/2.
        # So distance between edges is sqrt(3) - 2*(sqrt(3)/2) = 0.
        # They touch but do not overlap.
        # However, if the lattice is slightly compressed or rotated differently, it might overlap?
        # No, the local positions are fixed relative to each other. 
        # The only risk is if my axial coordinate calculation creates overlaps.
        # Axial coords with spacing 1.5 and sqrt(3) correspond to unit hexagons touching.
        # So disjointness is guaranteed for any rigid transformation.
        # We just need to ensure the cluster itself is valid.
        
        return side

    # Initial guesses
    # Start with cluster at origin, outer at origin
    init_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Multi-start for global optimization
    np.random.seed(42)
    starts = [init_params]
    for _ in range(10):
        # Random small perturbations
        p = init_params + np.random.normal(0, 1.0, 6)
        # Ensure angles are reasonable
        p[2] = np.random.uniform(0, 60)
        p[5] = np.random.uniform(0, 60)
        starts.append(p)
        
    for p0 in starts:
        if time.time() > deadline:
            break
        
        try:
            # Phase 1: Nelder-Mead for global search
            res1 = minimize(objective, p0, method='Nelder-Mead',
                            options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res1.fun < best_side:
                # Phase 2: BFGS for local polish
                res2 = minimize(objective, res1.x, method='BFGS',
                                options={'maxiter': 500, 'gtol': 1e-8})
                
                if res2.fun < res1.fun:
                    best_candidate = res2
                else:
                    best_candidate = res1
                
                # Verify and update best
                X, Y, Theta, Cx, Cy, Phi = best_candidate.x
                theta_rad = math.radians(Theta)
                c_t = math.cos(theta_rad)
                s_t = math.sin(theta_rad)
                R_cluster = np.array([[c_t, -s_t], [s_t, c_t]])
                global_positions = (cluster_local @ R_cluster.T) + np.array([X, Y])
                
                inner_list = []
                for i in range(N):
                    gx, gy = global_positions[i]
                    inner_list.append((float(gx), float(gy), float(Theta)))
                
                # Double check disjointness just in case (should be true)
                if check_disjoint(inner_list, eps=1e-5):
                    exact_side = compute_required_side_fast(inner_list, (Cx, Cy), Phi)
                    if exact_side < best_side:
                        best_side = exact_side
                        best_inner = inner_list
                        best_outer_center = (float(Cx), float(Cy))
                        best_outer_angle = float(Phi)
                        
        except Exception as e:
            pass

    # If no valid solution found (should not happen), fallback
    if best_inner is None:
        X, Y, Theta, Cx, Cy, Phi = 0, 0, 0, 0, 0, 0
        R_cluster = np.eye(2)
        global_positions = cluster_local
        inner_list = [(float(gx), float(gy), 0.0) for gx, gy in global_positions]
        best_inner = inner_list
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(inner_list, (0.0, 0.0), 0.0)

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
