# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Assume D3h symmetry: 
       - 6 points in a plane z=h forming a regular hexagon of radius r1
       - 3 points in plane z=-h forming an equilateral triangle of radius r2
       - 3 points in plane z=h forming an equilateral triangle of radius r2 (rotated 60 deg relative to hexagon? No, D3h has 3-fold axis. 
         Let's use a more robust symmetry: 
         - 3 points at z=h, radius r_top, angles 0, 120, 240
         - 3 points at z=-h, radius r_bot, angles 60, 180, 300
         - 8 points? No, 14 points.
         
    Let's look at the known optimal 14-point configuration.
    It is often described as:
    - 2 points at poles (0,0,1), (0,0,-1)
    - 12 points in two rings? 
    - Or 6 points in a hexagon, 3 above, 3 below, 2 poles? Total 14.
    
    Let's try a specific high-symmetry family that is a strong candidate:
    Family A: 
    - 2 points: (0,0,1), (0,0,-1)
    - 6 points: Regular hexagon in z=0 plane, radius r1
    - 6 points: Two triangles? 
      Actually, a common optimal structure for 14 is:
      - 2 poles
      - 6 points in a hexagon at z=h1
      - 6 points in a hexagon at z=-h1 (rotated 30 deg)
      Total 14. This has D6h symmetry (if aligned) or D3d (if rotated).
      
    Let's parameterize this D6h/D3d family:
    x = [r1, h1, r2, h2, delta_angle]
    Points:
    P0: (0,0,1)
    P1: (0,0,-1)
    Ring 1 (6 pts): z=h1, radius r1, angles k*60 deg
    Ring 2 (6 pts): z=-h1, radius r2, angles k*60 deg + delta
    
    Wait, if we fix poles at (0,0,1) and (0,0,-1), the max distance is likely 2 (between poles) or close to it.
    If we don't fix poles, we can scale.
    
    Let's use a more general 5-parameter family that captures the "two rings + poles" or "three rings" structure.
    
    Alternative Strategy: 
    Use a known good starting point from the literature or a high-quality random search in Cartesian space with re-centering and scaling, but use a **direct search method (Nelder-Mead)** on the Cartesian coordinates directly, but with a **penalty-based constraint** to keep points on the unit sphere? No, that's what the current one does poorly.
    
    Better: 
    The metric is scale invariant. We can fix the points to be on the unit sphere.
    The current program uses spherical coordinates. The problem with spherical coordinates is the singularity at the poles and the non-linearity.
    
    New Approach: 
    **Cayley-Klein Parameters / Quaternions?** Too complex.
    
    Let's try **Symmetry-Breaking Optimization** on a specific high-symmetry manifold.
    
    Candidate Symmetry: D3d (Octahedral-like distortion)
    14 points:
    - 2 points on z-axis: (0,0,z0), (0,0,-z0)
    - 12 points in two rings of 6?
    
    Actually, let's look at Inspiration 1 (Score 51.8): "exploit... specific symmetry group (specifically D3h or C3v related to triangular prisms/an...)"
    
    Let's implement a specific D3h symmetric configuration:
    1. 3 points at top: (r_t, 0, h_t), (r_t, 0, h_t) rotated 120, 240.
    2. 3 points at bottom: (r_b, 0, -h_b) rotated 60, 180, 300 (staggered).
    3. 8 points? No, 14 total. 3+3=6. Remaining 8.
    
    Maybe:
    1. 2 points: (0,0,1), (0,0,-1)
    2. 12 points: Two hexagonal rings.
       Ring 1: z=h, radius r1, 6 points.
       Ring 2: z=-h, radius r2, 6 points.
    
    Parameters: r1, r2, h, delta (rotation of ring 2 relative to ring 1).
    Total 4 parameters.
    
    Let's optimize this 4-parameter family using Nelder-Mead.
    We will also include a "free" Cartesian optimization for the best result found, but initialized from the symmetry solution.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580

    def get_points_symmetry(params):
        r1, r2, h, delta = params
        # Clip parameters to reasonable ranges
        r1 = np.clip(r1, 0, 1.5)
        r2 = np.clip(r2, 0, 1.5)
        h = np.clip(h, -1.5, 1.5)
        
        points = np.zeros((14, 3))
        
        # Poles
        points[0] = [0, 0, 1]
        points[1] = [0, 0, -1]
        
        # Ring 1 (6 points) at z=h
        angles1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
        for i in range(6):
            points[2+i] = [r1 * np.cos(angles1[i]), r1 * np.sin(angles1[i]), h]
            
        # Ring 2 (6 points) at z=-h, rotated by delta
        angles2 = angles1 + delta
        for i in range(6):
            points[8+i] = [r2 * np.cos(angles2[i]), r2 * np.sin(angles2[i]), -h]
            
        return points

    def objective_symmetry(params):
        if time.time() > deadline:
            return 1e10
        pts = get_points_symmetry(params)
        # Normalize to unit sphere? 
        # The metric is scale invariant. 
        # However, if we fix poles at (0,0,1), the scale is fixed.
        # But the optimal configuration might not have poles at distance 1 from origin if we allow scaling.
        # Since we can scale the whole configuration, fixing poles at (0,0,1) is a valid gauge choice ONLY IF the optimal config has 2 antipodal points.
        # Does it? Likely yes, for even N, antipodal pairs are common.
        
        # To be safe, let's allow the poles to move? 
        # If we fix poles, we restrict the search space.
        # Let's add parameters for pole height? 
        # If we fix poles at (0,0,1) and (0,0,-1), the max distance is at least 2.
        # If the optimal max distance is > 2, we are constrained.
        # But we can scale the final result.
        # So, let's just find the shape that maximizes d_min / d_max.
        # We can compute d_min and d_max for the generated points, then return -(d_min/d_max)^2.
        
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-10:
            return 1e10
        ratio = d_min / d_max
        return -ratio**2

    # Multi-start Nelder-Mead on the 4 parameters
    best_params = None
    best_score = -np.inf
    best_pts = None
    
    # Initial guesses
    # r1, r2, h, delta
    # Symmetric case: r1=r2, delta=0 or pi/6
    starts = [
        [1.0, 1.0, 0.5, 0.0],
        [1.0, 1.0, 0.5, np.pi/6],
        [0.8, 0.8, 0.3, 0.0],
        [0.8, 0.8, 0.3, np.pi/6],
        [1.2, 1.2, 0.8, 0.0],
        [1.2, 1.2, 0.8, np.pi/6],
        [0.5, 0.5, 0.1, 0.0],
        [0.5, 0.5, 0.1, np.pi/6],
        [1.0, 0.8, 0.5, 0.0],
        [0.8, 1.0, 0.5, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, np.pi/6],
        [0.7, 0.7, 0.5, 0.0],
        [0.7, 0.7, 0.5, np.pi/6],
    ]
    
    # Add random starts
    np.random.seed(42)
    for _ in range(10):
        r1 = np.random.uniform(0.5, 1.5)
        r2 = np.random.uniform(0.5, 1.5)
        h = np.random.uniform(0.1, 1.0)
        delta = np.random.uniform(0, np.pi/3)
        starts.append([r1, r2, h, delta])
        
    bounds = None # NM doesn't use bounds, we clip inside get_points_symmetry
    
    for s in starts:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                objective_symmetry, 
                s, 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
                best_pts = get_points_symmetry(best_params)
        except Exception:
            continue

    # Refine the best symmetric solution with a Cartesian Nelder-Mead
    # This allows symmetry breaking if it helps, starting from a good symmetric guess.
    if best_pts is not None and time.time() < deadline - 20:
        def objective_cartesian(x_flat):
            pts = x_flat.reshape(n, d)
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max < 1e-10:
                return 1e10
            ratio = d_min / d_max
            return -ratio**2

        try:
            res_nm = minimize(
                objective_cartesian, 
                best_pts.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_pts = res_nm.x.reshape(n, d)
        except Exception:
            pass

    # If no points found, fallback
    if best_pts is None:
        # Fallback to icosahedron + poles
        phi_golden = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi_golden], [0, -1, phi_golden],
            [0, 1, -phi_golden], [0, -1, -phi_golden],
            [1, phi_golden, 0], [-1, phi_golden, 0],
            [1, -phi_golden, 0], [-1, -phi_golden, 0],
            [phi_golden, 0, 1], [-phi_golden, 0, 1],
            [phi_golden, 0, -1], [-phi_golden, 0, -1]
        ])
        ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1)[:, np.newaxis]
        pole1 = np.array([0, 0, 1])
        pole2 = np.array([0, 0, -1])
        best_pts = np.vstack([ico_verts, pole1, pole2])

    # Final processing: Center and Scale
    points = best_pts
    points -= np.mean(points, axis=0)
    
    # Scale to unit max distance
    dists = pdist(points)
    d_max = np.max(dists)
    if d_max > 1e-12:
        points = points / d_max

    # Ensure finite
    if not np.all(np.isfinite(points)):
        phi_golden = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi_golden], [0, -1, phi_golden],
            [0, 1, -phi_golden], [0, -1, -phi_golden],
            [1, phi_golden, 0], [-1, phi_golden, 0],
            [1, -phi_golden, 0], [-1, -phi_golden, 0],
            [phi_golden, 0, 1], [-phi_golden, 0, 1],
            [phi_golden, 0, -1], [-phi_golden, 0, -1]
        ])
        ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1)[:, np.newaxis]
        points = np.zeros((n, d))
        points[:12] = ico_verts
        points[12] = [0, 0, 1]
        points[13] = [0, 0, -1]
        points -= np.mean(points, axis=0)
        dists = pdist(points)
        d_max = np.max(dists)
        if d_max > 0:
            points = points / d_max

    return points

# EVOLVE-BLOCK-END
