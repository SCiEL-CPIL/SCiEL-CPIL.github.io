# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Strategy:
    1. Assume the optimal configuration has a specific symmetry (D3h or similar).
    2. Parameterize the 14 points using a small number of geometric parameters (2D search space).
       - The configuration is modeled as two concentric equilateral triangles (prism/antiprism like)
         plus 2 poles? No, 14 points.
         A common high-symmetry configuration for 14 points is derived from a triangular prism 
         or related polyhedra. 
         Let's consider a configuration with 3-fold symmetry.
         14 points is not a standard highly symmetric polyhedron vertex count like 12 (ico) or 20 (dodeca).
         However, we can try a configuration with 3-fold rotational symmetry.
         14 = 3 + 3 + 3 + 3 + 2? 
         Or 14 = 12 (ico) + 2 (poles) which the current program uses.
         
         Let's try a different symmetry: D3h symmetry allows for sets of points in rings.
         Ring 1: 3 points at (r1, 0, z1) rotated by 120 deg.
         Ring 2: 3 points at (r2, 0, z2) rotated by 120 deg.
         Ring 3: 3 points at (r3, 0, z3) rotated by 120 deg.
         Ring 4: 3 points at (r4, 0, z4) rotated by 120 deg.
         Ring 5: 2 points? This breaks the 3-fold symmetry.
         
         Alternative: 14 points could be 12 points on a sphere (icosahedron) + 2 poles is one guess.
         Another guess: 14 points might be better arranged as two parallel hexagons + 2 points?
         Or a configuration based on the triangular prism but with more points.
         
         Let's use a more flexible 2D parameterization that captures a broad class of symmetric solutions.
         We will assume 3-fold symmetry for 12 of the points, and place the remaining 2 points on the axis of symmetry.
         Points:
         - 2 points on the z-axis: (0, 0, h1), (0, 0, -h1) [Assuming centrosymmetry for the poles]
         - 12 points in 4 rings of 3, or 3 rings of 4?
           If 3-fold symmetry:
           Ring 1: 3 points at radius r1, height z1
           Ring 2: 3 points at radius r2, height z2
           Ring 3: 3 points at radius r3, height z3
           Ring 4: 3 points at radius r4, height z4
           Total 12 points. Plus 2 poles = 14.
           
         Parameters: r1, z1, r2, z2, r3, z3, r4, z4, h1. (9 parameters). This is still too high for "2D".
         
         Let's try a stricter symmetry: D3h with 2 rings of 6? No, 14 is not divisible by 6.
         What if we assume the 14 points are vertices of a specific polyhedron?
         The "14-point problem" in 3D often relates to packing.
         
         Let's look at the "inspiration 1" hint: "symmetry-const".
         Let's try a configuration with 2-fold symmetry (centrosymmetric) and 3-fold?
         
         Actually, a very effective strategy for small N is to assume the points lie on a sphere and use a specific symmetric template.
         For N=14, a strong candidate is the Icosahedron (12) + 2 Poles, which is already tried.
         Another candidate: Two concentric hexagons (12) + 2 poles?
         Or a triangular prism (6) + another triangular prism (6) + 2 poles?
         
         Let's parameterize a configuration with 3-fold symmetry and centrosymmetry.
         Points:
         1. (0, 0, h)
         2. (0, 0, -h)
         3-8. 6 points forming a hexagonal ring at z=0? (r, 0, 0), (r, 120, 0)...
         9-14. 6 points forming another hexagonal ring at z=0? No.
         
         Let's try:
         - 2 poles: (0,0,h), (0,0,-h)
         - 6 points in a ring at z1: radius r1
         - 6 points in a ring at z2: radius r2
         Total 14.
         If we assume the rings are regular hexagons.
         Parameters: h, r1, z1, r2, z2. (5 parameters).
         We can fix scale by setting h=1 or r1=1.
         
         Let's search this 4D space (h, r1, z1, r2, z2) with h fixed to 1? Or scale invariant.
         Since the metric is scale invariant, we can fix one parameter, e.g., h=1.
         Then we search r1, z1, r2, z2.
         
         Initial guess:
         Hexagonal rings are often optimal for local packing.
         Let's try r1=r2=1, z1=0.5, z2=-0.5, h=1.
         
         We will perform a dense grid search over these 4 parameters and then refine the best point with Nelder-Mead.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580

    def generate_points(params):
        # params: [r1, z1, r2, z2, h]
        # We fix h=1 for scale invariance, but let's keep it in the search for flexibility 
        # and normalize at the end. Actually, fixing h=1 reduces dim to 4.
        # Let's use params: [r1, z1, r2, z2] with h=1.
        r1, z1, r2, z2 = params
        h = 1.0
        
        pts = []
        # Poles
        pts.append([0, 0, h])
        pts.append([0, 0, -h])
        
        # Ring 1: 6 points
        for i in range(6):
            angle = 2 * np.pi * i / 6
            pts.append([r1 * np.cos(angle), r1 * np.sin(angle), z1])
            
        # Ring 2: 6 points
        for i in range(6):
            angle = 2 * np.pi * i / 6 + np.pi/6 # Offset by 30 deg (antiprism-like)
            pts.append([r2 * np.cos(angle), r2 * np.sin(angle), z2])
            
        return np.array(pts)

    def objective(params):
        pts = generate_points(params)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0
        ratio = d_min / d_max
        return -ratio**2 # Minimize negative squared ratio

    # Grid Search
    # r1, r2: 0.5 to 2.0
    # z1, z2: -0.5 to 0.5
    
    r_vals = np.linspace(0.5, 2.0, 20)
    z_vals = np.linspace(-0.5, 0.5, 10)
    
    best_score = -np.inf
    best_params = None
    
    # Coarse grid
    for r1 in r_vals:
        for z1 in z_vals:
            for r2 in r_vals:
                for z2 in z_vals:
                    if time.time() > deadline - 100:
                        break
                    params = np.array([r1, z1, r2, z2])
                    score = objective(params)
                    if score > best_score:
                        best_score = score
                        best_params = params

    # Refine best parameter set using Nelder-Mead
    if best_params is not None and time.time() < deadline - 20:
        def obj_refine(params):
            return objective(params)
        
        try:
            res = minimize(
                obj_refine, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Generate final points
    if best_params is not None:
        points = generate_points(best_params)
    else:
        # Fallback: Icosahedron + Poles
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
        points = np.vstack([ico_verts, pole1, pole2])

    # Normalize and center
    points -= np.mean(points, axis=0)
    dists = pdist(points)
    d_max = np.max(dists)
    if d_max > 1e-12:
        points = points / d_max

    if not np.all(np.isfinite(points)):
        # Ultimate fallback
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
