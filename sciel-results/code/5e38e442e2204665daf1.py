import math
import itertools
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0

    # Hexagon geometry constants
    # Vertices of a unit hexagon (side 1) centered at origin, vertex at 0 deg
    # Angles: 0, pi/3, 2pi/3, pi, 4pi/3, 5pi/3
    base_verts = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])
    
    def rotate_points(points, angle):
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        return points @ R.T

    def calc_S_rigid(cluster_verts, outer_theta):
        """
        Calculate the minimum side length S of the outer hexagon 
        to contain the given cluster vertices with the given outer rotation.
        
        cluster_verts: (M, 2) array of all vertices of all inner hexagons, 
                       already rotated and translated to global coordinates.
        outer_theta: rotation of the outer hexagon.
        """
        # Outer normals
        normals = []
        for k in range(6):
            angle_n = outer_theta + k * math.pi / 3
            normals.append([math.cos(angle_n), math.sin(angle_n)])
        normals = np.array(normals) # (6, 2)
        
        # Project all cluster vertices onto all outer normals
        # cluster_verts: (M, 2)
        # normals: (6, 2)
        # dots: (M, 6) where dots[i, k] = cluster_verts[i] dot normals[k]
        dots = cluster_verts @ normals.T
        
        # The condition for containment is that for each normal k,
        # max(dots[:, k]) - min(dots[:, k]) <= S * sqrt(3)
        # Wait, the distance between parallel sides of a regular hexagon of side S is S * sqrt(3).
        # The projection width of the cluster onto normal k is width_k = max(dots[:, k]) - min(dots[:, k]).
        # We need width_k <= S * sqrt(3) for all k.
        # So S >= max_k (width_k / sqrt(3)).
        
        widths = np.max(dots, axis=0) - np.min(dots, axis=0)
        max_width = np.max(widths)
        S = max_width / math.sqrt(3)
        
        return S

    def get_cluster_verts(tx, ty, theta_cluster):
        """
        Generate global vertices for the rigid cluster of 11 hexagons.
        """
        # Define relative centers of the 11 hexagons in the cluster
        # Using axial coordinates for hex grid
        # Center: (0,0)
        # Ring 1: 6 neighbors
        # Ring 2: 4 selected neighbors for symmetry (e.g., at angles 0, 90, 180, 270 in hex grid logic? No, hex grid is 60 deg symmetry)
        # Let's pick a symmetric set of 4 from ring 2.
        # Axial coords (q, r):
        # Center: (0,0)
        # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Ring 2 candidates: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
        # Let's pick (2,0), (0,2), (-2,1), (-1,-1) to form a diamond-like shape?
        # Or (2,0), (-1,1), (-1,-1), (0,-2)? 
        # A common tight packing for 11 is a "flower" of 7 plus 4 more.
        # Let's use the first 4 from the ring 2 list that are spread out.
        # (2,0), (1,1), (0,2), (-1,2) -> This is a chunk.
        # Let's try (2,0), (-2,0), (0,2), (0,-2) if they exist? 
        # (2,0) exists. (-2,0) exists. (0,2) exists. (0,-2) exists.
        # These 4 are opposite each other.
        
        centers_axial = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2)
        ]
        
        # Convert axial to Cartesian
        # x = sqrt(3) * (q + r/2)
        # y = 3/2 * r
        # Side length 1 -> distance between centers is sqrt(3)
        centers_cart = []
        for q, r in centers_axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            centers_cart.append([x, y])
        
        centers_cart = np.array(centers_cart)
        
        # Rotate the cluster around its center (which is at 0,0 in local coords)
        rotated_centers = rotate_points(centers_cart, theta_cluster)
        
        # Translate
        rotated_centers = rotated_centers + np.array([tx, ty])
        
        # Generate vertices for each hexagon
        # We need all vertices of all 11 hexagons.
        # base_verts is (6, 2)
        # rotated_centers is (11, 2)
        # We want an array of shape (11 * 6, 2)
        
        # Repeat base_verts for each center
        # base_verts: (6, 2) -> (11, 6, 2) by broadcasting?
        # Actually, just loop or use meshgrid logic.
        # For speed, let's use einsum or simple stacking.
        
        # Vertices for hex i = rotated_centers[i] + base_verts
        # We can do:
        # c_exp = rotated_centers[:, np.newaxis, :] # (11, 1, 2)
        # v_exp = base_verts[np.newaxis, :, :] # (1, 6, 2)
        # verts = (c_exp + v_exp).reshape(-1, 2)
        
        c_exp = rotated_centers[:, np.newaxis, :]
        v_exp = base_verts[np.newaxis, :, :]
        verts = (c_exp + v_exp).reshape(-1, 2)
        
        return verts

    def objective(params):
        tx, ty, theta_cluster = params
        verts = get_cluster_verts(tx, ty, theta_cluster)
        # Outer hexagon rotation is fixed to 0 for simplicity in this rigid body approach?
        # Or we can optimize it too? 
        # The problem allows rotating the outer hexagon.
        # However, due to symmetry, rotating the cluster relative to the container is equivalent to rotating the container relative to the cluster.
        # So optimizing theta_cluster with fixed outer_theta=0 is sufficient.
        S = calc_S_rigid(verts, 0.0)
        return S

    # Define bounds for differential evolution
    # The cluster size is roughly 4*sqrt(3) ~ 7 in diameter.
    # Translation bounds: +/- 5 should be enough.
    # Rotation bounds: 0 to pi/3 (60 deg) due to hexagonal symmetry.
    bounds = [(-5, 5), (-5, 5), (0, math.pi / 3)]
    
    best_S = float('inf')
    best_params = None
    
    # Run Differential Evolution
    # maxiter should be low to ensure speed, or use timeout.
    # DE is stochastic.
    
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=50, 
            popsize=15,
            tol=1e-4,
            seed=42,
            polish=True
        )
        
        if res.success or res.fun < best_S:
            best_S = res.fun
            best_params = res.x
    except Exception as e:
        print(f"DE failed: {e}")
        
    # If DE failed or didn't find anything, try a few random starts with Nelder-Mead for polishing
    if best_params is None or best_S == float('inf'):
        # Fallback to initial guess
        best_params = [0.0, 0.0, 0.0]
        best_S = objective(best_params)
        
    # Polish with Nelder-Mead
    if time.time() < deadline - safety_margin:
        try:
            res_nm = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            if res_nm.fun < best_S:
                best_S = res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    # Extract final parameters
    tx, ty, theta_cluster = best_params
    
    # Construct the output
    # The outer hexagon is centered at (0,0) with side length best_S and rotation 0.
    # The inner hexagons are defined by their centers and rotation.
    # Wait, the output format requires:
    # inner rows (x, y, angle_degrees) of n unit hexagons
    # outer hexagon center (x, y), side length and rotation
    
    # We need to output the centers of the 11 hexagons.
    # The centers in global coords are:
    centers_global = get_cluster_verts(tx, ty, theta_cluster)
    # Wait, get_cluster_verts returns vertices. We need centers.
    # Let's re-calculate centers.
    
    centers_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    centers_cart = []
    for q, r in centers_axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        centers_cart.append([x, y])
    centers_cart = np.array(centers_cart)
    
    rotated_centers = rotate_points(centers_cart, theta_cluster)
    rotated_centers = rotated_centers + np.array([tx, ty])
    
    inner_rows = []
    # All inner hexagons have the same rotation as the cluster
    angle_deg = math.degrees(theta_cluster)
    for cx, cy in rotated_centers:
        inner_rows.append([cx, cy, angle_deg])
        
    center = (0.0, 0.0)
    side = best_S
    angle_deg_outer = 0.0
    
    return inner_rows, center, side, angle_deg_outer
# EVOLVE-BLOCK-END
