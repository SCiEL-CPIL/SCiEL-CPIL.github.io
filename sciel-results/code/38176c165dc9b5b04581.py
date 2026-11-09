# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import warnings

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Strategy:
    1. Assume the optimal configuration lies on a sphere (scale invariance allows fixing radius=1).
    2. Parameterize points using spherical coordinates (theta, phi).
    3. Use multi-start L-BFGS-B local optimization to maximize the minimum pairwise distance.
    4. Initialize with a geometric prior: Icosahedron vertices + 2 poles, plus random perturbations.
    5. Convert back to Cartesian coordinates and normalize.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s margin for safety

    # Objective function to minimize: -d_min
    # We fix points on the unit sphere.
    # x: array of shape (n, 2) containing [theta, phi]
    # theta: polar angle [0, pi]
    # phi: azimuthal angle [0, 2*pi]
    
    def spherical_to_cartesian(x):
        # x is (n, 2)
        theta = x[:, 0]
        phi = x[:, 1]
        # Clip to avoid numerical issues
        theta = np.clip(theta, 0, np.pi)
        x_cart = np.column_stack((
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ))
        return x_cart

    def objective_spherical(x_flat):
        if time.time() > deadline:
            return 1e10
        x = x_flat.reshape(n, 2)
        pts = spherical_to_cartesian(x)
        dists = pdist(pts)
        if np.any(np.isnan(dists)) or np.any(np.isinf(dists)):
            return 1e10
        d_min = np.min(dists)
        # We want to maximize d_min, so minimize -d_min
        return -d_min

    # Generate initial guesses
    def generate_initial_guesses(num_starts):
        guesses = []
        
        # 1. Geometric Prior: Icosahedron + 2 Poles
        phi_golden = (1 + np.sqrt(5)) / 2
        # Standard icosahedron vertices
        ico_verts = np.array([
            [0, 1, phi_golden], [0, -1, phi_golden],
            [0, 1, -phi_golden], [0, -1, -phi_golden],
            [1, phi_golden, 0], [-1, phi_golden, 0],
            [1, -phi_golden, 0], [-1, -phi_golden, 0],
            [phi_golden, 0, 1], [-phi_golden, 0, 1],
            [phi_golden, 0, -1], [-phi_golden, 0, -1]
        ])
        # Normalize to unit sphere
        ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1)[:, np.newaxis]
        
        # Add poles
        pole1 = np.array([0, 0, 1])
        pole2 = np.array([0, 0, -1])
        
        # Combine
        initial_pts = np.vstack([ico_verts, pole1, pole2])
        
        # Convert to spherical coordinates
        def cartesian_to_spherical(pts):
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arccos(np.clip(z / r, -1, 1))
            phi = np.arctan2(y, x)
            return np.column_stack((theta, phi))
        
        initial_sph = cartesian_to_spherical(initial_pts)
        guesses.append(initial_sph.flatten())
        
        # 2. Random Starts
        for i in range(num_starts - 1):
            theta = np.random.uniform(0, np.pi, n)
            phi = np.random.uniform(0, 2 * np.pi, n)
            guesses.append(np.column_stack((theta, phi)).flatten())
            
        return guesses

    best_x = None
    best_score = -np.inf
    best_cart = None

    # Run multi-start optimization
    num_starts = 20
    guesses = generate_initial_guesses(num_starts)
    
    bounds = [(0, np.pi), (0, 2 * np.pi)] * n
    
    for i, guess in enumerate(guesses):
        if time.time() > deadline - 60:
            break
            
        try:
            res = minimize(
                objective_spherical, 
                guess, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
                best_cart = spherical_to_cartesian(best_x.reshape(n, 2))
                
        except Exception as e:
            continue

    # If optimization failed or didn't run, use the geometric prior directly
    if best_cart is None:
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
        best_cart = np.vstack([ico_verts, pole1, pole2])

    # Refine the best Cartesian result slightly with a Cartesian-based Nelder-Mead
    # to ensure we are not stuck in a spherical coordinate singularity (poles)
    if time.time() < deadline - 20:
        def objective_cartesian(x_flat):
            pts = x_flat.reshape(n, d)
            # Re-normalize to unit sphere to maintain the constraint
            norms = np.linalg.norm(pts, axis=1)
            norms[norms < 1e-10] = 1.0
            pts = pts / norms[:, np.newaxis]
            
            dists = pdist(pts)
            if np.any(np.isnan(dists)) or np.any(np.isinf(dists)):
                return 1e10
            d_min = np.min(dists)
            return -d_min

        try:
            res_nm = minimize(
                objective_cartesian, 
                best_cart.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            # Check if NM improved the score
            nm_score = res_nm.fun
            if nm_score < best_score:
                best_score = nm_score
                # Normalize the result
                pts = res_nm.x.reshape(n, d)
                norms = np.linalg.norm(pts, axis=1)
                norms[norms < 1e-10] = 1.0
                best_cart = pts / norms[:, np.newaxis]
        except Exception:
            pass

    # Final processing
    points = best_cart
    # Center of mass to 0 (should already be close, but ensure)
    points -= np.mean(points, axis=0)
    
    # Scale to unit max distance (optional, but keeps numbers nice and consistent)
    dists = pdist(points)
    d_max = np.max(dists)
    if d_max > 1e-12:
        points = points / d_max

    # Ensure finite
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
