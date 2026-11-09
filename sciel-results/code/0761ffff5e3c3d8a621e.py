# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    timeout = 600.0
    safety_margin = 30.0
    
    def angles_to_points(angles):
        # angles is shape (n, 2): theta (polar, 0 to pi), phi (azimuth, 0 to 2*pi)
        theta = angles[:, 0]
        phi = angles[:, 1]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack((x, y, z))

    def raw_score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(angles):
        points = angles_to_points(angles)
        return -raw_score(points)

    def fibonacci_sphere(n):
        """Generate n points on a sphere using Fibonacci lattice."""
        points = []
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        for i in range(n):
            # i+0.5 to avoid poles exactly
            theta = np.arccos(1 - 2 * (i + 0.5) / n)
            phi_angle = 2 * np.pi * i / phi
            points.append([theta, phi_angle])
        return np.array(points)

    def icosahedron_angles():
        """
        Generate vertices of a regular icosahedron (12 points) plus 2 more
        (e.g., north and south poles) to make 14 points.
        This is a very strong structured starting point.
        """
        t = (1 + np.sqrt(5)) / 2
        # Standard icosahedron vertices
        verts = [
            [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
            [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
            [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
        ]
        verts = np.array(verts, dtype=float)
        # Normalize to unit sphere
        norms = np.linalg.norm(verts, axis=1)
        verts = verts / norms[:, np.newaxis]
        
        # Add North and South Poles
        north = np.array([0, 0, 1])
        south = np.array([0, 0, -1])
        
        all_verts = np.vstack([verts, north, south])
        
        # Convert to angles
        angles = np.zeros((14, 2))
        for i, v in enumerate(all_verts):
            z = v[2]
            r = np.sqrt(v[0]**2 + v[1]**2)
            if r < 1e-10:
                theta = 0.0 if z > 0 else np.pi
                phi = 0.0
            else:
                theta = np.arccos(np.clip(z, -1.0, 1.0))
                phi = np.arctan2(v[1], v[0])
                if phi < 0:
                    phi += 2 * np.pi
            angles[i, 0] = theta
            angles[i, 1] = phi
        return angles

    best_score = -np.inf
    best_points = None
    best_angles = None

    def update_best(angles, score):
        nonlocal best_score, best_points, best_angles
        if score > best_score:
            best_score = score
            best_angles = angles.copy()
            best_points = angles_to_points(angles)

    # 1. Global Optimization with Differential Evolution
    # We restrict the search to the sphere surface using angles.
    # To handle symmetries and reduce the search space effectively,
    # we can fix the first point at the North Pole (theta=0, phi=0).
    # This reduces the dimensionality from 28 to 26.
    # However, DE handles high dimensions okay if popsize is large enough,
    # but 26 is fine.
    
    # Let's try DE on the full 28D space first, it's robust.
    # Bounds: theta in [0, pi], phi in [0, 2*pi]
    bounds = [(0, np.pi)] * n + [(0, 2 * np.pi)] * n
    
    # DE is slow, so we limit it.
    # We will run DE with a moderate population size and maxiter.
    # Then we will polish with L-BFGS-B or Nelder-Mead.
    
    try:
        # DE requires a flat array
        def de_objective(flat_angles):
            angles = flat_angles.reshape(n, 2)
            return objective(angles)
        
        # DE parameters
        # popsize=10 means 10*28 = 280 function evaluations per generation
        # maxiter=50 means 14000 evaluations.
        # This might be slow. Let's reduce.
        # popsize=5, maxiter=30 -> 5*28*30 = 4200 evaluations.
        
        # Actually, let's use a smaller popsize and rely on polishing.
        de_res = differential_evolution(
            de_objective, 
            bounds, 
            seed=42, 
            maxiter=30, 
            popsize=5, 
            tol=0.01, 
            polish=False, # We will do our own polish
            init='sobol'
        )
        
        if time.time() - start_time < timeout - safety_margin:
            de_angles = de_res.x.reshape(n, 2)
            score = -de_res.fun
            update_best(de_angles, score)
            
    except Exception:
        pass

    # 2. Multi-start Local Optimization (L-BFGS-B and Nelder-Mead)
    # We will start from several diverse points:
    # - Fibonacci sphere
    # - Icosahedron + Poles
    # - DE result (if available)
    # - Random restarts
    
    start_points = []
    
    # Structured starts
    start_points.append(fibonacci_sphere(n))
    start_points.append(icosahedron_angles())
    
    if best_angles is not None:
        start_points.append(best_angles)
    
    # Random restarts
    num_random_starts = 20
    for i in range(num_random_starts):
        if time.time() - start_time > timeout - safety_margin:
            break
        np.random.seed(100 + i)
        angles = np.random.rand(n, 2)
        angles[:, 0] *= np.pi
        angles[:, 1] *= 2 * np.pi
        start_points.append(angles)
    
    # Polish each start point
    for init_angles in start_points:
        if time.time() - start_time > timeout - safety_margin:
            break
            
        # Try Nelder-Mead first (robust for non-smooth)
        try:
            res_nm = minimize(objective, init_angles, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            score_nm = -res_nm.fun
            update_best(res_nm.x, score_nm)
            
            # If NM improved, use its result as start for L-BFGS-B
            if score_nm > best_score - 1e-6:
                init_for_lbfgs = res_nm.x
            else:
                init_for_lbfgs = init_angles
                
            # Try L-BFGS-B (faster convergence if smooth enough)
            # L-BFGS-B needs bounds
            res_lbfgs = minimize(objective, init_for_lbfgs, method='L-BFGS-B',
                                 bounds=bounds,
                                 options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            score_lbfgs = -res_lbfgs.fun
            update_best(res_lbfgs.x, score_lbfgs)
            
        except Exception:
            continue

    # 3. Final Polish: If we have a best_angles, do one more very tight NM polish
    if best_angles is not None and time.time() - start_time < timeout - safety_margin:
        try:
            res_final = minimize(objective, best_angles, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            score_final = -res_final.fun
            update_best(res_final.x, score_final)
        except Exception:
            pass

    # Fallback
    if best_points is None:
        init_angles = fibonacci_sphere(n)
        best_points = angles_to_points(init_angles)

    return best_points

# EVOLVE-BLOCK-END
