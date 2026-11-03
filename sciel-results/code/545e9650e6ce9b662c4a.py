import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50

def _compute_radius(points):
    """Calculate the valid radius for a given set of points."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Check if inside container
    dists_to_center = np.linalg.norm(points, axis=1)
    min_dist_to_boundary = 1.0 - np.max(dists_to_center)
    
    if min_dist_to_boundary < 0:
        return 0.0

    # Check pairwise distances
    # Use a vectorized approach for efficiency
    # dist_matrix[i,j] = |p_i - p_j|
    # We only need the min off-diagonal
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_dist_to_boundary)
    
    # Validity check: no coincident centers (dist > 0)
    if min_pair_dist < 1e-9:
        return 0.0
        
    return r

def _generate_points_from_rings(radii, angles, counts):
    """
    Generate points from concentric rings.
    radii: list of ring radii
    angles: list of angular offsets for each ring
    counts: list of number of circles in each ring
    """
    points = []
    # Central circle
    points.append([0.0, 0.0])
    
    for k, (r, theta, n) in enumerate(zip(radii, angles, counts)):
        for i in range(n):
            angle = theta + 2 * np.pi * i / n
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            points.append([x, y])
            
    return np.array(points)

def _objective_ring(params, counts):
    """
    Objective function for ring-based optimization.
    Minimizes negative radius.
    """
    n_rings = len(counts) - 1 # Exclude center
    radii = params[:n_rings]
    angles = params[n_rings:2*n_rings]
    
    # Ensure radii are positive and ordered
    # Simple penalty if not ordered
    if any(r < 0 for r in radii):
        return 1.0
    if any(radii[i] > radii[i+1] for i in range(len(radii)-1)):
        return 1.0
        
    points = _generate_points_from_rings(radii, angles, counts)
    r = _compute_radius(points)
    return -r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # 1. Define ring structure
    # Based on optimal packings, a 4-ring structure (plus center) is common for N=50
    # Counts: 1 (center) + 6 + 12 + 18 + 13 = 50
    counts = [6, 12, 18, 13]
    n_rings = len(counts)
    
    # Initial guess for radii and angles
    # Radii should be roughly proportional to sqrt(k) or linear?
    # For hexagonal packing, ring k radius is approx k * 2r.
    # Let's start with a heuristic
    r_guess = 0.125
    init_radii = [r_guess * (i + 1) * 1.8 for i in range(n_rings)]
    init_angles = [0.0] * n_rings
    
    init_params = np.concatenate([init_radii, init_angles])
    
    best_score = 0.0
    best_points = None
    
    # 2. Global Search: Differential Evolution on 8D space
    # Bounds for radii: [0.1, 0.8]
    # Bounds for angles: [0, 2*pi]
    bounds = []
    for i in range(n_rings):
        bounds.append((0.05, 0.8))
    for i in range(n_rings):
        bounds.append((0.0, 2 * np.pi))
        
    try:
        # DE is robust for global search
        # maxiter=100, popsize=15 -> 1500 evaluations. Should be fast for 8D.
        res = differential_evolution(
            _objective_ring, 
            bounds, 
            args=(counts,),
            maxiter=100, 
            popsize=20,
            tol=1e-6,
            seed=42,
            init='sobol'
        )
        if time.time() > deadline:
            # Fallback to initial guess if DE took too long (unlikely for 8D)
            points = _generate_points_from_rings(init_radii, init_angles, counts)
        else:
            best_ring_params = res.x
            points = _generate_points_from_rings(best_ring_params[:n_rings], best_ring_params[n_rings:], counts)
            
        current_score = _compute_radius(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
    except Exception as e:
        print(f"DE failed: {e}")
        points = _generate_points_from_rings(init_radii, init_angles, counts)
        current_score = _compute_radius(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()

    # 3. Local Polish: L-BFGS-B on 100D space
    if time.time() < deadline - 10.0:
        # Use a smooth surrogate for the min function
        # Maximize r is equivalent to minimizing -r
        # Smooth approx: -1/beta * log(sum(exp(-beta * dist/2)) + sum(exp(-beta * (1-|c|))))
        # We will use a sequence of beta values
        
        def smooth_objective(params, beta):
            pts = params.reshape(-1, 2)
            # Check validity
            if not np.all(np.isfinite(pts)):
                return 1e10
            
            dists_to_center = np.linalg.norm(pts, axis=1)
            # Term for container boundary
            # We want 1 - |c| to be large, so exp(-beta * (1 - |c|)) small
            # If 1 - |c| < 0, term is large
            boundary_term = np.sum(np.exp(-beta * (1.0 - dists_to_center)))
            
            # Pairwise distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Term for pairwise
            # We want dist/2 to be large
            pairwise_term = np.sum(np.exp(-beta * (dists / 2.0)))
            
            # Objective to minimize
            obj = -(1.0 / beta) * np.log(boundary_term + pairwise_term + 1e-10)
            
            # Add a large penalty if any circle is outside or overlapping significantly
            # This helps guide the optimizer back to feasibility
            min_boundary = np.min(1.0 - dists_to_center)
            min_pair = np.min(dists) / 2.0
            
            if min_boundary < 0 or min_pair < 0:
                return obj + 100.0 * (abs(min_boundary) + abs(min_pair))
                
            return obj
        
        # Start with the best points from ring search
        x0 = best_points.flatten()
        
        # Sequence of betas
        betas = [10, 50, 200, 1000]
        
        current_x = x0.copy()
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            try:
                res_local = minimize(
                    smooth_objective, 
                    current_x, 
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res_local.success or res_local.fun < smooth_objective(current_x, beta):
                    new_x = res_local.x
                    new_pts = new_x.reshape(-1, 2)
                    new_score = _compute_radius(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_pts.copy()
                        current_x = new_x
            except Exception as e:
                print(f"Local opt failed: {e}")
                break

    # 4. Final Check and Return
    if best_points is None:
        # Fallback: Hexagonal lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_points = points[np.argsort(radius, kind="stable")[:N]]
                break
            spacing *= 0.98

    # Ensure shape is (N, 2)
    return best_points[:N]
