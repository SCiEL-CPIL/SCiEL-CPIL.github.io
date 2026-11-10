# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14, 3) containing the coordinates.
    """
    n = 14
    d = 3
    start_time = time.time()
    timeout = 580  # Safety margin for 600s limit

    # Soft-min and soft-max parameters
    # Higher k makes it closer to true min/max
    k = 20.0

    def soft_min_max_ratio(points):
        """
        Computes a smooth approximation of the min/max distance ratio.
        We want to MAXIMIZE min/max, so we MINIMIZE (soft_max - soft_min).
        """
        if not np.all(np.isfinite(points)):
            return 1e10
        dists = pdist(points)
        if len(dists) == 0:
            return 1e10
        
        # Soft max: (1/k) * log(sum(exp(k * d)))
        # Soft min: -(1/k) * log(sum(exp(-k * d)))
        # To avoid overflow, subtract max from d for exp
        d_max_val = np.max(dists)
        d_min_val = np.min(dists)
        
        # Stable soft max
        exp_d = np.exp(k * (dists - d_max_val))
        soft_max = d_max_val + (1/k) * np.log(np.sum(exp_d))
        
        # Stable soft min
        exp_neg_d = np.exp(-k * (dists - d_min_val))
        soft_min = d_min_val - (1/k) * np.log(np.sum(exp_neg_d))
        
        # Objective to minimize: soft_max - soft_min
        return soft_max - soft_min

    def objective_flat(x):
        points = x.reshape(n, d)
        return soft_min_max_ratio(points)

    def actual_score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    # Initial guesses
    initial_guesses = []

    # 1. Icosahedron vertices + 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Add 2 points at the "poles" of the icosahedron (axis of 5-fold symmetry)
    # The icosahedron has 6 axes of 5-fold symmetry. 
    # Let's just add 2 points at random positions on the sphere for now, 
    # or better, at the center of the faces? 
    # Actually, for 14 points, a known good config is Icosahedron + 2 points 
    # placed to optimize. Let's start with 2 points at [0,0,1] and [0,0,-1] 
    # after rotating the icosahedron so its symmetry axis aligns with Z?
    # The icosahedron vertices above are not aligned with Z-axis 5-fold symmetry easily.
    # Let's just use the 12 verts + 2 random points on sphere.
    
    ico_points = np.vstack([ico_verts, np.array([[0, 0, 1], [0, 0, -1]])])
    # Normalize to unit sphere to start
    ico_points = ico_points / np.linalg.norm(ico_points, axis=1, keepdims=True)
    initial_guesses.append(ico_points)

    # 2. Cuboctahedron vertices + 2 points
    # Cuboctahedron has 12 vertices: permutations of (0, 1, 1)
    cubo_verts = []
    for perm in np.permutations([0, 1, 1]):
        for signs in [[1,1,1], [1,1,-1], [1,-1,1], [-1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1], [1,-1,1]]: # This is messy, let's generate properly
            pass
    
    # Proper Cuboctahedron generation
    verts = []
    for x in [0, 1, -1]:
        for y in [0, 1, -1]:
            for z in [0, 1, -1]:
                if abs(x) + abs(y) + abs(z) == 2:
                    verts.append([x, y, z])
    cubo_verts = np.array(verts)
    cubo_verts = cubo_verts / np.linalg.norm(cubo_verts, axis=1, keepdims=True)
    
    # Add 2 points
    # The cuboctahedron has 8 faces (triangles) and 6 faces (squares).
    # Let's add points at the centers of opposite square faces? (0,0,1) and (0,0,-1)
    cubo_points = np.vstack([cubo_verts, np.array([[0, 0, 1], [0, 0, -1]])])
    cubo_points = cubo_points / np.linalg.norm(cubo_points, axis=1, keepdims=True)
    initial_guesses.append(cubo_points)

    # 3. Random points on sphere
    np.random.seed(42)
    rand_points = np.random.randn(n, d)
    rand_points = rand_points / np.linalg.norm(rand_points, axis=1, keepdims=True)
    initial_guesses.append(rand_points)

    best_points = None
    best_score = -1.0

    # Optimization Loop
    for i, init in enumerate(initial_guesses):
        if time.time() - start_time > timeout:
            break
            
        x0 = init.flatten()
        
        # Use L-BFGS-B as it is efficient for smooth problems
        # We use the soft objective for the gradient-free/finite-diff steps
        try:
            res = minimize(objective_flat, x0, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            opt_points = res.x.reshape(n, d)
            
            # Scale the points to have max distance 1? 
            # The objective is scale-invariant in the ratio, but the soft approx 
            # might be affected by scale. 
            # Actually, soft_max - soft_min is NOT scale invariant.
            # If we scale points by S, dists scale by S.
            # SoftMax(S*d) = S*SoftMax(d). SoftMin(S*d) = S*SoftMin(d).
            # Difference scales by S.
            # So the optimizer might just shrink the points to 0 to minimize the difference!
            # We MUST constrain the scale.
            # Let's normalize the points to unit sphere or fix one distance.
            
            # Better: Normalize the result to have max distance 1, then compute score.
            # But the optimization itself needs a constraint or a scale-invariant objective.
            
            # Let's redefine the objective to be scale-invariant.
            # Minimize (SoftMax - SoftMin) / SoftMax ?
            # Or just normalize inside the objective?
            # If we normalize inside, the gradient becomes tricky.
            
            # Alternative: Fix the scale by constraining the points to lie on a sphere of radius R?
            # No, optimal packing is not on a sphere.
            
            # Let's use the ratio directly in a smooth way.
            # Objective: - log( SoftMin / SoftMax ) = log(SoftMax) - log(SoftMin).
            # This is scale invariant!
            
            def objective_scale_inv_flat(x):
                points = x.reshape(n, d)
                if not np.all(np.isfinite(points)):
                    return 1e10
                dists = pdist(points)
                if len(dists) == 0:
                    return 1e10
                d_max_val = np.max(dists)
                d_min_val = np.min(dists)
                if d_max_val < 1e-10:
                    return 1e10
                
                exp_d = np.exp(k * (dists - d_max_val))
                soft_max = d_max_val + (1/k) * np.log(np.sum(exp_d))
                
                exp_neg_d = np.exp(-k * (dists - d_min_val))
                soft_min = d_min_val - (1/k) * np.log(np.sum(exp_neg_d))
                
                if soft_min <= 0 or soft_max <= 0:
                    return 1e10
                    
                return np.log(soft_max) - np.log(soft_min)

            # Re-run with scale-invariant objective
            res = minimize(objective_scale_inv_flat, x0, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            opt_points = res.x.reshape(n, d)
            
            score = actual_score(opt_points)
            
            if score > best_score:
                best_score = score
                best_points = opt_points
                
        except Exception as e:
            continue

    # If no optimization succeeded, return the best initial guess
    if best_points is None:
        best_points = initial_guesses[0]

    # Final check and cleanup
    if not np.all(np.isfinite(best_points)):
        best_points = initial_guesses[0]

    return best_points

# EVOLVE-BLOCK-END
