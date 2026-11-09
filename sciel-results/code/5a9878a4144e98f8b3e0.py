"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    if points is None or len(points) == 0:
        return 0.0
    # Distance to boundary
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(dist_boundary)
    
    # Pairwise distances
    from scipy.spatial.distance import pdist
    d = pdist(points)
    min_dist = np.min(d)
    min_pair = min_dist / 2.0
    
    return min(min_boundary, min_pair)

def _hexagonal_lattice():
    """Generate a high-quality hexagonal lattice packing."""
    # Optimal hexagonal packing density is pi / (2 * 3**0.5) ~ 0.9069
    # For N=300 circles in unit disk:
    # Area of N circles = N * pi * r^2
    # Area of container = pi * 1^2
    # Density = N * r^2 / 1^2
    # r ~ sqrt(Density / N) = sqrt(0.9069 / 300) ~ 0.055
    # Let's try a few spacings around this theoretical max to find the best integer lattice fit.
    
    best_r = 0.0
    best_points = None
    
    # Spacing a relates to radius r by r = a/2 in ideal hex packing.
    # So a ~ 2 * 0.055 = 0.11
    for a in np.linspace(0.105, 0.115, 20):
        h = a * np.sqrt(3) / 2
        
        points = []
        # Generate enough points to fill the disk
        for j in range(-30, 31):
            for i in range(-30, 31):
                x = (i + 0.5 * (j % 2)) * a
                y = j * h
                # Ensure the circle fits inside the unit circle
                # Distance from origin + radius <= 1
                # sqrt(x^2 + y^2) + a/2 <= 1
                if x*x + y*y + a <= 1.0: # approximation for quick filter, exact check later
                    points.append([x, y])
        
        if len(points) < N:
            continue
            
        points = np.array(points)
        
        # Exact filter: ensure 1 - norm >= a/2
        norms = np.linalg.norm(points, axis=1)
        mask = (1.0 - norms) >= (a/2) - 1e-9
        points = points[mask]
        
        if len(points) < N:
            continue
        
        # Select N points closest to center to maximize symmetry and minimize boundary issues
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        selected = points[idx]
        
        r = _compute_radius(selected)
        if r > best_r:
            best_r = r
            best_points = selected.copy()
            
    return best_points, best_r

def _soft_min_objective(points, beta=50.0):
    """
    Differentiable approximation of min(d_ij/2, 1-d_i).
    We maximize this, so we minimize the negative.
    Using log-sum-exp trick for soft-min.
    """
    # Reshape points to (N, 2)
    pts = points.reshape(N, 2)
    
    # 1. Boundary constraints: 1 - ||p_i||
    dists = np.linalg.norm(pts, axis=1)
    boundary_terms = 1.0 - dists
    
    # 2. Pairwise constraints: ||p_i - p_j|| / 2
    # Compute all pairwise distances
    # To save memory/time, we can use pdist, but for gradient we need explicit matrix or careful handling.
    # Since N=300, 300x300 matrix is fine.
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, np.inf)
    pair_terms = dist_matrix / 2.0
    
    # Combine all terms into a single vector
    # We want min(all terms)
    # soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    # We want to maximize soft_min, so minimize -soft_min = 1/beta * log(sum(exp(-beta * x)))
    
    # Flatten terms
    all_terms = np.concatenate([boundary_terms, pair_terms.flatten()])
    
    # Scale for numerical stability
    min_term = np.min(all_terms)
    exp_terms = np.exp(-beta * (all_terms - min_term))
    
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = min_term - np.log(np.sum(exp_terms)) / beta
    
    return -soft_min_val, -1.0 # Gradient placeholder, we'll use numerical or analytic if easy. 
                                # Actually, scipy L-BFGS-B can use numerical gradients if not provided, 
                                # but for 600D it might be slow. Let's provide analytic gradient if possible 
                                # or just use numerical with tight tolerances.
                                # Given 150s budget, numerical gradient for L-BFGS-B on 600D 
                                # might take ~100-200 function evals. Each eval is O(N^2) ~ 90k ops.
                                # 200 * 90k = 18M ops. Should be fast enough in Python/NumPy.

def _optimize_gradient(points):
    """Optimize the packing using L-BFGS-B on the soft-min objective."""
    x0 = points.flatten()
    
    # Objective function wrapper
    def objective(x):
        val, _ = _soft_min_objective(x, beta=50.0)
        return val
        
    # Run optimization
    # maxiter needs to be high enough to converge but low enough to finish in time.
    # L-BFGS-B is efficient.
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if res.success:
        return res.x.reshape(N, 2)
    else:
        # Return the best point found during optimization
        # Note: L-BFGS-B doesn't easily expose the best intermediate point without callback.
        # We'll just return the final result which is usually good.
        return res.x.reshape(N, 2)

def construct():
    """Construct the optimal packing."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # 1. Generate initial hexagonal lattice
    points, init_r = _hexagonal_lattice()
    
    if points is None:
        # Fallback: simple grid
        points = np.zeros((N, 2))
        init_r = 0.0
        
    # 2. Gradient-based optimization
    # We will run L-BFGS-B. It uses numerical gradients by default.
    # For N=300, dim=600.
    # To improve speed, we can reduce the dimensionality by fixing the center of mass?
    # No, the container is fixed at origin, so we shouldn't fix COM unless we add a constraint.
    # L-BFGS-B handles box constraints, but we have no box constraints on x,y other than the objective.
    # However, points must stay inside the unit circle. The objective penalizes going outside.
    # It might drift outside if the penalty isn't strong enough.
    # The soft-min objective goes to -inf if any term goes to -inf?
    # If 1 - ||p|| becomes negative, exp(-beta * negative) = exp(positive) -> large.
    # log(sum) -> large. -soft_min -> large positive.
    # So the optimizer will try to keep points inside.
    
    # Let's add a small regularization or just rely on the soft-min.
    # To ensure we don't waste time if the initial guess is bad, we check time.
    
    if time.time() < deadline - 10:
        opt_points = _optimize_gradient(points)
        opt_r = _compute_radius(opt_points)
        
        if opt_r > init_r:
            points = opt_points
            init_r = opt_r
    
    # 3. If we have time left, try a second pass with higher beta (sharper min) 
    # or just return the best.
    # Given the 150s limit and L-BFGS-B speed, one pass is likely sufficient.
    # If we have lots of time, we could do a local search on the result.
    
    if time.time() < deadline - 5:
        # Quick local polish? No, L-BFGS-B should have converged.
        pass

    return points
# EVOLVE-BLOCK-END
