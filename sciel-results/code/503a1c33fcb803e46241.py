import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _hex_lattice_init(spacing, offset_x=0.0, offset_y=0.0):
    """Generate points on a hexagonal lattice inside the unit circle."""
    points = []
    # Estimate range
    k = int(1.0 / spacing) + 3
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing + offset_x
            y = j * spacing * np.sqrt(3.0) / 2.0 + offset_y
            if x*x + y*y <= 1.0:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        # If not enough, reduce spacing and retry or just pad (should not happen with good spacing)
        return None
    # Select the N points closest to the origin to form a compact cluster
    radii = np.hypot(points[:, 0], points[:, 1])
    indices = np.argsort(radii)[:N]
    return points[indices]

def _calc_radius(points):
    """Calculate the exact packing radius for a given set of points."""
    # Boundary constraint
    boundary_r = 1.0 - np.hypot(points[:, 0], points[:, 1])
    min_boundary = np.min(boundary_r)
    
    # Pairwise constraint
    # Use broadcasting for efficiency, O(N^2) memory but N=50 is small
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def _objective_and_grad(params):
    """
    Objective: minimize sum of squared violations.
    Violations are:
    1. Boundary: if ||p_i|| + r > 1, violation = ||p_i|| + r - 1
    2. Pairwise: if ||p_i - p_j|| < 2r, violation = 2r - ||p_i - p_j||
    
    Here r is a variable we also optimize? No, the task says maximize r.
    Usually in these problems, we fix r and check feasibility, or we maximize r directly.
    Since we want to maximize r, we can treat r as a variable in the optimization.
    Let params = [x_1, y_1, ..., x_N, y_N, r].
    
    Objective: minimize (sum of violations)^2 + penalty for r being small?
    Actually, we want to MAXIMIZE r.
    So we minimize -r + penalty(violations).
    
    Let's use a soft constraint approach.
    f(params) = -r + lambda * sum(violations^2)
    
    If we use this, the optimizer will try to increase r while keeping violations small.
    """
    points = params[:2*N].reshape(-1, 2)
    r = params[2*N]
    
    # Boundary violations
    norms = np.hypot(points[:, 0], points[:, 1])
    boundary_viol = np.maximum(0.0, norms + r - 1.0)
    
    # Pairwise violations
    dists = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    # We want dists >= 2r. Violation is max(0, 2r - dists)
    pair_viol = np.maximum(0.0, 2.0*r - dists)
    
    # Sum of squared violations
    # Only non-zero elements contribute
    total_viol_sq = np.sum(boundary_viol**2) + 0.5 * np.sum(pair_viol**2) # 0.5 to avoid double counting
    
    # Objective: -r + penalty
    # We need to tune the penalty. If penalty is too low, r goes to infinity (or 1).
    # If penalty is too high, r stays small.
    # A common trick is to use a large constant or adaptive penalty.
    # Let's use a fixed large penalty for now, or just minimize violations and see what r we get.
    # Actually, if we minimize violations, the optimizer will find the best packing for a given r?
    # No, r is a variable.
    
    # Better: Minimize sum(violations^2) subject to r being maximized?
    # Multi-objective is hard.
    
    # Alternative: Minimize -r + M * sum(violations^2).
    # If M is large enough, violations will be driven to 0.
    # Then r will be maximized subject to feasibility.
    
    M = 100.0
    obj = -r + M * total_viol_sq
    
    # Gradient
    grad = np.zeros_like(params)
    
    # Gradient of -r is -1
    grad[2*N] = -1.0
    
    # Gradient of M * sum(violations^2)
    # d/dp_i (max(0, ||p_i|| + r - 1)^2)
    # If ||p_i|| + r > 1:
    #   d/dp_i (||p_i|| + r - 1)^2 = 2(||p_i|| + r - 1) * d/dp_i(||p_i||)
    #   d/dp_i(||p_i||) = p_i / ||p_i||
    
    active_b = boundary_viol > 1e-10
    if np.any(active_b):
        # d/dp_i = 2 * boundary_viol_i * (p_i / ||p_i||)
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        grad_b = 2.0 * M * boundary_viol[active_b, np.newaxis] * (points[active_b] / safe_norms[active_b, np.newaxis])
        grad[:2*N].reshape(-1, 2)[active_b] += grad_b
        
        # d/dr = 2 * boundary_viol_i * 1
        grad[2*N] += 2.0 * M * np.sum(boundary_viol[active_b])
    
    # Pairwise: d/dp_i sum_{j!=i} max(0, 2r - ||p_i - p_j||)^2
    # For a pair (i,j) with violation v = 2r - d_ij > 0:
    # d/dp_i (v^2) = 2v * d/dp_i(2r - d_ij) = 2v * (- (p_i - p_j) / d_ij)
    # d/dr (v^2) = 2v * 2 = 4v
    
    active_p = pair_viol > 1e-10
    if np.any(active_p):
        # We need to sum contributions for each i
        # pair_viol is (N, N)
        # For each i, sum over j where pair_viol[i,j] > 0
        # contrib to grad[i] = 2*M * sum_j (pair_viol[i,j] * -(p_i - p_j) / d_ij)
        # Note: pair_viol is symmetric, so we can just use the matrix.
        
        # d_ij is dists
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        unit_vecs = (points[:, np.newaxis, :] - points[np.newaxis, :, :]) / safe_dists[:, :, np.newaxis]
        
        # Weighted sum
        # weights = 2*M*pair_viol
        weights = 2.0 * M * pair_viol
        # grad_i = - sum_j weights[i,j] * unit_vecs[i,j]
        # Note: unit_vecs[i,j] = (p_i - p_j)/d_ij
        # So -unit_vecs[i,j] = (p_j - p_i)/d_ij
        # This pushes i towards j if they are too close?
        # Wait. If d_ij < 2r, they overlap. We want to push them apart.
        # Force on i should be in direction of p_i - p_j.
        # Let's re-verify the sign.
        # v = 2r - d. d decreases as p_i moves towards p_j.
        # We want to minimize v^2.
        # d/dp_i v = - (p_i - p_j) / d.
        # d/dp_i v^2 = 2v * (-(p_i - p_j)/d).
        # If v > 0, this is a vector in direction -(p_i - p_j) = p_j - p_i.
        # So the gradient points towards j.
        # We are MINIMIZING the objective.
        # So we move AGAINST the gradient.
        # Move in direction -(p_j - p_i) = p_i - p_j.
        # This pushes i AWAY from j. Correct.
        
        # So the gradient term added to grad is:
        # 2*M * pair_viol[i,j] * (-(p_i - p_j)/d_ij)
        # which is what we have.
        
        # Vectorized sum
        weighted_vecs = weights[:, :, np.newaxis] * unit_vecs
        pair_grad = -np.sum(weighted_vecs, axis=1) # sum over j
        grad[:2*N] = grad[:2*N].reshape(-1, 2) + pair_grad
        
        # d/dr
        # sum over i,j of 4*M*pair_viol[i,j]
        # Since pair_viol is symmetric, sum over all i,j is 2 * sum over i<j.
        # My objective had 0.5 * sum(pair_viol^2).
        # So the gradient wrt r is 0.5 * 4 * M * sum(pair_viol) = 2*M*sum(pair_viol).
        grad[2*N] += 2.0 * M * np.sum(pair_viol)
        
    return obj, grad

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    best_radius = 0.0
    best_points = None
    
    # Initializations
    init_strategies = []
    spacings = [0.25, 0.26, 0.24, 0.27]
    offsets = [(0,0), (0.01, 0.01), (-0.01, 0.01), (0.01, -0.01)]
    
    for s in spacings:
        for ox, oy in offsets:
            pts = _hex_lattice_init(s, ox, oy)
            if pts is not None and len(pts) == N:
                init_strategies.append(pts)
    
    if not init_strategies:
        pts = np.random.rand(N, 2) * 2 - 1
        pts = pts[np.hypot(pts[:,0], pts[:,1]) <= 1]
        if len(pts) < N:
             while len(pts) < N:
                 new_pts = np.random.rand(N, 2) * 2 - 1
                 pts = np.vstack([pts, new_pts])
        init_strategies.append(pts[:N])

    for init_pts in init_strategies:
        if time.time() - start_time > time_limit:
            break
            
        current_pts = init_pts.copy()
        
        # Estimate initial r
        r_est = _calc_radius(current_pts)
        
        # Params: [x1, y1, ..., xN, yN, r]
        params = np.zeros(2*N + 1)
        params[:2*N] = current_pts.flatten()
        params[2*N] = r_est
        
        try:
            # Optimize
            res = minimize(
                lambda p: _objective_and_grad(p)[0],
                params,
                jac=lambda p: _objective_and_grad(p)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.success or res.status in [0, 1, 2, 3, 4, 5, 6]:
                new_params = res.x
                new_pts = new_params[:2*N].reshape(-1, 2)
                new_r = new_params[2*N]
                
                # Validate
                # Check if points are inside
                if np.all(np.hypot(new_pts[:, 0], new_pts[:, 1]) + new_r <= 1.0 + 1e-6):
                    # Check pairwise
                    dists = np.linalg.norm(new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :], axis=2)
                    np.fill_diagonal(dists, np.inf)
                    if np.all(dists >= 2.0*new_r - 1e-6):
                        # Valid
                        actual_r = _calc_radius(new_pts)
                        if actual_r > best_radius:
                            best_radius = actual_r
                            best_points = new_pts.copy()
        except Exception:
            pass

    if best_points is None:
        pts = _hex_lattice_init(0.25)
        if pts is not None:
            best_points = pts
            best_radius = _calc_radius(pts)
        else:
            best_points = np.zeros((N, 2))
            best_radius = 0.0

    return best_points
# EVOLVE-BLOCK-END
