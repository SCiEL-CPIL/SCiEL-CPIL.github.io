import numpy as np
from scipy.optimize import minimize
import time

N = 300

def _hex_lattice_init(n_points, rotation=0.0):
    """Generate a hexagonal lattice of points inside the unit circle."""
    # Estimate spacing s such that N points fit in area pi
    # Area of hex cell is s^2 * sqrt(3)/2. Total area ~ pi.
    # N * s^2 * sqrt(3)/2 ~ pi  => s ~ sqrt(2*pi / (N*sqrt(3)))
    s = np.sqrt(2 * np.pi / (n_points * np.sqrt(3)))
    # Expand range to ensure we cover the unit circle
    k = int(1.0 / s) + 4
    
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            # Standard hex lattice coordinates
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            
            # Apply rotation
            xr = x * np.cos(rotation) - y * np.sin(rotation)
            yr = x * np.sin(rotation) + y * np.cos(rotation)
            
            if np.hypot(xr, yr) <= 1.0:
                points.append([xr, yr])
                
    points = np.array(points)
    if len(points) < n_points:
        # Fallback if lattice is too sparse (should not happen with k+4)
        # Fill with random points
        while len(points) < n_points:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) <= 1.0:
                points = np.vstack([points, p])
    else:
        # Select the n_points closest to the origin to form a compact cluster
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)
        points = points[indices[:n_points]]
        
    # Small random perturbation to break symmetry
    points += np.random.normal(0, s * 0.01, size=points.shape)
    
    # Ensure inside unit circle
    norms = np.linalg.norm(points, axis=1)
    for i in range(len(points)):
        if norms[i] > 1.0:
            points[i] = points[i] * (1.0 / norms[i])
            
    return points

def _compute_true_score(points):
    """Compute the true packing radius."""
    # Pairwise distances
    D = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    np.fill_diagonal(L, np.inf)
    min_pair = np.min(L) / 2.0
    
    # Wall distances
    L_wall = np.linalg.norm(points, axis=1)
    min_wall = 1.0 - np.max(L_wall)
    
    return min(min_pair, min_wall)

def _soft_min_obj_and_grad(x, beta, penalty_weight=1000.0):
    """
    Compute the negative soft-min value with penalty for constraint violations.
    Objective: Maximize min(distances). Equivalent to Minimize -min(distances).
    Soft min: S = -1/beta * log(sum(exp(-beta * x_k)))
    Gradient of S w.r.t x_k is -P_k, where P_k is the softmax probability.
    """
    C = x.reshape(N, 2)
    
    # 1. Pairwise distances
    # To save memory and time, we can compute upper triangle only, 
    # but for N=300, full matrix is 90k elements, which is fine.
    D = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    L = np.linalg.norm(D, axis=2)
    
    # Wall distances
    L_wall = np.linalg.norm(C, axis=1)
    x_walls = 1.0 - L_wall
    
    # Get upper triangle indices for pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    n_pairs = len(i_idx)
    
    # Extract pair distances
    L_flat = L[i_idx, j_idx]
    
    # Handle zero distances for gradient stability
    eps = 1e-10
    L_safe = np.maximum(L_flat, eps)
    
    x_pairs_flat = L_safe / 2.0
    
    # Combine all x values: [pair_distances/2, wall_distances]
    x_all = np.concatenate([x_pairs_flat, x_walls])
    
    # Compute soft-min using log-sum-exp trick
    # S = -1/beta * log(sum(exp(-beta * x_k)))
    # To avoid overflow/underflow, subtract min
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)
    P = exp_terms / Z
    
    # Value: S = min_x - (1/beta) * log(Z)
    val = min_x - (1.0 / beta) * np.log(Z)
    
    # Penalty for points outside the unit circle
    # We want to keep points inside. If |C_i| > 1, penalty increases.
    outside = np.maximum(0.0, L_wall - 1.0)
    penalty = penalty_weight * np.sum(outside ** 2)
    
    # Gradient of S with respect to C
    grad = np.zeros_like(C)
    
    # Pair contributions
    P_pairs = P[:n_pairs]
    
    D_flat = D[i_idx, j_idx]
    # U = (C_i - C_j) / L
    U = D_flat / L_safe[:, np.newaxis]
    
    # x_pair = L/2
    # d(x_pair)/dC_i = 0.5 * (C_i - C_j) / L = 0.5 * U
    # dS/dC_i = sum_j P_ij * d(x_ij)/dC_i = sum_j P_ij * 0.5 * U_ij
    # Note: S is the soft min. We are minimizing -S.
    # So we need gradient of -S.
    # d(-S)/dC_i = - sum_j P_ij * 0.5 * U_ij
    
    W = P_pairs * 0.5
    
    # Accumulate gradients for i and j
    # For each pair (i, j), contribution to i is W * U, to j is -W * U
    np.add.at(grad, i_idx, W[:, np.newaxis] * U)
    np.add.at(grad, j_idx, -W[:, np.newaxis] * U)
    
    # Wall contributions
    P_walls = P[n_pairs:]
    
    L_wall_safe = np.maximum(L_wall, eps)
    # x_wall = 1 - ||C_i||
    # d(x_wall)/dC_i = -C_i / ||C_i||
    U_walls = -C / L_wall_safe[:, np.newaxis]
    
    # d(-S)/dC_i = - P_wall_i * d(x_wall_i)/dC_i = - P_wall_i * (-C_i/||C_i||) = P_wall_i * C_i/||C_i||
    # Wait, let's re-verify signs.
    # S = soft_min(x). dS/dx_k = -P_k.
    # We minimize f = -S. df/dx_k = P_k.
    # df/dC_i = sum_k P_k * dx_k/dC_i.
    
    # For pairs:
    # dx_ij/dC_i = 0.5 * (C_i - C_j)/L_ij = 0.5 * U_ij
    # df/dC_i += P_ij * 0.5 * U_ij
    # My code above: grad += W * U where W = P * 0.5. This is correct for df/dC_i.
    
    # For walls:
    # dx_wall_i/dC_i = -C_i / ||C_i||
    # df/dC_i += P_wall_i * (-C_i / ||C_i||)
    # My code above: grad += P_walls[:, np.newaxis] * U_walls where U_walls = -C/||C||.
    # So grad += P * (-C/||C||). This is correct.
    
    # Penalty gradient
    # Penalty = w * sum(outside^2) where outside = max(0, ||C_i|| - 1)
    # dPenalty/dC_i = 2*w*outside * C_i/||C_i||
    penalty_grad = 2.0 * penalty_weight * outside[:, np.newaxis] * C / L_wall_safe[:, np.newaxis]
    grad += penalty_grad
    
    # We want to minimize -S + penalty.
    # Gradient is grad_S_neg + penalty_grad.
    # val is S. We return -val + penalty.
    # Gradient is -grad_S + penalty_grad? No.
    # Let f(x) = -S(x) + Penalty(x).
    # grad f = -grad S + grad Penalty.
    # My 'grad' variable currently holds grad S (from chain rule derivation above? No, I derived df/dC where f=-S).
    # Let's re-read my derivation.
    # "We minimize f = -S. df/dx_k = P_k."
    # "df/dC_i = sum P_k dx_k/dC_i".
    # So the `grad` variable I computed is actually the gradient of f = -S.
    # So I just add penalty_grad to it.
    
    return -(val) + penalty, grad + penalty_grad

def _enforce_constraints(points, r_margin=0.0):
    """Ensure points are inside the unit circle with a small margin."""
    norms = np.linalg.norm(points, axis=1)
    for i in range(len(points)):
        if norms[i] > 1.0 - r_margin:
            if norms[i] > 1e-10:
                points[i] = points[i] * (1.0 - r_margin) / norms[i]
            else:
                points[i] = np.array([1.0 - r_margin, 0.0])
    return points

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    # Generate multiple initializations with different rotations
    inits = []
    for rot in [0.0, np.pi/12, np.pi/6, np.pi/3]:
        inits.append(_hex_lattice_init(N, rotation=rot))
    
    best_points = None
    best_score = -np.inf
    
    # Evaluate initializations
    for init_points in inits:
        init_points = _enforce_constraints(init_points)
        init_score = _compute_true_score(init_points)
        if init_score > best_score:
            best_score = init_score
            best_points = init_points.copy()
    
    if best_points is None:
        best_points = inits[0]
        best_score = _compute_true_score(best_points)
    
    current_points = best_points.copy()
    
    # Beta schedule: Start low, increase gradually
    # Start with beta=10, end with beta=10000
    # Number of steps depends on time
    # Let's do a fixed number of annealing steps, but check time
    
    beta = 10.0
    max_beta = 20000.0
    beta_factor = 1.5
    
    # Pre-compute indices for speed
    # (Already done inside _soft_min_obj_and_grad, but keeping it local might be faster if we move it out? 
    # For N=300, np.triu_indices is fast enough to call repeatedly or we can pass it.)
    # Let's stick to the function for clarity, it's optimized in C under the hood.
    
    while time.time() - start_time < time_limit - 10.0:
        def obj_and_grad(x):
            return _soft_min_obj_and_grad(x, beta, penalty_weight=1000.0)
        
        try:
            remaining = time_limit - (time.time() - start_time)
            # Limit iterations
            # L-BFGS-B is fast. 200-500 iterations per step is reasonable.
            maxiter = min(300, max(50, int(remaining * 2)))
            
            res = minimize(obj_and_grad, current_points.flatten(), method='L-BFGS-B', jac=True, 
                           options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 50})
            
            if res.x is not None and np.all(np.isfinite(res.x)):
                new_points = res.x.reshape(N, 2)
                new_points = _enforce_constraints(new_points)
                new_score = _compute_true_score(new_points)
                
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                
                current_points = new_points
        except Exception:
            pass
            
        beta *= beta_factor
        if beta > max_beta:
            break

    # Final high-precision polish
    # Use very high beta and tight tolerances
    for final_beta in [max_beta, max_beta * 2]:
        if time.time() - start_time > time_limit - 2.0:
            break
        
        def final_obj_and_grad(x, fb=final_beta):
            return _soft_min_obj_and_grad(x, fb, penalty_weight=10000.0)
        
        try:
            remaining = time_limit - (time.time() - start_time)
            maxiter = min(500, max(100, int(remaining * 5)))
            res = minimize(final_obj_and_grad, current_points.flatten(), method='L-BFGS-B', jac=True, 
                           options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 100})
            if res.x is not None and np.all(np.isfinite(res.x)):
                new_points = res.x.reshape(N, 2)
                new_points = _enforce_constraints(new_points)
                new_score = _compute_true_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                current_points = new_points
        except Exception:
            pass
            
    # Return the best points found
    return best_points
