import numpy as np
import time
from scipy.optimize import minimize

N = 200
DEADLINE = 145.0  # Safety margin for 150s limit

def _compute_radius(C):
    """Compute the packing radius for centers C (N, 2)."""
    # Pairwise distances
    D = np.sqrt(np.sum((C[:, np.newaxis, :] - C[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(D, np.inf)
    min_pair_dist = np.min(D)
    
    # Boundary distance
    dx = np.minimum(0.5 - C[:, 0], C[:, 0] + 0.5)
    dy = np.minimum(0.5 - C[:, 1], C[:, 1] + 0.5)
    min_boundary_dist = np.min(np.minimum(dx, dy))
    
    # Check validity
    if np.any(np.abs(C) > 0.5 + 1e-9):
        return 0.0
    
    return min(min_pair_dist / 2.0, min_boundary_dist)

def _soft_min_objective_and_grad(C, beta=100.0):
    """
    Soft-min surrogate objective. We want to maximize r = min(min_pair_dist/2, min_boundary_dist).
    Equivalently, minimize -r.
    Using soft-min: r_soft = -1/beta * log(sum(exp(-beta * d_i)))
    where d_i are the candidate radii (pair distances/2 and boundary distances).
    
    Actually, let's define the objective as the negative of the soft-min radius.
    We want to minimize: f(C) = -soft_min_radius(C)
    
    Let distances be:
    d_ij = ||C_i - C_j|| / 2 for i < j
    d_i = min(0.5 - |C_ix|, 0.5 - |C_iy|) for each i
    
    r_soft = -1/beta * log( sum_{pairs} exp(-beta * d_ij) + sum_{points} exp(-beta * d_i) )
    
    Gradient:
    df/dC_k = -dr_soft/dC_k
    
    dr_soft/dC_k = (1/beta) * d/dC_k [ -log(S) ] where S = sum exp(-beta * d)
    = (1/beta) * (1/S) * dS/dC_k
    dS/dC_k = sum_j (-beta * exp(-beta * d_j) * dd_j/dC_k)
    
    So dr_soft/dC_k = - sum_j [ exp(-beta * d_j) / S * dd_j/dC_k ]
    df/dC_k = sum_j [ exp(-beta * d_j) / S * dd_j/dC_k ]
    
    This is a weighted sum of gradients of the individual distances, weighted by their softmax probability.
    """
    # Compute all relevant distances
    # Pairwise
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(D, np.inf)
    
    # We only need upper triangle for pairs to avoid double counting, but for gradient
    # symmetry it's easier to sum over all j != i and divide by 2? 
    # Let's stick to the definition: S = sum_{i<j} exp(-beta * D_ij/2) + sum_i exp(-beta * B_i)
    
    # Boundary distances
    dx = np.minimum(0.5 - C[:, 0], C[:, 0] + 0.5)
    dy = np.minimum(0.5 - C[:, 1], C[:, 1] + 0.5)
    B = np.minimum(dx, dy)
    
    # Clip for numerical stability
    D_half = D / 2.0
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # Compute weights
    # To avoid overflow/underflow, we can shift by min, but beta=100 is large.
    # exp(-100 * 0.036) = exp(-3.6) ~ 0.027
    # exp(-100 * 0.030) = exp(-3.0) ~ 0.05
    # It's manageable.
    
    exp_D = np.where(mask, np.exp(-beta * D_half), 0.0)
    exp_B = np.exp(-beta * B)
    
    S = np.sum(exp_D) + np.sum(exp_B)
    
    if S == 0:
        return 0.0, np.zeros_like(C)
    
    # Objective: -r_soft = (1/beta) * log(S)
    obj = (1.0 / beta) * np.log(S)
    
    # Gradient
    # d(obj)/dC_k = (1/beta) * (1/S) * dS/dC_k
    # dS/dC_k = sum_{j!=k} exp_D[k,j] * d(D_kj/2)/dC_k + exp_B[k] * dB_k/dC_k
    # Note: exp_D is symmetric.
    # d(D_kj/2)/dC_k = (C_k - C_j) / ||C_k - C_j||
    
    grad = np.zeros_like(C)
    
    # Pairwise part
    # For each k, sum over j
    # Term: exp_D[k,j] * (C_k - C_j) / D_kj
    # Vectorized:
    # diff is (N, N, 2)
    # D is (N, N)
    # We need to divide diff by D, mask diagonal.
    D_safe = np.where(D > 1e-12, D, 1.0)
    unit_vecs = diff / D_safe[:, :, np.newaxis]
    # Mask diagonal
    unit_vecs = np.where(mask[:, :, np.newaxis], unit_vecs, 0.0)
    
    # Weighted sum
    # grad_pair[k] = sum_j exp_D[k,j] * unit_vecs[k,j]
    grad_pair = np.sum(exp_D[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # Boundary part
    # dB_k/dC_k
    # B_k = min(0.5 - x, x + 0.5, 0.5 - y, y + 0.5)
    # If B_k = 0.5 - x, dB/dx = -1
    # If B_k = x + 0.5, dB/dx = 1
    # If B_k = 0.5 - y, dB/dy = -1
    # If B_k = y + 0.5, dB/dy = 1
    
    grad_bound = np.zeros_like(C)
    # Determine which wall is closest for each point
    # Walls: 0: left (x=-0.5), 1: right (x=0.5), 2: bottom (y=-0.5), 3: top (y=0.5)
    dist_left = C[:, 0] + 0.5
    dist_right = 0.5 - C[:, 0]
    dist_bottom = C[:, 1] + 0.5
    dist_top = 0.5 - C[:, 1]
    
    # Indices of min
    # We need gradients of the min function.
    # If multiple are equal, take average? Or just pick one? 
    # Subgradient. Let's pick the one with the smallest distance.
    # For stability, if ties, it doesn't matter much for gradient descent.
    
    # Let's compute gradient contributions
    # d(dist_left)/dx = 1, d(dist_right)/dx = -1
    # d(dist_bottom)/dy = 1, d(dist_top)/dy = -1
    
    # If dist_left is min: grad_x += 1
    # If dist_right is min: grad_x -= 1
    # If dist_bottom is min: grad_y += 1
    # If dist_top is min: grad_y -= 1
    
    # Using np.argmin on a (N, 4) array
    dists = np.stack([dist_left, dist_right, dist_bottom, dist_top], axis=1)
    min_idx = np.argmin(dists, axis=1)
    
    # Map index to gradient
    # 0: left -> (1, 0)
    # 1: right -> (-1, 0)
    # 2: bottom -> (0, 1)
    # 3: top -> (0, -1)
    
    for i in range(N):
        if min_idx[i] == 0:
            grad_bound[i, 0] += 1.0
        elif min_idx[i] == 1:
            grad_bound[i, 0] -= 1.0
        elif min_idx[i] == 2:
            grad_bound[i, 1] += 1.0
        elif min_idx[i] == 3:
            grad_bound[i, 1] -= 1.0
    
    grad_bound *= exp_B[:, np.newaxis]
    
    total_grad = (1.0 / (beta * S)) * (grad_pair + grad_bound)
    
    return obj, total_grad

def _generate_hex_lattice():
    """Generate a hexagonal lattice initialization for N points in [-0.5, 0.5]^2."""
    # Hex packing density is pi / (2 * sqrt(3)) ~ 0.9069
    # Area per circle ~ 1 / 200 = 0.005
    # Radius ~ sqrt(0.005 / pi) ~ 0.0398 (theoretical max for infinite plane)
    # Target radius is 0.0366.
    
    # Let's create a grid of hex centers.
    # Spacing s. Area per hexagon in hex lattice is s^2 * sqrt(3) / 2.
    # We want to fit 200 points.
    
    # Start with a rectangular grid of hex centers.
    # Row 0: x = 0, s, 2s, ...
    # Row 1: x = s/2, 3s/2, ...
    
    # Estimate s.
    # Total area 1. 200 points.
    # s ~ sqrt(2 * 1 / (200 * sqrt(3))) ~ sqrt(1 / (100 * 1.732)) ~ sqrt(0.00577) ~ 0.076
    
    s = 0.076
    
    points = []
    # Generate points in a bounding box slightly larger than [-0.5, 0.5]
    # Then filter
    
    # Number of rows
    # Height of hex row is s * sqrt(3) / 2 ~ 0.0658
    # 1 / 0.0658 ~ 15 rows
    
    y = -0.6
    row = 0
    while len(points) < N + 50 and y < 0.6:
        x_offset = 0 if row % 2 == 0 else s / 2
        x = -0.6 + x_offset
        while x < 0.6:
            points.append([x, y])
            x += s
        y += s * np.sqrt(3) / 2
        row += 1
    
    points = np.array(points)
    
    # Filter points inside [-0.5, 0.5] with some margin?
    # Actually, we want to start with points inside, but BFGS will push them.
    # It's better to start with points strictly inside.
    mask = (np.abs(points[:, 0]) <= 0.45) & (np.abs(points[:, 1]) <= 0.45)
    points = points[mask]
    
    # If we have more than N, take the first N? Or center them?
    # Let's take the ones closest to the center to ensure symmetry.
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:N]]
    
    # Shuffle to break symmetry?
    # np.random.shuffle(points)
    
    return points

def _generate_random_initial():
    """Random initialization."""
    return np.random.uniform(-0.45, 0.45, (N, 2))

def _run_optimization(C_init, start_time, deadline=DEADLINE):
    """Run BFGS optimization."""
    current_time = time.time()
    if current_time > deadline - 10:
        return C_init, _compute_radius(C_init)
    
    # We need to project points to stay inside the square during optimization?
    # BFGS doesn't handle constraints directly.
    # We can use a penalty method or just rely on the wall potential in the objective.
    # The soft-min objective includes boundary distances, so it penalizes going out.
    # However, if a point goes far out, exp(-beta * d) where d is negative (distance to boundary is negative? No, d is defined as min(0.5-|x|...))
    # If x > 0.5, 0.5 - x is negative. dist_right is negative.
    # exp(-beta * negative) = exp(positive) -> Overflow!
    
    # So we MUST ensure points stay inside or handle the overflow.
    # Option 1: Clip C in the objective and gradient calculation.
    # Option 2: Add a strong quadratic penalty for going out of bounds.
    
    # Let's modify the objective to include a hard penalty.
    # If any point is outside, return a large value.
    # This makes the function non-smooth at the boundary, but BFGS can handle it if we're careful.
    # Better: use a smooth penalty.
    
    # Let's redefine the boundary distance to be negative if outside?
    # No, let's just clip the input C to [-0.5, 0.5] before calculating distances?
    # No, that changes the gradient.
    
    # Let's add a penalty term: P = sum( max(0, |C_ik| - 0.5)^2 )
    # And add it to the objective.
    # Grad P = 2 * (|C_ik| - 0.5) * sign(C_ik) if |C_ik| > 0.5 else 0.
    
    def objective_and_grad(C):
        # Check time
        if time.time() > deadline - 5:
            # Return current best? No, just return a large value to stop.
            # But we need to return a value.
            pass
        
        # Penalty for going out of bounds
        # Use a soft penalty to keep it smooth
        # Penalty = sum( log(1 + exp(k * (|C_ik| - 0.5))) ) ?
        # Simpler: Quadratic penalty with large coefficient.
        # But if it goes far, it explodes.
        
        # Let's just clip C to [-0.5, 0.5] for the distance calculation, 
        # but add a penalty for the amount clipped?
        
        # Actually, the soft-min objective with boundary distances works IF distances are positive.
        # If C is outside, boundary distance is negative.
        # exp(-beta * negative) is huge.
        # This effectively creates a very steep repulsive wall.
        # So it might work naturally!
        # But overflow is a risk.
        # Let's clip the exponent argument.
        
        C_clip = np.clip(C, -0.5, 0.5)
        
        # Calculate objective and grad on C_clip
        obj, grad = _soft_min_objective_and_grad(C_clip, beta=100.0)
        
        # Add penalty for deviation from C_clip (i.e., going out of bounds)
        # If C != C_clip, there is a penalty.
        # Penalty = 1000 * sum((C - C_clip)^2)
        # Grad = 2000 * (C - C_clip)
        
        diff = C - C_clip
        penalty = 1000.0 * np.sum(diff**2)
        grad_penalty = 2000.0 * diff
        
        return obj + penalty, grad + grad_penalty
    
    # Run BFGS
    # Max iterations
    # We have ~140s.
    # Each iteration of BFGS is O(N^2) for gradient.
    # N=200, N^2=40000. Very fast.
    # We can do many iterations.
    
    try:
        res = minimize(
            objective_and_grad, 
            C_init, 
            jac=True, 
            method='BFGS', 
            options={'maxiter': 1000, 'gtol': 1e-8, 'disp': False}
        )
        
        # Ensure result is inside
        C_res = np.clip(res.x, -0.5, 0.5)
        
    except Exception as e:
        C_res = C_init
    
    return C_res, _compute_radius(C_res)

def construct():
    start_time = time.time()
    
    # Initializations
    C_hex = _generate_hex_lattice()
    best_C = C_hex
    best_score = _compute_radius(C_hex)
    
    # Run optimization on hex
    C_opt, score_opt = _run_optimization(C_hex, start_time)
    if score_opt > best_score:
        best_C = C_opt
        best_score = score_opt
    
    # Try random restarts if time permits
    np.random.seed(42)
    for i in range(10):
        if time.time() > DEADLINE - 15:
            break
        
        C_rand = _generate_random_initial()
        # Quick optimization with fewer iterations?
        # Or just run the same optimizer.
        
        # To save time, maybe use a smaller beta or fewer maxiter for random restarts?
        # Or just run it.
        
        C_opt, score_opt = _run_optimization(C_rand, start_time)
        if score_opt > best_score:
            best_C = C_opt
            best_score = score_opt
    
    return best_C

# EVOLVE-BLOCK-END
