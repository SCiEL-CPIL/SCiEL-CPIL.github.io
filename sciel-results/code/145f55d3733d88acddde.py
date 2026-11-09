"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Distance to walls
    min_dist_to_wall = np.min(0.5 - np.abs(centers), axis=(0, 1))

    # Minimum pairwise distance / 2
    # Use a tree for efficiency, though for N=50 a full distance matrix is also fine
    # Let's use the full matrix for simplicity and robustness in the objective, 
    # but for radius calculation, KDTree is good.
    tree = cKDTree(centers)
    # Query all pairs within a reasonable distance to find min
    # Since we want the global min, we can just compute all pairwise distances for N=50
    # N=50 -> 1225 pairs. Very fast.
    dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)

    return min(min_dist_to_wall, min_pair_dist / 2.0)


def _halton_sequence(n, base=2):
    """Generate a Halton sequence in [0, 1)."""
    # Simple implementation for 3D
    # This is a basic Halton generator. For better performance, 
    # a precomputed or more efficient implementation could be used,
    # but N=50 is small.
    
    def halton_1d(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    points = []
    for i in range(1, n + 1):
        x = halton_1d(i, 2)
        y = halton_1d(i, 3)
        z = halton_1d(i, 5)
        points.append([x, y, z])
    return np.array(points)


def _qp_objective(x):
    """
    Objective to minimize: Negative sum of squared pairwise distances 
    and squared wall clearances.
    This is a quadratic function of the coordinates, making it smooth 
    and suitable for L-BFGS-B.
    """
    centers = x.reshape(N, 3)
    
    # Wall clearances: (0.5 - |c|)^2
    # We want to maximize the minimum, but sum of squares is a proxy for spreading.
    # Actually, maximizing sum of squared wall clearances pushes points away from walls.
    # Maximizing sum of squared pairwise distances pushes points apart.
    
    # Sum of squared pairwise distances
    # D = ||c_i - c_j||^2 = sum_k (c_ik - c_jk)^2
    # Sum_{i<j} D_{ij}
    # This can be computed efficiently.
    # Sum_{i<j} ||c_i - c_j||^2 = N * sum_i ||c_i||^2 - ||sum_i c_i||^2
    # Let's verify:
    # Sum_{i,j} ||c_i - c_j||^2 = Sum_{i,j} (||c_i||^2 + ||c_j||^2 - 2 c_i.c_j)
    # = Sum_i Sum_j ||c_i||^2 + Sum_i Sum_j ||c_j||^2 - 2 Sum_i Sum_j c_i.c_j
    # = N Sum_i ||c_i||^2 + N Sum_j ||c_j||^2 - 2 ||Sum_i c_i||^2
    # = 2N Sum_i ||c_i||^2 - 2 ||Sum_i c_i||^2
    # Since we want i<j, we divide by 2.
    # Sum_{i<j} ||c_i - c_j||^2 = N * Sum_i ||c_i||^2 - ||Sum_i c_i||^2
    
    sq_norms = np.sum(centers**2, axis=1)
    sum_norm_sq = np.sum(sq_norms)
    sum_centers = np.sum(centers, axis=0)
    sq_sum_norm = np.sum(sum_centers**2)
    
    pair_sum_sq = N * sum_norm_sq - sq_sum_norm
    
    # Wall clearances
    # We want points to be away from walls. 
    # Wall distance is 0.5 - |c_k|.
    # Let's maximize sum of squared wall distances?
    # Or maybe just use a penalty for being close to walls?
    # The QP formulation usually maximizes sum of squared distances.
    # Let's add sum of squared wall clearances.
    wall_clearances = 0.5 - np.abs(centers)
    wall_sum_sq = np.sum(wall_clearances**2)
    
    # Total objective to maximize: pair_sum_sq + lambda * wall_sum_sq
    # We minimize the negative.
    # We need to balance the scales. 
    # Pair distances are in [0, sqrt(3)]. Squared: [0, 3]. Sum ~ 1225 * 1.5 ~ 1800.
    # Wall clearances are in [0, 0.5]. Squared: [0, 0.25]. Sum ~ 150 * 0.1 ~ 15.
    # The pair term dominates. This is good, as packing is mostly about pair distances.
    # But we need to ensure points stay inside. L-BFGS-B will handle bounds.
    
    return -(pair_sum_sq + 10.0 * wall_sum_sq)


def _qp_gradient(x):
    """
    Gradient of the objective function for L-BFGS-B.
    f(x) = - (N * Sum_i ||c_i||^2 - ||Sum_i c_i||^2 + 10 * Sum_k (0.5 - |c_k|)^2)
    
    Let S = Sum_i c_i.
    Pair term P = N * Sum_i ||c_i||^2 - ||S||^2.
    dP/dx_i = 2 * N * c_i - 2 * S.
    
    Wall term W = 10 * Sum_{i,k} (0.5 - |c_{ik}|)^2.
    dW/dx_{ik} = 10 * 2 * (0.5 - |c_{ik}|) * d(0.5 - |c_{ik}|)/dx_{ik}
    d(0.5 - |c_{ik}|)/dx_{ik} = -sign(c_{ik}) if c_{ik} != 0.
    So dW/dx_{ik} = -20 * (0.5 - |c_{ik}|) * sign(c_{ik}).
    
    Total Gradient G_i = - (2 * N * c_i - 2 * S - 20 * (0.5 - |c_i|) * sign(c_i))
    G_i = 2 * S - 2 * N * c_i + 20 * (0.5 - |c_i|) * sign(c_i)
    """
    centers = x.reshape(N, 3)
    S = np.sum(centers, axis=0)
    
    grad_centers = np.zeros_like(centers)
    
    # Pair term gradient: 2 * N * c_i - 2 * S
    grad_pair = 2 * N * centers - 2 * S
    
    # Wall term gradient: -20 * (0.5 - |c_i|) * sign(c_i)
    # Note: sign(0) is 0 in numpy, which is fine (subgradient).
    signs = np.sign(centers)
    wall_clearances = 0.5 - np.abs(centers)
    grad_wall = -20.0 * wall_clearances * signs
    
    # Total gradient of the function to minimize (which is -Objective)
    # Objective = P + 10*W
    # d(Obj)/dx = grad_pair + grad_wall
    # d(-Obj)/dx = -grad_pair - grad_wall
    
    grad_total = -(grad_pair + grad_wall)
    
    return grad_total.flatten()


def construct():
    """Construct 50 sphere centers in unit cube to maximize minimum radius."""
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Initialize with Halton sequence mapped to [-0.5, 0.5]
    # Halton points are in [0, 1). Map to [-0.5, 0.5].
    halton_pts = _halton_sequence(N)
    centers = halton_pts - 0.5
    
    # Ensure they are strictly inside
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # Define bounds for L-BFGS-B
    # Spheres must be inside [-0.5, 0.5]. 
    # Since we don't know the optimal radius yet, we can't set tight bounds.
    # We can set bounds to [-0.5, 0.5] and let the optimizer work.
    # However, if a point hits the bound, the gradient might be tricky if we use absolute values.
    # The sign function is non-differentiable at 0. But L-BFGS-B is quasi-Newton, 
    # it approximates the Hessian. It should handle small non-smoothness or we can 
    # rely on the fact that optimal points won't be exactly at 0 for coordinates 
    # unless constrained by symmetry, but even then, the subgradient is well-defined.
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Run L-BFGS-B
    # We might need multiple restarts or a long run.
    # L-BFGS-B is good for this.
    
    try:
        res = minimize(
            _qp_objective, 
            centers.flatten(), 
            jac=_qp_gradient, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
        )
        
        if res.success or res.fun < _qp_objective(centers.flatten()):
            new_centers = res.x.reshape(N, 3)
            # Clip to be safe against floating point errors
            new_centers = np.clip(new_centers, -0.5, 0.5)
            new_radius = _compute_radius(new_centers)
            
            if new_radius > best_radius:
                best_centers = new_centers
                best_radius = new_radius
    except Exception as e:
        pass
        
    # If time remains, we could try a second phase with a different initialization 
    # or a local polish using the soft-min method from the original, 
    # but the QP landscape is often quite good.
    # Let's check time.
    if time.time() < deadline - 5:
        # Try a few random restarts with L-BFGS-B?
        # Or just accept the result.
        # The Halton start is usually good.
        # Let's try one more run with a slightly perturbed Halton start if time permits.
        pass

    final_radius = _compute_radius(best_centers)
    if final_radius <= 0:
        # Fallback to a simple grid if something went wrong
        # 4x4x4 grid, take 50
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
# EVOLVE-BLOCK-END
