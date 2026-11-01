"""Rewritten program: differentiable rigid-body mechanics approach for circle packing."""
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed and select n points closest to center."""
    # Estimate spacing
    # Area of unit circle is pi. Area per circle approx pi/n.
    # Circle radius r ~ sqrt(1/n). Distance between centers ~ 2r.
    # Hex lattice area per point ~ (sqrt(3)/2) * s^2 where s is spacing.
    # s^2 ~ 2*pi / (n*sqrt(3)) -> s ~ sqrt(2*pi/(n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n * np.sqrt(3.0)))
    
    # Generate grid points
    # Radius of grid in units of s
    R_grid = int(1.0 / s) + 2
    points = []
    for j in range(-R_grid, R_grid + 1):
        for i in range(-R_grid, R_grid + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            points.append([x, y])
            
    points = np.array(points)
    dists = np.hypot(points[:, 0], points[:, 1])
    
    # Keep points inside unit circle (with margin for radius)
    # We don't know exact radius yet, but s/2 is a good estimate for r.
    # Let's just keep all points within radius 1.0
    mask = dists <= 1.0
    points = points[mask]
    dists = dists[mask]
    
    if len(points) < n:
        # If not enough, shrink spacing
        for _ in range(20):
            s *= 0.95
            R_grid = int(1.0 / s) + 2
            points = []
            for j in range(-R_grid, R_grid + 1):
                for i in range(-R_grid, R_grid + 1):
                    x = (i + 0.5 * (j % 2)) * s
                    y = j * s * np.sqrt(3.0) / 2.0
                    points.append([x, y])
            points = np.array(points)
            dists = np.hypot(points[:, 0], points[:, 1])
            mask = dists <= 1.0
            points = points[mask]
            dists = dists[mask]
            if len(points) >= n:
                break
            
    # Sort by distance from center and take first n
    idx = np.argsort(dists)
    return points[idx[:n]]

def _compute_score(centers):
    """Compute the packing radius for given centers (vectorized)."""
    # Pairwise distances
    # Use broadcasting to compute all pairwise distances
    # N=300, 300x300 matrix is 90k elements, fine.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _soft_min_objective(x, beta):
    """
    Objective function to minimize: -softmin(packing radius terms).
    Terms are: d_ij/2 for i<j, and 1-|c_i| for all i.
    """
    n = len(x) // 2
    c = x.reshape(n, 2)
    
    # Compute pairwise distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    # Compute boundary terms
    norms = np.linalg.norm(c, axis=1)
    boundary_terms = 1.0 - norms
    
    # Flatten terms
    # We only need unique pairs for the sum to avoid double counting in the energy,
    # but for softmin, double counting just changes the effective beta.
    # To be precise, let's take the upper triangle.
    iu = np.triu_indices(n, k=1)
    pair_vals = pair_terms[iu]
    boundary_vals = boundary_terms
    
    all_vals = np.concatenate([pair_vals, boundary_vals])
    
    # Soft min: -1/beta * log(sum(exp(-beta * vals)))
    # Numerical stability: subtract max val (which is min of -vals)
    # Let's use: m = min(all_vals)
    # softmin = m - 1/beta * log(sum(exp(-beta * (all_vals - m))))
    m = np.min(all_vals)
    exp_terms = np.exp(-beta * (all_vals - m))
    sum_exp = np.sum(exp_terms)
    
    soft_min_val = m - (1.0 / beta) * np.log(sum_exp)
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val

def _soft_min_gradient(x, beta):
    """
    Gradient of -softmin(packing radius terms).
    """
    n = len(x) // 2
    c = x.reshape(n, 2)
    
    # Recompute values and weights
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    pair_terms = dists / 2.0
    
    norms = np.linalg.norm(c, axis=1)
    boundary_terms = 1.0 - norms
    
    iu = np.triu_indices(n, k=1)
    pair_vals = pair_terms[iu]
    boundary_vals = boundary_terms
    
    all_vals = np.concatenate([pair_vals, boundary_vals])
    
    m = np.min(all_vals)
    exp_terms = np.exp(-beta * (all_vals - m))
    sum_exp = np.sum(exp_terms)
    
    # Weights: w_k = exp(-beta * (v_k - m)) / sum_exp
    weights = exp_terms / sum_exp
    
    grad = np.zeros_like(x)
    
    # 1. Pairwise contributions
    # For each pair (i, j), term is d_ij/2.
    # d_dd (d/2) = 0.5 * (c_i - c_j) / d
    # Gradient of -softmin wrt c_i is - sum_k w_k * grad_k(softmin)
    # Wait, objective is -softmin.
    # d(-softmin)/dc = - d(softmin)/dc
    # softmin = sum w_k v_k (in the limit, but generally it's the weighted avg of v_k gradients? No.)
    # Let S = sum exp(-beta v_k). softmin = -1/beta log S.
    # d(softmin)/dc = -1/beta * (1/S) * dS/dc
    # dS/dc = sum exp(-beta v_k) * (-beta * dv_k/dc) = -beta * sum w_k * dv_k/dc
    # So d(softmin)/dc = sum w_k * dv_k/dc
    # Therefore d(-softmin)/dc = - sum w_k * dv_k/dc
    
    # We need dv_k/dc for each term.
    
    # Pair terms
    # v_ij = d_ij / 2
    # dv_ij/dc_i = 0.5 * (c_i - c_j) / d_ij
    # dv_ij/dc_j = 0.5 * (c_j - c_i) / d_ij
    
    # We have weights for each pair in `weights` corresponding to `pair_vals`.
    # We need to accumulate gradients.
    
    # To avoid N^2 loops in Python, we can use vectorized operations if we store the diffs.
    # But N=300, pairs ~45k. A loop over 45k pairs is slow in Python.
    # We can vectorize the gradient accumulation using broadcasting and matrix multiplication.
    
    # Let's construct the gradient for pairs.
    # For each pair (i,j), we add:
    # -w_ij * 0.5 * (c_i - c_j) / d_ij to grad[i]
    # -w_ij * 0.5 * (c_j - c_i) / d_ij to grad[j]
    
    # This is equivalent to:
    # grad[i] -= 0.5 * sum_j w_ij * (c_i - c_j) / d_ij
    # grad[j] += 0.5 * sum_i w_ij * (c_i - c_j) / d_ij
    
    # Let's compute the matrix of weights W (n x n), where W[i,j] = w_ij for i<j, 0 otherwise.
    W = np.zeros((n, n))
    W[iu] = weights[:len(pair_vals)]
    
    # We need D = (c_i - c_j) / d_ij.
    # diff_matrix is n x n x 2.
    # d_matrix is n x n.
    # We want G[i,j] = W[i,j] * (c_i - c_j) / d_ij.
    
    # Note: W is upper triangular.
    # grad[i] = - sum_j G[i,j]  (from the -w * (ci-cj)/d term? No.)
    # Let's re-derive carefully.
    # Term v_ij = 0.5 * ||c_i - c_j||.
    # grad_{c_i} v_ij = 0.5 * (c_i - c_j) / ||c_i - c_j||.
    # grad_{c_j} v_ij = 0.5 * (c_j - c_i) / ||c_i - c_j||.
    
    # Objective F = - softmin.
    # grad F = - sum_k w_k grad v_k.
    
    # For pair (i,j) with weight w_ij:
    # Contribution to grad[i] = - w_ij * 0.5 * (c_i - c_j) / d_ij.
    # Contribution to grad[j] = - w_ij * 0.5 * (c_j - c_i) / d_ij = + w_ij * 0.5 * (c_i - c_j) / d_ij.
    
    # So, let V_ij = (c_i - c_j) / d_ij.
    # grad[i] -= 0.5 * w_ij * V_ij.
    # grad[j] += 0.5 * w_ij * V_ij.
    
    # Sum over j for fixed i:
    # grad[i] -= 0.5 * sum_j w_ij V_ij.
    
    # We can compute this using matrix multiplication.
    # Let M be n x n matrix where M[i,j] = w_ij * V_ij? No, V_ij is a vector.
    # We have two components.
    
    # Let's compute the vector field for each dimension separately.
    # dim 0:
    # V0[i,j] = (c[i,0] - c[j,0]) / d[i,j]
    # grad_x[i] -= 0.5 * sum_j W[i,j] * V0[i,j]
    
    # dim 1:
    # V1[i,j] = (c[i,1] - c[j,1]) / d[i,j]
    # grad_y[i] -= 0.5 * sum_j W[i,j] * V1[i,j]
    
    # W is n x n.
    # V0 is n x n.
    # W * V0 is n x n.
    # Sum over j (axis=1) gives n vector.
    
    # Handle d=0 (diagonal) by setting V=0 there.
    d_safe = np.where(dists > 1e-8, dists, 1.0)
    
    V0 = (c[:, 0:1] - c[0:1, :]) / d_safe
    V1 = (c[:, 1:2] - c[1:2, :]) / d_safe
    
    # Zero out diagonal
    np.fill_diagonal(V0, 0.0)
    np.fill_diagonal(V1, 0.0)
    
    # W is only non-zero for i<j.
    # The sum for grad[i] involves all j.
    # Note that for j<i, W[i,j] is 0. But the pair (j,i) has weight W[j,i] and contributes to grad[i] with + sign?
    # Let's check the sign for grad[j] (where j is the "larger" index in pair (i,j) with i<j).
    # Pair (i,j), i<j.
    # grad[i] -= 0.5 * w * V_ij.
    # grad[j] += 0.5 * w * V_ij.
    
    # So for a fixed index k, it receives contributions from:
    # 1. Pairs (k, j) where k < j. Here k is the "i" role.
    #    Contribution: -0.5 * W[k,j] * V_kj.
    # 2. Pairs (i, k) where i < k. Here k is the "j" role.
    #    Contribution: +0.5 * W[i,k] * V_ik.
    #    Note V_ik = (c_i - c_k) / d_ik = - (c_k - c_i) / d_ki = - V_ki.
    #    So Contribution: +0.5 * W[i,k] * (-V_ki) = -0.5 * W[i,k] * V_ki.
    
    # So in both cases, the contribution to grad[k] is -0.5 * W[other, k] * V_k(other)?
    # Let's verify.
    # Case 1: k is i. Pair (k, j). W[k,j]. V_kj. Contrib: -0.5 W[k,j] V_kj.
    # Case 2: k is j. Pair (i, k). W[i,k]. V_ik. Contrib: +0.5 W[i,k] V_ik.
    # Since V_ik = -V_ki, Contrib = -0.5 W[i,k] V_ki.
    
    # So grad[k] = -0.5 * ( sum_{j>k} W[k,j] V_kj + sum_{i<k} W[i,k] V_ki ).
    
    # This is exactly -0.5 * sum_{j != k} W_eff[k,j] V_kj, where W_eff is the weight matrix.
    # Note W is upper triangular. W[k,j] is non-zero for j>k. W[i,k] is non-zero for i<k.
    # So we can just sum over all j != k using the matrix W?
    # No, W[i,k] is in the upper triangle.
    # Let's construct a full weight matrix W_full where W_full[i,j] = w_ij if i!=j, 0 if i=j.
    # But w_ij is stored in W[i,j] for i<j.
    # So W_full[i,j] = W[i,j] if i<j else W[j,i].
    
    # Actually, we can just use the matrix multiplication with the full symmetric weight matrix?
    # Let W_sym be symmetric such that W_sym[i,j] = w_ij.
    # Then grad[k] = -0.5 * sum_j W_sym[k,j] V_kj.
    
    # Let's build W_sym.
    W_sym = W + W.T
    
    # Now compute sums.
    sum_V0 = np.sum(W_sym * V0, axis=1)
    sum_V1 = np.sum(W_sym * V1, axis=1)
    
    grad[0::2] -= 0.5 * sum_V0
    grad[1::2] -= 0.5 * sum_V1
    
    # 2. Boundary contributions
    # Term v_i = 1 - ||c_i||.
    # grad_{c_i} v_i = - c_i / ||c_i||.
    # Contribution to grad[i] = - w_i * grad v_i = - w_i * (- c_i / ||c_i||) = w_i * c_i / ||c_i||.
    
    # Weights for boundary terms are the second part of `weights`.
    w_b = weights[len(pair_vals):]
    
    norms_safe = np.where(norms > 1e-8, norms, 1.0)
    
    # grad[i] += w_b[i] * c_i / norms[i]
    boundary_grad = (w_b[:, np.newaxis] * c) / norms_safe[:, np.newaxis]
    
    grad[0::2] += boundary_grad[:, 0]
    grad[1::2] += boundary_grad[:, 1]
    
    return grad

def construct():
    """
    Construct 300 equal circles in a unit circle, maximizing the common radius.
    Uses a vectorized soft-min optimization approach.
    """
    # Generate initial hexagonal seed
    centers = _hexagonal_seed(N)
    
    # Initial score
    initial_score = _compute_score(centers)
    
    # Optimization
    # Use a moderate beta for soft-min
    # Too high beta -> noisy gradient, hard to optimize.
    # Too low beta -> approximates mean, not min.
    # Start with beta=50, maybe increase later?
    # For now, single optimization run with beta=100.
    
    beta = 100.0
    
    x0 = centers.flatten()
    
    # Define objective and gradient
    def obj(x):
        return _soft_min_objective(x, beta)
        
    def grad(x):
        return _soft_min_gradient(x, beta)
        
    # Optimize
    result = minimize(
        obj,
        x0,
        jac=grad,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    optimized_centers = result.x.reshape(N, 2)
    
    # Verify and cleanup
    final_score = _compute_score(optimized_centers)
    
    if final_score < initial_score:
        optimized_centers = centers
        final_score = initial_score
        
    # Ensure points are inside
    for i in range(N):
        r = np.linalg.norm(optimized_centers[i])
        if r > 1.0:
            scale = (1.0 - 1e-10) / r
            optimized_centers[i] *= scale
            
    return optimized_centers
# EVOLVE-BLOCK-END
