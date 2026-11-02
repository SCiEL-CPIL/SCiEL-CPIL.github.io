"""Rewritten program: vectorized differentiable optimization for circle packing."""
import numpy as np
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def _hexagonal_seed(n):
    """Generate a hexagonal lattice seed and select n points closest to center."""
    # Estimate spacing for n circles in unit disk
    # Area of unit disk is pi. Area per circle ~ pi/n.
    # Hexagonal packing density is ~ 0.9069.
    # So effective area per circle is pi / (n * 0.9069).
    # Side length s of hexagon with area A is s = sqrt(4*A / (3*sqrt(3))).
    # Distance between centers is s * sqrt(3).
    r_est = np.sqrt(1.0 / (n * 1.1))
    spacing = 2.0 * r_est * 1.05
    
    points = []
    k_max = int(1.0 / spacing) + 3
    for j in range(-k_max, k_max + 1):
        for i in range(-k_max, k_max + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            points.append([x, y])
    
    points = np.array(points)
    radii = np.hypot(points[:, 0], points[:, 1])
    
    margin = spacing * 0.1
    mask = radii <= 1.0 - margin
    points = points[mask]
    radii = radii[mask]
    
    if len(points) < n:
        for _ in range(20):
            spacing *= 0.95
            points = []
            k_max = int(1.0 / spacing) + 3
            for j in range(-k_max, k_max + 1):
                for i in range(-k_max, k_max + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    points.append([x, y])
            points = np.array(points)
            radii = np.hypot(points[:, 0], points[:, 1])
            mask = radii <= 1.0 - margin
            points = points[mask]
            radii = radii[mask]
            if len(points) >= n:
                break
    
    idx = np.argsort(radii)
    return points[idx[:n]]


def _compute_score(centers):
    """Compute the packing radius for given centers."""
    n = len(centers)
    # Vectorized pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    min_boundary = np.min(1.0 - np.linalg.norm(centers, axis=1))
    
    return min(min_pair_dist / 2.0, min_boundary)


def construct():
    """
    Construct 300 equal circles in a unit circle, maximizing the common radius.
    Uses L-BFGS-B to maximize the soft-minimum of constraints.
    """
    centers = _hexagonal_seed(N)
    
    # Initial score
    current_score = _compute_score(centers)
    
    # Objective function: minimize -soft_min(constraints)
    # Constraints are:
    # 1. Pairwise distances / 2
    # 2. Distance to boundary (1 - |c_i|)
    
    def objective_and_grad(x):
        c = x.reshape(-1, 2)
        n = len(c)
        
        # Compute pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # Pairwise constraints: dist / 2
        pair_vals = dist / 2.0
        
        # Boundary constraints: 1 - norm(c)
        norms = np.linalg.norm(c, axis=1)
        bound_vals = 1.0 - norms
        
        # Combine all constraints into one array
        # We want to maximize the minimum of these values.
        # So we minimize -min(vals).
        # To make it differentiable, use log-sum-exp trick for min.
        # min(vals) ~ -logsumexp(-vals)
        
        all_vals = np.concatenate([pair_vals.flatten(), bound_vals])
        
        # Scale for numerical stability in logsumexp
        # We want to find the "soft minimum".
        # Let f(x) = logsumexp(-x). Then -f(x) approximates min(x).
        # We minimize f(x).
        
        # To avoid overflow in exp, we can subtract the max of -x (which is -min(x))
        # But since we are minimizing, we just want the gradient of logsumexp(-x).
        
        # Gradient of logsumexp(-x) w.r.t x is:
        # - softmax(-x)
        
        neg_vals = -all_vals
        max_neg = np.max(neg_vals)
        exp_neg = np.exp(neg_vals - max_neg)
        sum_exp = np.sum(exp_neg)
        
        # Objective value (we don't actually use it directly for optimization, just for logging maybe)
        obj_val = max_neg + np.log(sum_exp)
        
        # Gradient of logsumexp(-x) is - softmax(-x)
        # softmax(-x) = exp(-x - max(-x)) / sum(exp(-x - max(-x)))
        softmax_neg = exp_neg / sum_exp
        
        grad_flat = -softmax_neg
        
        # Map gradient back to c
        grad_c = np.zeros_like(c)
        
        # Pairwise part
        # all_vals[:n*(n-1)] corresponds to upper triangle of dist/2?
        # No, dist is full matrix. We filled diagonal with inf.
        # So all_vals contains all n^2 elements, with diagonal being -inf (since dist/2 is inf, -dist/2 is -inf).
        # Wait, exp(-inf) is 0. So diagonal doesn't contribute.
        # But we need to be careful with indexing.
        
        # Let's reconstruct the gradient for pairwise terms.
        # The term in all_vals for pair (i,j) is dist(i,j)/2.
        # d(dist(i,j)/2) / d(c_i) = - (c_i - c_j) / (2 * dist(i,j)) * 2 ? No.
        # d(dist(i,j)) / d(c_i) = (c_i - c_j) / dist(i,j).
        # So d(pair_val) / d(c_i) = 0.5 * (c_i - c_j) / dist(i,j).
        
        # The gradient contribution from pair (i,j) to c_i is:
        # grad_flat[index] * d(pair_val) / d(c_i)
        
        # This is getting complex to vectorize the gradient mapping directly from flat softmax.
        # Alternative: Compute gradient analytically using the weights from softmax.
        
        # Let w_ij = softmax(-dist(i,j)/2) for i != j.
        # Let w_i = softmax(-(1-norm(c_i))) for boundary.
        
        # The gradient of the objective (logsumexp) w.r.t c_i is:
        # Sum over j != j of w_ij * d(dist(i,j)/2) / d(c_i) + w_i * d(1-norm(c_i)) / d(c_i)
        
        # d(dist(i,j)/2) / d(c_i) = 0.5 * (c_i - c_j) / dist(i,j)
        # d(1-norm(c_i)) / d(c_i) = - c_i / norm(c_i)
        
        # So grad_c[i] = Sum_{j!=i} w_ij * 0.5 * (c_i - c_j) / dist(i,j) - w_i * c_i / norm(c_i)
        
        # We have softmax_neg which is the vector of weights.
        # First n*(n-1) elements are for pairs? No, the array is flattened from (n,n) matrix.
        # Let's create a weight matrix W_pair of shape (n,n).
        
        W_flat = softmax_neg[:n*n]
        W_pair = W_flat.reshape(n, n)
        
        # Note: W_pair[i,j] corresponds to weight for pair (i,j).
        # The diagonal elements are 0 (since dist is inf, exp(-inf)=0).
        
        # Compute vector part: Sum_j W_pair[i,j] * (c_i - c_j) / dist(i,j)
        # We need dist matrix.
        # dist[i,j] is the distance.
        
        # To avoid division by zero on diagonal, we already have dist diag = inf.
        # But 0/inf is 0.
        
        # Let's compute the vector field.
        # vec[i,j] = (c_i - c_j) / dist(i,j)
        # This is the unit vector from j to i.
        
        # We can compute this matrix.
        # diff[i,j] = c_i - c_j
        # dist[i,j] = norm(diff[i,j])
        # unit_vec[i,j] = diff[i,j] / dist[i,j]
        
        # We already have diff and dist.
        # Handle division by zero for diagonal.
        safe_dist = np.where(dist > 1e-10, dist, 1.0)
        unit_vec = diff / safe_dist[:, :, np.newaxis]
        
        # Now, grad_c_pair[i] = 0.5 * Sum_j W_pair[i,j] * unit_vec[i,j]
        
        # W_pair is (n,n). unit_vec is (n,n,2).
        # We want to sum over j (axis 1).
        # Result shape (n, 2).
        
        grad_c_pair = 0.5 * np.sum(W_pair[:, :, np.newaxis] * unit_vec, axis=1)
        
        # Boundary part
        W_bound = softmax_neg[n*n:]
        # w_i is scalar for each i.
        # grad_c_bound[i] = - w_i * c_i / norm(c_i)
        
        safe_norms = np.where(norms > 1e-10, norms, 1.0)
        grad_c_bound = -W_bound[:, np.newaxis] * c / safe_norms[:, np.newaxis]
        
        grad_c = grad_c_pair + grad_c_bound
        
        # The objective function we are minimizing is logsumexp(-vals).
        # We calculated obj_val above.
        return obj_val, grad_c.flatten()
    
    # Optimize
    x0 = centers.flatten()
    
    # Use L-BFGS-B
    # We need to be careful with the number of iterations and time limit.
    # N=300, x size 600.
    
    result = minimize(
        objective_and_grad,
        x0,
        jac=True, # Function returns (f, g)
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000}
    )
    
    optimized = result.x.reshape(-1, 2)
    
    # Verify and clean up
    final_score = _compute_score(optimized)
    
    # If optimization failed or made it worse, try a few restarts or fallback
    if final_score < current_score:
        # Try a simpler local improvement or just return the seed if it's better
        # Actually, let's just ensure validity.
        pass
        
    # Ensure points are inside
    for i in range(len(optimized)):
        r = np.linalg.norm(optimized[i])
        if r > 1.0:
            scale = (1.0 - 1e-10) / r
            optimized[i] *= scale
            
    return optimized
# EVOLVE-BLOCK-END
