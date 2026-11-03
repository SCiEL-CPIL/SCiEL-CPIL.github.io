"""Rewritten program for packing 50 equal circles in a unit circle."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Place 50 equal, non-overlapping circles inside the unit circle.
    Returns an array of shape (N, 2) with the centres.
    """
    # Initialize with a hexagonal lattice pattern
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    k = int(1.0 / spacing) + 2
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= 1.0:
                rows.append([x, y])
    points_init = np.array(rows)
    
    if len(points_init) < N:
        points_init = np.random.rand(N, 2) * 1.8 - 0.9
        mask = np.hypot(points_init[:, 0], points_init[:, 1]) < 1.0
        points_init = points_init[mask]
        if len(points_init) < N:
            points_init = np.random.rand(N, 2) * 0.8
    
    if len(points_init) >= N:
        points_init = points_init[:N]
    else:
        extra = np.random.rand(N - len(points_init), 2) * 0.8
        points_init = np.vstack([points_init, extra])
    
    def compute_score(pts):
        """Compute the actual packing score."""
        dx = pts[:, 0, np.newaxis] - pts[:, 0, np.newaxis].T
        dy = pts[:, 1, np.newaxis] - pts[:, 1, np.newaxis].T
        dists = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists) / 2.0
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        return min(min_pairwise, min_boundary)

    def objective_and_grad(x):
        """
        Compute objective and gradient for soft-min packing score.
        """
        pts = x.reshape(-1, 2)
        
        # Pairwise distances
        dx = pts[:, 0, np.newaxis] - pts[:, 0, np.newaxis].T
        dy = pts[:, 1, np.newaxis] - pts[:, 1, np.newaxis].T
        dists = np.sqrt(dx * dx + dy * dy + 1e-12)
        
        # Boundary distances
        radii = np.hypot(pts[:, 0], pts[:, 1])
        boundary_dists = 1.0 - radii
        
        # Get upper triangle indices
        iu = np.triu_indices_from(dists, k=1)
        pairwise_vals = dists[iu] / 2.0
        boundary_vals = boundary_dists
        
        all_vals = np.concatenate([pairwise_vals, boundary_vals])
        
        # Soft-min parameters
        T = 0.02  # Temperature
        
        neg_vals = -all_vals / T
        neg_vals = np.clip(neg_vals, -100, 100)
        exp_vals = np.exp(neg_vals)
        sum_exp = np.sum(exp_vals)
        log_sum = np.log(sum_exp + 1e-12)
        soft_min = -T * log_sum
        obj = -soft_min
        
        # Weights for gradient
        weights = exp_vals / (sum_exp + 1e-12)
        
        grad_all = -weights
        
        # Propagate gradient
        grad_pts = np.zeros_like(pts)
        
        n_pairs = len(iu)
        pair_grads = grad_all[:n_pairs]
        
        # Vectorized gradient calculation for pairwise
        # We need to accumulate gradients for each point
        # dists[i,j] = ||p_i - p_j||
        # d(dist/2)/dp_i = 0.5 * (p_i - p_j) / dist
        # d(dist/2)/dp_j = 0.5 * (p_j - p_i) / dist
        
        # Instead of looping, we can use matrix operations if we structure it right,
        # but for N=50, a loop is fast enough and clearer.
        
        for idx in range(n_pairs):
            i, j = iu[0][idx], iu[1][idx]
            d = dists[i, j]
            if d < 1e-8:
                continue
            diff = pts[i] - pts[j]
            w = pair_grads[idx]
            # grad obj w.r.t p_i: -w * (p_i - p_j) / (2*d) * 2 ? 
            # Let's re-verify:
            # v = d/2
            # d(v)/dp_i = 0.5 * (p_i - p_j)/d
            # d(obj)/dv = -w
            # d(obj)/dp_i = -w * 0.5 * (p_i - p_j)/d
            grad_pts[i] += -0.5 * w * diff / d
            grad_pts[j] += 0.5 * w * diff / d
        
        # Boundary part
        n_boundary = len(boundary_vals)
        boundary_grads = grad_all[n_pairs:n_pairs + n_boundary]
        
        # boundary_dists = 1 - |p_i|
        # d(boundary_dists)/dp_i = -p_i / |p_i|
        # d(obj)/dp_i = -w * (-p_i / |p_i|) = w * p_i / |p_i|
        
        for i in range(N):
            r = radii[i]
            if r < 1e-8:
                continue
            grad_pts[i] += boundary_grads[i] * pts[i] / r
        
        grad = grad_pts.flatten()
        return obj, grad

    best_result = None
    best_score = -np.inf
    
    # Adam optimizer parameters
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    max_iter = 2000
    
    # Multiple restarts
    for restart in range(5):
        if restart == 0:
            x = points_init.flatten().copy()
        else:
            x = points_init.flatten().copy() + np.random.normal(0, 0.2, size=2 * N)
            # Project back
            pts = x.reshape(-1, 2)
            r = np.hypot(pts[:, 0], pts[:, 1])
            mask = r > 0.9
            if np.any(mask):
                pts[mask] *= 0.9 / r[mask, np.newaxis]
            x = pts.flatten()
        
        # Adam state
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        
        for t in range(max_iter):
            # Decay learning rate
            current_lr = lr / (1 + t / 100.0)
            
            obj, grad = objective_and_grad(x)
            
            # Adam update
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            
            m_hat = m / (1 - beta1 ** (t + 1))
            v_hat = v / (1 - beta2 ** (t + 1))
            
            x -= current_lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Optional: Project points to stay within unit disk to prevent divergence
            # This is a heuristic to help convergence
            pts = x.reshape(-1, 2)
            r = np.hypot(pts[:, 0], pts[:, 1])
            mask = r > 0.99
            if np.any(mask):
                pts[mask] *= 0.99 / r[mask, np.newaxis]
            x = pts.flatten()
            
            # Early stopping if score is very high
            if t % 100 == 0:
                current_score = compute_score(pts)
                if current_score > 0.1258: # Best known
                    # We can break early if we exceed best known, but let's try to optimize further
                    pass
        
        final_pts = x.reshape(-1, 2)
        # Final projection
        r = np.hypot(final_pts[:, 0], final_pts[:, 1])
        mask = r > 1.0
        if np.any(mask):
            final_pts[mask] *= 0.999 / r[mask, np.newaxis]
        
        score = compute_score(final_pts)
        
        if score > best_score:
            best_score = score
            best_result = final_pts.copy()
            
        # If we find a very good score, we can stop
        if best_score > 0.1258254895304:
            break

    if best_result is None:
        return points_init
    
    return best_result
# EVOLVE-BLOCK-END
