import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    best_score = 0.0
    best_centers = None

    # Pre-compute pair indices for vectorization
    iu = np.triu_indices(N, 1)
    num_pairs = len(iu[0])

    def _score_raw(centers):
        """Compute the actual score (minimum radius) for a set of centers."""
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0

        norms = np.sqrt(np.sum(centers**2, axis=1))
        if np.any(norms >= 1.0):
            return 0.0

        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        min_pair_dist = np.min(dists[iu])
        min_edge_dist = np.min(1.0 - norms)

        r = min(min_pair_dist / 2.0, min_edge_dist)
        return max(r, 0.0)

    def _objective_and_grad(centers, beta):
        """
        Compute objective (-soft_min) and its gradient.
        Objective: -r_soft = -(-1/beta * log(sum(exp(-beta * x_k))))
                   = 1/beta * log(sum(exp(-beta * x_k)))
        
        Wait, let's re-verify signs.
        We want to MAXIMIZE r.
        r = min(x_k).
        soft_min(x) = -1/beta * log(sum(exp(-beta * x_k))).
        As beta -> inf, soft_min(x) -> min(x).
        So we want to MAXIMIZE soft_min(x).
        Which is equivalent to MINIMIZING -soft_min(x).
        
        Let f(x) = -soft_min(x) = 1/beta * log(sum(exp(-beta * x_k))).
        
        Gradient of f:
        Let S = sum(exp(-beta * x_k)).
        f = 1/beta * log(S).
        df/dx_k = 1/beta * 1/S * dS/dx_k
        dS/dx_k = d/dx_k sum(exp(-beta * x_j)) = -beta * exp(-beta * x_k).
        So df/dx_k = 1/beta * 1/S * (-beta * exp(-beta * x_k)) = -exp(-beta * x_k) / S.
        
        So grad_f = - sum_k (exp(-beta * x_k) / S) * grad(x_k).
        Let w_k = exp(-beta * x_k) / S. Note sum(w_k) = 1.
        grad_f = - sum_k w_k * grad(x_k).
        """
        centers = centers.reshape(N, 2)
        
        # 1. Compute constraints x_k
        # Edge constraints: x_i = 1 - ||c_i||
        norms = np.sqrt(np.sum(centers**2, axis=1))
        edge_constraints = 1.0 - norms
        
        # Pair constraints: x_ij = ||c_i - c_j|| / 2
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists[iu]
        pair_constraints = pair_dists / 2.0
        
        all_constraints = np.concatenate([edge_constraints, pair_constraints])
        
        # 2. Compute weights w_k
        # For numerical stability, subtract max
        neg_beta_x = -beta * all_constraints
        max_val = np.max(neg_beta_x)
        exp_vals = np.exp(neg_beta_x - max_val)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        # Objective value: f = 1/beta * (max_val + log(sum_exp))
        # Note: soft_min = -1/beta * (max_val + log(sum_exp))
        # f = -soft_min = 1/beta * (max_val + log(sum_exp))
        obj_val = (max_val + np.log(sum_exp)) / beta
        
        # 3. Compute Gradient
        grad = np.zeros_like(centers)
        
        # Edge gradients:
        # x_i = 1 - sqrt(c_ix^2 + c_iy^2)
        # d x_i / d c_i = - c_i / ||c_i||
        # grad_f += - w_i * (- c_i / ||c_i||) = w_i * c_i / ||c_i||
        edge_grads = np.zeros_like(centers)
        # Avoid division by zero
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        edge_grads = centers / safe_norms[:, np.newaxis]
        
        # Weighted sum for edges
        # weights[:N] are the weights for edge constraints
        grad += weights[:N, np.newaxis] * edge_grads
        
        # Pair gradients:
        # x_ij = ||c_i - c_j|| / 2
        # d x_ij / d c_i = (c_i - c_j) / (2 * ||c_i - c_j||) = (c_i - c_j) / (2 * 2 * x_ij) ... wait.
        # dist = ||c_i - c_j||. x = dist/2.
        # d x / d c_i = (1/2) * (c_i - c_j) / dist.
        # d x / d c_j = (1/2) * (c_j - c_i) / dist.
        
        diff_pairs = centers[iu[0]] - centers[iu[1]]
        dist_pairs = pair_dists # This is ||c_i - c_j||
        
        # Avoid division by zero
        safe_dists = np.where(dist_pairs > 1e-12, dist_pairs, 1.0)
        
        # Gradient of x_ij w.r.t c_i is (c_i - c_j) / (2 * dist)
        # Gradient of x_ij w.r.t c_j is (c_j - c_i) / (2 * dist)
        
        grad_pair_i = diff_pairs / (2.0 * safe_dists[:, np.newaxis])
        grad_pair_j = -grad_pair_i
        
        # Weights for pairs are weights[N:]
        w_pairs = weights[N:]
        
        # Scatter add to gradient
        # grad_f = - sum w_k grad(x_k)
        # So we subtract the weighted gradients
        
        # For c_i: subtract w * grad_pair_i
        np.add.at(grad, iu[0], -w_pairs[:, np.newaxis] * grad_pair_i)
        # For c_j: subtract w * grad_pair_j (which is w * grad_pair_i)
        # Wait, grad_pair_j is -grad_pair_i.
        # So -w * grad_pair_j = -w * (-grad_pair_i) = +w * grad_pair_i.
        np.add.at(grad, iu[1], w_pairs[:, np.newaxis] * grad_pair_i)
        
        return obj_val, grad.flatten()

    def _make_ring_config(ring_radii, ring_counts, angles_offsets=None):
        """Create a ring-based configuration."""
        centers = []
        if ring_counts[0] > 0:
            centers.append([0.0, 0.0])
        
        for ring_idx in range(1, len(ring_radii)):
            r = ring_radii[ring_idx]
            count = ring_counts[ring_idx]
            offset = angles_offsets[ring_idx] if angles_offsets else 0.0
            for k in range(count):
                theta = 2 * np.pi * k / count + offset
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        return centers

    # Generate initial configurations
    inits = []
    
    # 1. Hexagonal Lattice
    def _hex_init(spacing=0.26):
        points = []
        # Generate a grid large enough
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            return None
        
        # Select N closest to center? Or random subset?
        # Closest to center is usually better for packing density in the core
        norms = np.sqrt(np.sum(points**2, axis=1))
        idx = np.argsort(norms, kind='stable')[:N]
        return points[idx]
        
    hex_cfg = _hex_init()
    if hex_cfg is not None:
        inits.append(hex_cfg)
        
    # Try a few spacings
    for sp in [0.24, 0.25, 0.27, 0.28]:
        cfg = _hex_init(sp)
        if cfg is not None:
            inits.append(cfg)

    # 2. Ring Configurations
    # Known good structure for N=50 is often 1-6-12-18-13 or similar
    # Let's try variations
    ring_specs = [
        ([0.0, 0.25, 0.50, 0.75, 0.95], [1, 6, 12, 18, 13]),
        ([0.0, 0.24, 0.48, 0.74, 0.94], [1, 6, 12, 18, 13]),
        ([0.0, 0.26, 0.52, 0.76, 0.96], [1, 6, 12, 18, 13]),
        ([0.0, 0.25, 0.50, 0.75, 0.95], [1, 6, 12, 19, 12]), # Variation in count
        ([0.0, 0.25, 0.50, 0.75, 0.95], [1, 7, 12, 18, 12]), # Variation
    ]
    
    for radii, counts in ring_specs:
        cfg = _make_ring_config(radii, counts)
        inits.append(cfg)
        
    # Add angular offsets to ring configs
    offsets1 = [0.0, 0.0, np.pi/12, 0.0, 0.0]
    offsets2 = [0.0, 0.0, 0.0, np.pi/18, 0.0]
    offsets3 = [0.0, np.pi/6, 0.0, 0.0, 0.0]
    
    inits.append(_make_ring_config(ring_specs[0][0], ring_specs[0][1], offsets1))
    inits.append(_make_ring_config(ring_specs[0][0], ring_specs[0][1], offsets2))
    inits.append(_make_ring_config(ring_specs[0][0], ring_specs[0][1], offsets3))

    # 3. Random Initializations
    np.random.seed(42)
    for _ in range(5):
        cfg = np.random.uniform(-0.9, 0.9, size=(N, 2))
        # Project inside circle
        norms = np.sqrt(np.sum(cfg**2, axis=1))
        mask = norms > 0.95
        if np.any(mask):
            cfg[mask] *= 0.95 / norms[mask, np.newaxis]
        inits.append(cfg)

    # Optimization Loop
    betas = [20, 50, 100, 200, 400, 800, 1500]
    
    for init_idx, x0 in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
            
        current_centers = x0.copy()
        
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
                
            try:
                # Use L-BFGS-B
                # We need a function that returns (f, g)
                def func(x):
                    return _objective_and_grad(x, beta)
                    
                result = minimize(
                    func,
                    current_centers.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                
                new_centers = result.x.reshape(N, 2)
                
                # Check validity
                if not np.all(np.isfinite(new_centers)):
                    continue
                    
                score = _score_raw(new_centers)
                if score > best_score:
                    best_score = score
                    best_centers = new_centers.copy()
                    current_centers = new_centers.copy()
                    
            except Exception:
                continue

    # Final Polish on the best solution
    if best_centers is not None:
        # Try a few very high beta steps
        for beta in [2000, 3000, 5000]:
            if time.time() > deadline - 5.0:
                break
            try:
                def func(x):
                    return _objective_and_grad(x, beta)
                
                result = minimize(
                    func,
                    best_centers.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                
                new_centers = result.x.reshape(N, 2)
                if not np.all(np.isfinite(new_centers)):
                    continue
                    
                score = _score_raw(new_centers)
                if score > best_score:
                    best_score = score
                    best_centers = new_centers.copy()
                    
            except Exception:
                continue
        
        # Perturbation search
        np.random.seed(123)
        for _ in range(10):
            if time.time() > deadline - 5.0:
                break
            perturbed = best_centers + np.random.normal(0, 0.001, size=(N, 2))
            norms = np.sqrt(np.sum(perturbed**2, axis=1))
            mask = norms > 0.98
            if np.any(mask):
                perturbed[mask] *= 0.98 / norms[mask, np.newaxis]
            
            try:
                def func(x):
                    return _objective_and_grad(x, 1000)
                
                result = minimize(
                    func,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                
                new_centers = result.x.reshape(N, 2)
                if not np.all(np.isfinite(new_centers)):
                    continue
                    
                score = _score_raw(new_centers)
                if score > best_score:
                    best_score = score
                    best_centers = new_centers.copy()
                    
            except Exception:
                continue

    # Fallback
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        best_centers = _make_ring_config([0.0, 0.25, 0.50, 0.75, 0.95], [1, 6, 12, 18, 13])

    return best_centers
