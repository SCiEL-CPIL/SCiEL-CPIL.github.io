"""Circle packing in a unit square - 200 equal circles.

Strategy: soft-minimum (sum-of-exponentials) smooth objective optimized by
L-BFGS-B with logarithmic annealing on the softness parameter beta,
seeded by diverse hexagonal lattice initializations.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 200


def construct():
    t0 = time.time()
    deadline = t0 + 143.0  # safety margin before 150s kill

    def compute_radius(c):
        """Return the largest feasible radius for centers c (N,2)."""
        wall = 0.5 - np.abs(c)
        min_wall = float(np.min(wall))
        # Use chunked computation to avoid memory issues with full NxN matrix
        # Actually 200x200 is fine (40000 elements)
        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = float(np.min(dist))
        return max(0.0, min(min_wall, min_pair * 0.5))

    def soft_min_objective(x, beta):
        """
        Objective to minimize: -soft_min(distances).
        We want to maximize the minimum distance, so minimize negative soft-min.
        soft_min(d_i) = -1/beta * log(sum(exp(-beta * d_i)))
        This approximates min(d_i) as beta -> inf.
        
        x is flattened centers (2N,).
        Includes wall constraints via soft-min as well.
        """
        c = x.reshape(N, 2)
        
        # Wall distances
        wall = 0.5 - np.abs(c)
        
        # Pairwise distances (upper triangle only to save computation)
        # For N=200, full matrix is 40000 elements, which is fine
        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Combine wall and pair distances
        # We want soft_min of all these distances
        # To avoid numerical issues, subtract max before exp
        all_dists = np.concatenate([wall.ravel(), dist[dist < np.inf].ravel()])
        
        # Soft min approximation
        # f = -1/beta * log(sum(exp(-beta * d)))
        # For numerical stability: let d_max = max(all_dists)
        # f = -1/beta * (d_max + log(sum(exp(-beta * (d - d_max)))))
        # Wait, soft_min should approximate the MINIMUM.
        # min(d) approx -1/beta * log(sum(exp(-beta * d)))
        # If we want to MAXIMIZE min(d), we MINIMIZE -soft_min(d)
        # So objective = 1/beta * log(sum(exp(-beta * d)))
        
        d_max = np.max(all_dists)
        # exp(-beta * (d - d_max)) is stable
        exp_terms = np.exp(-beta * (all_dists - d_max))
        log_sum = np.log(np.sum(exp_terms))
        soft_min_val = d_max - log_sum / beta
        # We want to maximize soft_min_val, so minimize -soft_min_val
        return -soft_min_val

    def soft_min_gradient(x, beta):
        """Gradient of the soft-min objective."""
        c = x.reshape(N, 2)
        
        # Wall distances
        wall = 0.5 - np.abs(c)
        
        # Pairwise distances
        diff = c[:, None, :] - c[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # We need d(soft_min)/d(x)
        # soft_min = -1/beta * log(sum(exp(-beta * d_i)))
        # d(soft_min)/d(x_k) = sum_i [ (beta * exp(-beta * d_i) / sum(exp(-beta * d_j))) * d(d_i)/d(x_k) ]
        # Objective = -soft_min, so grad_obj = -grad_soft_min
        
        # For efficiency, let's compute weights w_i = exp(-beta * d_i) / sum(exp(-beta * d_j))
        # Then grad_soft_min_k = beta * sum_i w_i * d(d_i)/d(x_k)
        # grad_obj_k = -beta * sum_i w_i * d(d_i)/d(x_k)
        
        # Combine all distances
        wall_flat = wall.ravel()
        # Get pair distances excluding diagonal
        triu_idx = np.triu_indices(N, k=1)
        pair_dist_flat = dist[triu_idx]
        
        all_dists = np.concatenate([wall_flat, pair_dist_flat])
        
        # Compute weights
        d_max = np.max(all_dists)
        exp_terms = np.exp(-beta * (all_dists - d_max))
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp  # w_i
        
        # Gradient with respect to centers
        grad_c = np.zeros((N, 2))
        
        # Wall gradient contribution
        # d(wall_i)/d(c) = -sign(c_i)
        # wall is (N, 2), flattened is (2N,)
        # indices 0 to 2N-1 correspond to wall distances
        wall_grad = -np.sign(c)  # (N, 2)
        # weights for wall are weights[0:2N]
        w_wall = weights[:2*N].reshape(N, 2)
        grad_c += beta * np.sum(w_wall * wall_grad, axis=1, keepdims=True) * 0  # Wait, sum over what?
        
        # Let's be more careful.
        # wall_flat has 2N elements. Each corresponds to a specific coordinate.
        # wall_flat[k] = 0.5 - |c[i, j]| where k = 2*i + j
        # d(wall_flat[k])/d(x[m]) is non-zero only if x[m] is c[i, j]
        # So the contribution to grad_c[i, j] from wall_flat[k] is w_k * d(wall_flat[k])/d(c[i, j])
        # d(wall_flat[k])/d(c[i, j]) = -sign(c[i, j])
        
        # So grad_c[i, j] += beta * w_wall[i, j] * (-sign(c[i, j]))
        grad_c += beta * w_wall * (-np.sign(c))
        
        # Pair gradient contribution
        # pair_dist_flat has N*(N-1)/2 elements
        # Each corresponds to distance between c[a] and c[b]
        # d(dist_ab)/d(c[a]) = (c[a] - c[b]) / dist_ab
        # d(dist_ab)/d(c[b]) = (c[b] - c[a]) / dist_ab = -(c[a] - c[b]) / dist_ab
        
        # We need to map pair_flat indices back to (a, b)
        # triu_idx gives (rows, cols) for upper triangle
        rows, cols = triu_idx
        w_pair = weights[2*N:]  # weights for pair distances
        
        # For each pair (a, b) with weight w:
        # grad_c[a] += beta * w * (c[a] - c[b]) / dist_ab
        # grad_c[b] -= beta * w * (c[a] - c[b]) / dist_ab
        
        # Vectorized approach:
        # diff_ab = c[rows] - c[cols]  # (M, 2)
        # unit = diff_ab / dist_ab[:, None]
        # contribution to grad_c[rows] += beta * w_pair[:, None] * unit
        # contribution to grad_c[cols] -= beta * w_pair[:, None] * unit
        
        diff_ab = c[rows] - c[cols]
        # Avoid division by zero
        safe_dist = np.where(dist[rows, cols] < 1e-12, 1.0, dist[rows, cols])
        unit = diff_ab / safe_dist[:, None]
        
        contrib = beta * w_pair[:, None] * unit  # (M, 2)
        
        # Accumulate into grad_c
        np.add.at(grad_c, rows, contrib)
        np.add.at(grad_c, cols, -contrib)
        
        return grad_c.ravel()

    def hex_grid(n_rows, n_cols, d_scale=1.0):
        """Generate hexagonal grid points."""
        d = d_scale / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        pts = []
        for row in range(n_rows):
            y = -0.5 + d / 2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d / 2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= N:
            return pts[:N]
        return None

    def square_grid(m):
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    # Generate diverse initial configurations
    inits = []
    for n_rows in range(13, 19):
        n_cols = math.ceil(N / n_rows)
        for d_scale in [0.90, 0.95, 1.0, 1.05, 1.10]:
            pts = hex_grid(n_rows, n_cols, d_scale)
            if pts is not None:
                inits.append(pts)
    
    m = int(math.ceil(math.sqrt(N)))
    inits.append(square_grid(m))
    
    # Add random perturbations
    rng = np.random.default_rng(123)
    for base in inits[:4]:
        perturbed = base + rng.normal(0, 0.01, (N, 2))
        perturbed = np.clip(perturbed, -0.499, 0.499)
        inits.append(perturbed)

    # Score and select top initializations
    scored = []
    for pts in inits:
        r = compute_radius(pts)
        scored.append((r, pts))
    scored.sort(key=lambda x: -x[0])
    top_inits = [pts for _, pts in scored[:min(8, len(scored))]]
    
    if not top_inits:
        top_inits = [square_grid(m)]

    best_centers = None
    best_radius = 0.0

    # Annealing schedule: beta from low to high
    # Low beta = smooth, global; High beta = sharp, local
    # We'll use a logarithmic schedule
    beta_start = 10.0
    beta_end = 500.0
    n_anneal_steps = 5  # number of L-BFGS-B runs with increasing beta
    
    for init_idx, init in enumerate(top_inits):
        if time.time() > deadline - 15:
            break
            
        r_init = compute_radius(init)
        if r_init > best_radius:
            best_radius = r_init
            best_centers = init.copy()
        
        # Run annealing from this initialization
        current_c = init.copy().ravel()
        
        for step in range(n_anneal_steps):
            if time.time() > deadline - 5:
                break
                
            # Interpolate beta logarithmically
            frac = step / max(1, n_anneal_steps - 1)
            beta = beta_start * (beta_end / beta_start) ** frac
            
            # L-BFGS-B optimization
            try:
                result = minimize(
                    fun=soft_min_objective,
                    x0=current_c,
                    jac=soft_min_gradient,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (2 * N),
                    options={
                        'maxiter': 100,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 20
                    }
                )
                current_c = result.x
            except Exception:
                # If optimization fails, keep current
                pass
            
            # Evaluate actual radius
            c_eval = current_c.reshape(N, 2)
            r_eval = compute_radius(c_eval)
            if r_eval > best_radius:
                best_radius = r_eval
                best_centers = c_eval.copy()
        
        # After annealing, do a final high-beta polish
        if time.time() < deadline - 3:
            try:
                result = minimize(
                    fun=soft_min_objective,
                    x0=current_c,
                    jac=soft_min_gradient,
                    args=(beta_end * 2,),  # even sharper
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (2 * N),
                    options={
                        'maxiter': 200,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                        'maxls': 20
                    }
                )
                current_c = result.x
            except Exception:
                pass
            
            c_eval = current_c.reshape(N, 2)
            r_eval = compute_radius(c_eval)
            if r_eval > best_radius:
                best_radius = r_eval
                best_centers = c_eval.copy()

    if best_centers is None:
        best_centers = square_grid(m)

    return best_centers


if __name__ == "__main__":
    result = construct()
    print(f"Shape: {result.shape}")
    print(f"Radius: {compute_radius(result)}")
