"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    best_centers = None
    best_radius = 0.0

    # Helper to compute the actual radius (for scoring/selection)
    def _radius_for_centers(centers):
        centers = np.asarray(centers)
        if centers.shape != (N, 3):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        if min_wall < 0:
            return 0.0
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        return min(min_wall, min_pair)

    # Smooth energy function: Sum of repulsive potentials
    # We want to maximize the minimum distance r.
    # Minimizing sum(1/d^p) approximates maximizing min(d) for large p.
    # We also add wall repulsion.
    def _energy_and_grad(x_flat):
        centers = x_flat.reshape(N, 3)
        
        # Ensure we are inside the box for the potential calculation to be well-defined
        # The optimizer bounds will keep them in, but numerical noise might push slightly out.
        # We clip for calculation stability but return gradient based on unclipped positions?
        # Better: rely on bounds. But if a point hits the boundary, gradient should push it back.
        # L-BFGS-B handles bounds, but the objective should be smooth.
        # Let's use a potential that goes to infinity at d=0 and wall=0.
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 1.0) # Avoid div by zero on diagonal
        
        # We use a high power to make it "hard" repulsion, similar to hard spheres
        # V_pair = sum(1/dist^p)
        # Let p=4 or higher.
        p = 4
        # To avoid overflow, we can work with dist_sq
        # 1/dist^p = 1/(dist_sq)^(p/2)
        # Let q = p/2 = 2
        q = 2.0
        
        # Energy from pairs
        # E_pair = sum_{i<j} 1/d_ij^(2q)
        # We sum over all i!=j, so we divide by 2 at the end or just sum upper triangle.
        # For gradient, it's easier to sum all and multiply by 0.5.
        
        # dist_sq is (N, N)
        # inv_dist_sq_q = (dist_sq)^(-q)
        # Careful with diagonal. We set diagonal to 1.0, so term is 1.0. We subtract N later?
        # Or just mask.
        
        # Let's use a soft min / log-sum-exp style or just simple power law.
        # Simple power law is differentiable everywhere except d=0.
        # Since we start from a grid, d > 0.
        
        # To make it smoother near the "optimal" packing, we can use a potential that
        # is flat for d > r_target and rises sharply for d < r_target.
        # But we don't know r_target.
        # The "hard sphere" potential is the standard approach.
        
        # Let's try: E = sum_{i<j} (1/d_ij)^4 + sum_i (1/wall_dist_i)^4
        # This is very steep.
        
        # A better potential for packing is often:
        # E = -sum_{i<j} (1/d_ij)  (attractive? No, repulsive is 1/d)
        # Actually, minimizing sum(1/d^2) works well for TSP-like problems, but for packing,
        # we want to maximize the minimum distance.
        # Minimizing sum(1/d^p) with large p pushes the closest pair apart most aggressively.
        
        # Let's use p=6 (q=3) for strong repulsion.
        q = 3.0
        
        # Calculate pairwise energy and gradient
        # d_ij^2 = dist_sq
        # term = (dist_sq)^(-q)
        # grad_x_i = -q * (dist_sq)^(-q-1) * 2 * (x_i - x_j)
        # Sum over j
        
        # For efficiency and stability:
        # E_pair = 0.5 * sum_{i!=j} (dist_sq[i,j])^(-q)
        # Note: diagonal is 1.0, so (1)^(-q) = 1. We subtract N from the sum to ignore self-interaction.
        
        inv_dist_sq_q = dist_sq ** (-q)
        np.fill_diagonal(inv_dist_sq_q, 0.0) # Zero out self-interaction
        
        E_pair = 0.5 * np.sum(inv_dist_sq_q)
        
        # Gradient of E_pair w.r.t x_k
        # d/dx_k [ 0.5 * sum_{i!=j} (|x_i - x_j|^2)^(-q) ]
        # Term involving x_k:
        # 0.5 * sum_{j!=k} (|x_k - x_j|^2)^(-q) + 0.5 * sum_{i!=k} (|x_i - x_k|^2)^(-q)
        # = sum_{j!=k} (|x_k - x_j|^2)^(-q)
        # Gradient = sum_{j!=k} -q * (|x_k - x_j|^2)^(-q-1) * 2 * (x_k - x_j)
        # = -2q * sum_{j!=k} (dist_sq[k,j])^(-q-1) * (x_k - x_j)
        
        # Let W = (dist_sq)^(-q-1)
        W = dist_sq ** (-(q + 1))
        np.fill_diagonal(W, 0.0)
        
        # grad_pair[k] = -2q * sum_j W[k,j] * (x_k - x_j)
        # = -2q * (W @ x_k? No, W is matrix of coefficients for each pair)
        # sum_j W[k,j] * x_k - sum_j W[k,j] * x_j
        # = x_k * sum_j W[k,j] - (W @ x)[k]
        
        # Let S[k] = sum_j W[k,j]
        S = np.sum(W, axis=1)
        
        # (W @ x)[k] = sum_j W[k,j] * x_j
        Wx = W.dot(centers)
        
        grad_pair = -2.0 * q * (S[:, np.newaxis] * centers - Wx)
        
        # Wall Energy
        # Walls at +/- 0.5
        # For each coordinate c of each center i:
        # d_wall = 0.5 - |x_ic|
        # E_wall = sum_{i,c} (d_wall)^(-q)
        # This is problematic if x_ic is close to 0.5.
        # Also |x| is not differentiable at 0.
        # However, centers should stay away from 0 in the center of the box? Not necessarily.
        # To handle |x|, we can split into positive and negative sides.
        # Or use a smooth approximation.
        # Given the symmetry and the fact that optimal packings usually don't have centers exactly on the axis,
        # we can assume signs are stable or use a soft absolute value.
        # Let's use a smooth abs: sqrt(x^2 + eps)
        eps = 1e-8
        smooth_abs = np.sqrt(centers**2 + eps)
        
        # Distance to nearest wall for each coordinate
        # d = 0.5 - smooth_abs
        # If d <= 0, we are outside. The potential should be huge.
        # To avoid numerical issues with negative base for fractional power, we clip d to a small positive value.
        d_wall = 0.5 - smooth_abs
        d_wall = np.clip(d_wall, 1e-6, 1.0)
        
        E_wall = np.sum(d_wall ** (-q))
        
        # Gradient of E_wall
        # d/dx_ic [ (0.5 - sqrt(x_ic^2 + eps))^(-q) ]
        # Let u = 0.5 - sqrt(x_ic^2 + eps)
        # d/dx u = -x_ic / sqrt(x_ic^2 + eps)
        # d/dx u^(-q) = -q * u^(-q-1) * du/dx
        # = -q * (d_wall)^(-q-1) * (-x_ic / smooth_abs_ic)
        # = q * (d_wall)^(-q-1) * (x_ic / smooth_abs_ic)
        
        d_wall_grad_factor = d_wall ** (-(q + 1))
        # x_ic / smooth_abs_ic is effectively sign(x_ic) for large x, but smooth at 0.
        ratio = centers / smooth_abs
        
        grad_wall = q * d_wall_grad_factor * ratio
        
        # Total Energy and Gradient
        E_total = E_pair + E_wall
        grad_total = grad_pair + grad_wall
        
        return E_total, grad_total.ravel()

    def _neg_energy(x_flat):
        # We minimize -E? No, we want to MINIMIZE the repulsive energy.
        # High repulsion means small distances.
        # So we minimize E_total.
        val, _ = _energy_and_grad(x_flat)
        return val

    def _neg_energy_grad(x_flat):
        _, g = _energy_and_grad(x_flat)
        return g

    # Initializations
    initializations = []
    
    # 1. BCC-like / Checkerboard Grid
    # 4x4x4 grid is 64 points. We take 50.
    # To make it more "BCC", we can offset alternate layers.
    # Layer 0: (0,0,0) type
    # Layer 1: (0.5, 0.5, 0) type relative to layer grid
    # Let's try a distorted grid.
    
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    # Standard grid
    grid_pts = []
    for z in ticks:
        for y in ticks:
            for x in ticks:
                grid_pts.append([x, y, z])
    grid_pts = np.array(grid_pts[:N])
    initializations.append(grid_pts)
    
    # 2. Random
    rng = np.random.RandomState(42)
    rand_pts = rng.uniform(-0.45, 0.45, size=(N, 3))
    initializations.append(rand_pts)
    
    # 3. Two-layer grid (more like BCC projection)
    # 5x5 grid on z=0 and z=0.5? No, 50 points.
    # 5x5x2 = 50.
    ticks_5 = (np.arange(5) + 0.5) / 5 - 0.5
    pts_5x5x2 = []
    for z in [ticks_5[0], ticks_5[-1]]: # Use outer layers? Or middle?
        for y in ticks_5:
            for x in ticks_5:
                pts_5x5x2.append([x, y, z])
    # This is 50 points.
    # Shift the second layer by half a unit to create BCC-like structure
    pts_5x5x2 = np.array(pts_5x5x2)
    # The first 25 are at z[0], next 25 at z[1].
    # z[0] is -0.4, z[1] is 0.4.
    # Shift x,y of the second layer by 0.1 (half of 0.2 spacing)
    shift = 0.1
    pts_5x5x2[25:, 0] += shift
    pts_5x5x2[25:, 1] += shift
    # Clip
    pts_5x5x2 = np.clip(pts_5x5x2, -0.49, 0.49)
    initializations.append(pts_5x5x2)
    
    # 4. Random with symmetry breaking
    rng2 = np.random.RandomState(123)
    rand_pts2 = rng2.uniform(-0.45, 0.45, size=(N, 3))
    initializations.append(rand_pts2)

    bounds = [(-0.499, 0.499)] * (N * 3)

    for init_idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
            
        remaining = deadline - time.time()
        n_remaining = len(initializations) - init_idx
        time_per_init = min(30.0, remaining / (n_remaining + 1))
        
        if time_per_init < 5.0:
            break
            
        x0 = init_centers.flatten()
        
        try:
            # Use L-BFGS-B with gradient
            # We minimize the repulsive energy.
            # The lower the energy, the larger the minimum distance.
            result = minimize(
                _neg_energy,
                x0,
                jac=_neg_energy_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 1000, 'maxls': 50}
            )
            
            if result.success or result.fun < 1e10: # Check if it blew up
                new_centers = result.x.reshape(N, 3)
                # Ensure validity
                if np.all(np.isfinite(new_centers)) and np.all(np.abs(new_centers) <= 0.5):
                    radius = _radius_for_centers(new_centers)
                    if radius > best_radius:
                        best_radius = radius
                        best_centers = new_centers.copy()
        except Exception:
            continue

    if best_centers is None:
        # Fallback
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    return best_centers
# EVOLVE-BLOCK-END
