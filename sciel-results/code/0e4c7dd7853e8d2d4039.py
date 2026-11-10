"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # 1. Generate initial hexagonal lattice points
    def gen_hex_lattice(spacing):
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing * 0.6:
                    pts.append((x, y))
        return np.array(pts)

    # Initial spacing estimate
    # Area argument: N * pi * r^2 <= pi * 1^2, so r <= 1/sqrt(N)
    # Hex packing efficiency ~0.907, so r ~ sqrt(0.907/N)
    # Spacing = 2r ~ 2*sqrt(0.907/N) ~ 2*sqrt(0.907/300) ~ 0.078
    spacing = 0.078
    points = gen_hex_lattice(spacing)
    
    # If we don't have enough points, reduce spacing
    while len(points) < N and spacing > 0.01:
        spacing *= 0.95
        points = gen_hex_lattice(spacing)
        
    # If we have more than N, take the ones closest to center
    if len(points) >= N:
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        init_points = points[sorted_idx[:N]]
    else:
        # If still not enough, just pad with random points (should not happen)
        init_points = points
        while len(init_points) < N:
            init_points = np.vstack([init_points, np.random.uniform(-0.9, 0.9, size=(1, 2))])
        init_points = init_points[:N]

    # 2. Define energy function to minimize
    # We want to maximize the minimum distance between any two circles and the boundary.
    # Score R = min( min_{i<j} |c_i - c_j|/2, min_i (1 - |c_i|) )
    # Instead of directly maximizing R (non-smooth), we minimize a potential energy
    # that encourages separation.
    # E = sum_{i<j} 1/d_ij^2 + sum_i 1/(1-|c_i|)^2

    def energy_and_grad(flat_x):
        x = flat_x.reshape((N, 2))
        
        # Calculate pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero for i==j
        mask = ~np.eye(N, dtype=bool)
        
        # Energy term 1: Pairwise repulsion
        # E_pair = sum_{i!=j} 1/d_ij^2
        # Note: dist_sq is d^2. So we want 1/dist_sq.
        # To avoid inf at start if points coincide (unlikely with hex init), add small epsilon
        inv_dist_sq = 1.0 / (dist_sq + 1e-9)
        np.fill_diagonal(inv_dist_sq, 0.0)
        
        E_pair = np.sum(inv_dist_sq)
        
        # Gradient of E_pair w.r.t x
        # d/dx_i (1/|x_i - x_j|^2) = -1 * |x_i - x_j|^-4 * 2 * (x_i - x_j)
        # So grad_i = sum_{j != i} -2 * (x_i - x_j) / |x_i - x_j|^4
        # dist_sq is |x_i - x_j|^2. So |x_i - x_j|^4 is dist_sq^2.
        # factor = -2 / dist_sq^2
        # grad = sum_j factor * (x_i - x_j)
        
        # We have diff[i,j] = x_i - x_j
        # inv_dist_sq[i,j] = 1/dist_sq[i,j]
        # We need 1/dist_sq^2 for the gradient magnitude scaling?
        # Let's derive carefully.
        # f(x) = 1/|x|^2 = |x|^-2
        # grad f(x) = -2 |x|^-4 x
        # So term is -2 * (x_i - x_j) / dist_sq[i,j]^2
        
        grad_factor = -2.0 / (dist_sq + 1e-9)**2
        np.fill_diagonal(grad_factor, 0.0)
        
        # grad_x_i = sum_j grad_factor[i,j] * diff[i,j]
        # diff is (N, N, 2). grad_factor is (N, N).
        # Multiply grad_factor[:, :, np.newaxis] with diff, sum over j (axis 1)
        grad_pair = np.sum(grad_factor[:, :, np.newaxis] * diff, axis=1)
        
        # Energy term 2: Boundary repulsion
        # E_bound = sum_i 1 / (1 - |c_i|)^2
        # Let r_i = |c_i|. term = (1 - r_i)^-2
        # d/dx_i (1 - r_i)^-2 = -2 (1 - r_i)^-3 * (-1) * dr_i/dx_i
        # dr_i/dx_i = x_i / r_i
        # So grad = 2 * (1 - r_i)^-3 * (x_i / r_i)
        
        r_i = np.linalg.norm(x, axis=1)
        # Avoid division by zero if center is exactly at origin (unlikely for all)
        r_safe = np.where(r_i < 1e-9, 1e-9, r_i)
        
        # (1 - r_i) must be positive. If a point is outside, this becomes complex or negative.
        # We should constrain points to stay inside.
        # If 1 - r_i < 0, the energy should be huge.
        # Let's use a penalty that grows if r_i > 1.
        # Actually, simpler: if we optimize repulsion, they won't go out unless forced.
        # But to be safe, let's define distance to boundary d_b = 1 - r_i.
        # If d_b < 0, we are invalid.
        # Let's use: E_bound = sum_i (1 / max(d_b, 1e-9)^2)
        # This will blow up if they try to go out.
        
        d_b = 1.0 - r_i
        d_b_safe = np.where(d_b < 1e-9, 1e-9, d_b)
        
        E_bound = np.sum(1.0 / d_b_safe**2)
        
        # Gradient of E_bound
        # d/dx_i (1/d_b^2) = -2 d_b^-3 * d(d_b)/dx_i
        # d_b = 1 - r_i
        # d(d_b)/dx_i = - x_i / r_i
        # So grad = -2 d_b^-3 * (- x_i / r_i) = 2 * (x_i / r_i) / d_b^3
        
        grad_bound = 2.0 * (x / r_safe[:, np.newaxis]) / d_b_safe[:, np.newaxis]**3
        
        # Handle case where d_b was clamped
        # If d_b < 1e-9, the gradient direction is ambiguous, but the force is huge.
        # The clamping in E_bound creates a discontinuity.
        # For optimization stability, this is usually fine.
        
        total_energy = E_pair + E_bound
        total_grad = grad_pair + grad_bound
        
        return total_energy, total_grad

    # Initial guess
    x0 = init_points.flatten()
    
    # Use L-BFGS-B
    # We don't have explicit bounds other than staying in unit circle.
    # The energy function penalizes going out.
    # We can set bounds to (-1, 1) for all coordinates to help.
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    best_points = init_points
    best_score = -1.0

    def calculate_score(points):
        # Pairwise min distance
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_dist = np.min(np.sqrt(dist_sq))
        
        # Boundary min distance
        r_i = np.linalg.norm(points, axis=1)
        min_bound_dist = np.min(1.0 - r_i)
        
        return min(min_pair_dist / 2.0, min_bound_dist)

    best_score = calculate_score(init_points)

    try:
        res = minimize(
            energy_and_grad, 
            x0, 
            jac=True, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 2000, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        optimized_points = res.x.reshape((N, 2))
        
        # Check validity
        if np.all(np.isfinite(optimized_points)):
            # Ensure points are inside the circle
            r_i = np.linalg.norm(optimized_points, axis=1)
            if np.all(r_i <= 1.0):
                score = calculate_score(optimized_points)
                if score > best_score:
                    best_points = optimized_points
                    best_score = score
    except Exception:
        pass

    # Check time
    if time.time() > deadline:
        return best_points

    # If we have time, try a second pass with tighter tolerance or different initialization?
    # For now, just return the best found.
    return best_points
# EVOLVE-BLOCK-END
