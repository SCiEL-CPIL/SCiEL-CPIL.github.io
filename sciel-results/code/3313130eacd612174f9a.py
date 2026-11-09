"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    # Physics-based relaxation: start with hexagonal lattice, then optimize
    # via gradient ascent on a soft-penalty energy function.
    
    # Generate initial hexagonal lattice points
    def generate_hex_lattice(n_target):
        """Generate hexagonal lattice points and select n_target closest to center."""
        # Try different spacings to get enough points
        for spacing in np.linspace(0.06, 0.04, 50):
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            # Keep points that can fit circles of radius spacing/2
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            if len(points) >= n_target:
                # Select the n_target closest to center
                idx = np.argsort(radius, kind='stable')[:n_target]
                return points[idx]
        # Fallback: uniform random in disk
        angles = np.random.uniform(0, 2 * np.pi, n_target)
        rads = np.sqrt(np.random.uniform(0, 1, n_target))
        return np.column_stack([rads * np.cos(angles), rads * np.sin(angles)])
    
    points = generate_hex_lattice(N)
    
    # Energy function: maximize minimum distance between centers and from boundary
    # We'll minimize a penalty function that penalizes violations
    
    def energy_and_grad(x):
        """
        x: shape (N, 2) array of centers
        Returns: scalar energy (to minimize) and gradient
        """
        x = x.reshape(N, 2)
        # Compute all pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-12)  # avoid division by zero
        
        # Distance from each center to boundary
        rad = np.sqrt(np.sum(x ** 2, axis=1))
        boundary_dist = 1.0 - rad
        
        # We want to maximize the minimum of:
        # 1) min_{i<j} dist_{ij} / 2
        # 2) min_i boundary_dist_i
        # 
        # Equivalent to minimizing a penalty that grows when these are small.
        # Use a soft-min approach: penalize values below a threshold.
        
        # Pairwise distances (upper triangle)
        iu = np.triu_indices(N, k=1)
        pairwise_dists = dist[iu]
        
        # Soft minimum of pairwise distances
        # Use log-sum-exp trick for stability
        # We want to encourage all distances to be large
        # Penalty: sum of exp(-alpha * d) for small d, plus boundary penalty
        
        alpha = 50.0  # sharpness of penalty
        
        # Pairwise penalty: encourages separation
        pairwise_penalty = np.sum(np.exp(-alpha * pairwise_dists))
        
        # Boundary penalty: encourages staying away from boundary
        boundary_penalty = np.sum(np.exp(-alpha * np.maximum(boundary_dist, 0)))
        
        # Total energy to minimize
        energy = pairwise_penalty + boundary_penalty
        
        # Gradient computation
        # d(energy)/d(x_i)
        grad = np.zeros_like(x)
        
        # Pairwise gradient
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            d = diff[i, j]
            dist_ij = dist[i, j]
            # d/dx_i of exp(-alpha * dist_ij) = -alpha * exp(-alpha * dist_ij) * d(dist_ij)/d(x_i)
            # d(dist_ij)/d(x_i) = (x_i - x_j) / dist_ij = -d / dist_ij
            # So d/dx_i = -alpha * exp(-alpha * dist_ij) * (-d / dist_ij) = alpha * exp(-alpha * dist_ij) * d / dist_ij
            coeff = alpha * np.exp(-alpha * dist_ij) / dist_ij
            grad[i] += coeff * d
            grad[j] -= coeff * d
        
        # Boundary gradient
        for i in range(N):
            if rad[i] > 1e-10:
                # d/dx_i of exp(-alpha * (1 - rad_i)) = -alpha * exp(-alpha * (1 - rad_i)) * d(1 - rad_i)/d(x_i)
                # d(1 - rad_i)/d(x_i) = -x_i / rad_i
                # So = -alpha * exp(-alpha * (1 - rad_i)) * (-x_i / rad_i) = alpha * exp(-alpha * (1 - rad_i)) * x_i / rad_i
                bd = 1.0 - rad[i]
                if bd > 0:
                    coeff = alpha * np.exp(-alpha * bd) / rad[i]
                    grad[i] += coeff * x[i]
        
        return energy, grad
    
    # Optimize using L-BFGS-B
    x0 = points.flatten()
    
    # Run multiple restarts with different initial conditions
    best_x = x0.copy()
    best_energy = energy_and_grad(x0)[0]
    
    for restart in range(3):
        if restart > 0:
            # Add small random perturbation
            x0 = points.flatten() + np.random.normal(0, 0.005, size=N * 2)
        
        try:
            result = minimize(
                energy_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_x = result.x
        except Exception:
            pass
    
    x_opt = best_x.reshape(N, 2)
    
    # Final check: ensure all points are inside the unit circle
    rad = np.hypot(x_opt[:, 0], x_opt[:, 1])
    mask = rad < 1.0
    if not np.all(mask):
        # Project points that are outside back inside
        for i in range(N):
            if rad[i] >= 1.0:
                x_opt[i] = x_opt[i] / rad[i] * 0.999
    
    return x_opt
# EVOLVE-BLOCK-END
