# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a relaxation approach with a Newton-Raphson solver to find equilibrium
    positions for a given radius. A binary search on the radius is performed
    to find the maximum feasible radius.
    """
    
    def compute_energy_and_grad(centers, r):
        """
        Compute the energy and gradient for a fixed radius r.
        Energy is the sum of squared overlap violations and boundary violations.
        """
        n = len(centers)
        energy = 0.0
        grad = np.zeros_like(centers)
        
        # Boundary constraint: centers must be within (1 - r)
        # Penalty: max(0, |c| - (1-r))^2
        dists = np.linalg.norm(centers, axis=1)
        violations = np.maximum(0, dists - (1.0 - r))
        energy += np.sum(violations ** 2)
        
        # Gradient of boundary penalty
        # d/dx (max(0, |x| - R)^2) = 2 * max(0, |x|-R) * (x/|x|)
        mask = dists > (1.0 - r)
        if np.any(mask):
            # Avoid division by zero
            safe_dists = np.where(dists > 1e-12, dists, 1.0)
            factor = 2.0 * violations[mask] / safe_dists[mask]
            grad[mask] += factor[:, np.newaxis] * centers[mask]
            
        # Pairwise constraint: |c_i - c_j| >= 2r
        # Penalty: max(0, 2r - d_ij)^2
        if n > 1:
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists_sq = np.sum(diff ** 2, axis=2)
            dists_pair = np.sqrt(dists_sq)
            np.fill_diagonal(dists_pair, np.inf)
            
            overlap_violations = np.maximum(0, 2.0 * r - dists_pair)
            energy += np.sum(overlap_violations ** 2)
            
            # Gradient calculation
            # For each pair (i, j) with violation v = 2r - d
            # Force on i: 2*v * (c_i - c_j) / d
            # Force on j: 2*v * (c_j - c_i) / d = -Force on i
            # We sum over all j != i for each i
            
            # To avoid O(N^3) or heavy loops, we can vectorize the update
            # grad_i += sum_{j != i} 2 * v_ij * (c_i - c_j) / d_ij
            
            # Efficiently:
            # Create a matrix of forces
            # factor_ij = 2 * v_ij / d_ij
            # force_on_i = sum_j factor_ij * (c_i - c_j)
            
            # Handle zeros in dists_pair (should be inf on diagonal, but check for coincident points)
            safe_dists_pair = np.where(dists_pair > 1e-12, dists_pair, 1.0)
            factors = 2.0 * overlap_violations / safe_dists_pair
            np.fill_diagonal(factors, 0.0)
            
            # grad_i = sum_j factors[i,j] * (c_i - c_j)
            # = sum_j factors[i,j] * c_i - sum_j factors[i,j] * c_j
            # = c_i * sum_j factors[i,j] - (factors @ c)
            
            sum_factors = np.sum(factors, axis=1)
            grad += sum_factors[:, np.newaxis] * centers - (factors @ centers)
            
        return energy, grad

    def relax(centers, r, max_iter=500, tol=1e-10):
        """
        Minimize energy for fixed r using Newton's method (or BFGS fallback).
        """
        current_centers = centers.copy()
        n = len(current_centers)
        
        # Hessian approximation or numerical Hessian?
        # For robustness, let's use BFGS from scipy as it handles non-convexities better than raw Newton
        # but the directive asks for "implicit" relaxation. BFGS is an implicit quasi-Newton method.
        
        def obj_and_grad(x):
            c = x.reshape(n, 2)
            e, g = compute_energy_and_grad(c, r)
            return e, g.flatten()
            
        res = minimize(obj_and_grad, current_centers.flatten(), jac=True, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        
        final_centers = res.x.reshape(n, 2)
        final_energy, _ = compute_energy_and_grad(final_centers, r)
        
        return final_centers, final_energy

    def is_valid(centers, r, tol=1e-6):
        """Check if the packing is valid for radius r."""
        n = len(centers)
        # Boundary
        if np.any(np.linalg.norm(centers, axis=1) > 1.0 - r + tol):
            return False
        # Pairwise
        if n > 1:
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) < 2.0 * r - tol:
                return False
        return True

    def get_actual_radius(centers):
        """Compute the actual maximum radius for given centers."""
        n = len(centers)
        min_boundary = np.min(1.0 - np.linalg.norm(centers, axis=1))
        if n > 1:
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
        else:
            min_pair = np.inf
        return min(min_boundary, min_pair)

    def make_initial_hexagonal(n, scale=0.12):
        """Create initial configuration using hexagonal lattice."""
        points = []
        # Generate hexagonal lattice points
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = i * scale + 0.5 * scale * (j % 2)
                y = j * scale * np.sqrt(3) / 2
                if np.sqrt(x * x + y * y) < 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        
        # If we have more than n points, select n closest to center
        if len(points) > n:
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)[:n]
            points = points[indices]
        elif len(points) < n:
            # Add random points if not enough
            np.random.seed(42)
            while len(points) < n:
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, 0.9)
                points = np.vstack([points, [radius * np.cos(angle), radius * np.sin(angle)]])
        
        return points

    def make_initial_ring(n, rings=5):
        """Create initial configuration using concentric rings."""
        points = [[0.0, 0.0]]
        remaining = n - 1
        for ring in range(1, rings + 1):
            if remaining <= 0:
                break
            n_on_ring = min(remaining, 6 * ring)
            radius = ring * 0.12 # Adjusted scale
            for i in range(n_on_ring):
                angle = 2 * np.pi * i / n_on_ring
                points.append([radius * np.cos(angle), radius * np.sin(angle)])
            remaining -= n_on_ring
        
        while len(points) < n:
            np.random.seed(42)
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, 0.9)
            points.append([radius * np.cos(angle), radius * np.sin(angle)])
        
        return np.array(points[:n])

    def make_initial_random(n, seed=42):
        """Create random initial configuration."""
        np.random.seed(seed)
        points = []
        for _ in range(n):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, 0.95)
            points.append([radius * np.cos(angle), radius * np.sin(angle)])
        return np.array(points)

    # Multi-start strategy
    best_r_global = 0.0
    best_centers_global = None
    
    initial_configs = [
        make_initial_hexagonal(N, scale=0.12),
        make_initial_hexagonal(N, scale=0.15),
        make_initial_ring(N, rings=5),
        make_initial_ring(N, rings=7),
        make_initial_random(N, seed=42),
        make_initial_random(N, seed=123),
        make_initial_random(N, seed=456),
    ]
    
    # Binary search range for r
    r_low = 0.10
    r_high = 0.13 # Known best is ~0.1258
    
    for init_centers in initial_configs:
        # Try to find the max r for this initial configuration
        r_lo = r_low
        r_hi = r_high
        best_r_local = r_lo
        best_centers_local = init_centers
        
        # Binary search for r
        for _ in range(20): # 20 iterations of binary search
            r_mid = (r_lo + r_hi) / 2.0
            relaxed_centers, energy = relax(init_centers, r_mid, max_iter=200)
            
            if is_valid(relaxed_centers, r_mid, tol=1e-5):
                # Valid, try higher r
                r_lo = r_mid
                best_r_local = r_mid
                best_centers_local = relaxed_centers
            else:
                # Invalid, try lower r
                r_hi = r_mid
        
        # Refine the best found r with a final optimization
        if best_r_local > best_r_global:
            # Do a final polish to maximize the actual radius
            # We can use the relaxed centers from the highest valid r
            # And then run a local optimization on the positions to increase the min distance
            
            # Let's just use the relaxed centers for the highest valid r
            final_centers = best_centers_local
            final_r = get_actual_radius(final_centers)
            
            if final_r > best_r_global:
                best_r_global = final_r
                best_centers_global = final_centers
    
    # If no valid packing found (unlikely), fallback
    if best_centers_global is None:
        best_centers_global = make_initial_hexagonal(N, scale=0.12)
        
    return best_centers_global
# EVOLVE-BLOCK-END
