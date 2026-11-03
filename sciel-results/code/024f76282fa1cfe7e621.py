"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def compute_energy_and_grad(points, r):
    """
    Computes the energy and gradient for the packing problem.
    Energy = sum of squared overlaps + sum of squared boundary violations.
    """
    n = len(points)
    E = 0.0
    grad = np.zeros_like(points)
    
    # Pairwise distances
    # Vectorized calculation for speed
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    dist[dist < 1e-12] = 1e-12
    
    # Overlaps: dist < 2r
    # violation = max(0, 2r - dist)
    violation = np.maximum(0, 2*r - dist)
    
    # Energy contribution from pairs
    # E_pair = sum_{i<j} violation_{ij}^2
    # We sum over all i!=j, so we need to divide by 2 at the end for the energy, 
    # but the gradient logic needs to be consistent.
    # Let's use the full matrix approach.
    
    # Gradient for pairs
    # dE/dci = sum_j 2 * violation_ij * (2r - dist_ij) / dist_ij * (ci - cj) ? 
    # Let v = max(0, 2r - d). E = v^2. dE/dci = 2v * dv/dci.
    # dv/dci = -sign(2r-d) * d(dist)/dci. If 2r>d, sign is 1, so dv/dci = - (ci-cj)/d.
    # So dE/dci = 2v * (-(ci-cj)/d) = 2v * (cj-ci)/d.
    
    # We only care about pairs where violation > 0.
    mask = violation > 0
    # Exclude diagonal
    mask[np.diag_indices(n)] = False
    
    # Calculate gradient contributions
    # coeff[i,j] = 2 * violation[i,j] / dist[i,j]
    coeff = np.zeros((n, n))
    coeff[mask] = 2.0 * violation[mask] / dist[mask]
    
    # grad[i] = sum_j coeff[i,j] * (points[i] - points[j])
    # Note: coeff[i,j] is symmetric.
    # grad = points @ coeff.sum(axis=1) - (coeff @ points) ? No.
    # grad[i] = sum_j coeff[i,j] * (p_i - p_j) = p_i * sum_j coeff[i,j] - sum_j coeff[i,j] p_j
    row_sums = coeff.sum(axis=1)
    grad_pairs = points * row_sums[:, np.newaxis] - coeff @ points
    
    # Boundary violations
    radii = np.linalg.norm(points, axis=1)
    boundary_violation = np.maximum(0, radii + r - 1.0)
    
    E_boundary = np.sum(boundary_violation**2)
    E_pairs = np.sum(violation**2) / 2.0  # Divide by 2 because we counted i,j and j,i
    
    # Gradient for boundary
    # dE/dboundary_violation = 2 * bv
    # dboundary_violation/dp_i = p_i / |p_i| (if violation > 0)
    boundary_grad = np.zeros_like(points)
    mask_b = boundary_violation > 0
    if np.any(mask_b):
        boundary_grad[mask_b] = 2.0 * boundary_violation[mask_b] * (points[mask_b] / radii[mask_b][:, np.newaxis])
    
    total_grad = grad_pairs + boundary_grad
    total_energy = E_pairs + E_boundary
    
    return total_energy, total_grad

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Initialization: Hexagonal lattice
    # We want to place N points.
    # Let's generate a dense hex grid and pick the best N or just use a random start if grid is bad.
    # A random start is often better for breaking symmetry in non-convex problems.
    # Let's use a random start within the disk, but scaled to avoid immediate massive overlaps.
    # Actually, a structured start like rings is often good.
    
    # Let's try a random initialization first, it's robust.
    np.random.seed(42)
    points = np.random.uniform(-1, 1, (N, 2))
    # Scale to be inside disk
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    # Push them in a bit so they don't start on the boundary
    points *= 0.8
    
    best_r = 0.0
    best_points = points.copy()
    
    # Initial radius guess
    # Area of unit circle = pi. Area of N circles = N * pi * r^2.
    # Max r approx sqrt(1/N) * 0.9 (packing efficiency)
    r_init = 0.12
    r_step = 0.001
    
    # Optimization function
    def objective_and_grad(flat_points):
        p = flat_points.reshape(N, 2)
        e, g = compute_energy_and_grad(p, r_init)
        return e, g.flatten()
    
    # We will iteratively increase r_init
    # Start with a small r to ensure a valid configuration is found quickly
    r_init = 0.10
    
    while True:
        if time.time() - start_time > timeout:
            break
            
        # Try to optimize for current r_init
        x0 = best_points.flatten()
        
        # Use L-BFGS-B or BFGS
        # We need to handle the case where energy is already 0
        e0, g0 = objective_and_grad(x0)
        
        if e0 < 1e-12:
            # Already valid, try increasing r
            r_init += r_step
            continue
            
        # Run optimizer
        try:
            res = minimize(
                objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            # If optimization fails, try a smaller step or break
            r_init -= r_step / 2
            if r_init < 0.05:
                break
            continue
        
        e_final, g_final = objective_and_grad(res.x)
        
        if e_final < 1e-8:
            # Valid packing found for this r_init
            best_r = r_init
            best_points = res.x.reshape(N, 2)
            r_init += r_step
            # Maybe increase step size if we are doing well?
            # Keep it simple for now
        else:
            # Failed to find valid packing for this r_init
            # Backtrack
            r_init -= r_step / 2
            if r_init < 0.05:
                break
            # Optionally, try to polish the current best_points with the failed r_init 
            # to see if it was just a local minima issue?
            # For simplicity, we just reduce r_init.
            
    # Final Polish:
    # We have best_r which is valid.
    # Let's do a final high-precision optimization at best_r to squeeze out any last bit of validity
    # or just return the best points found.
    
    # It's possible that the "best_r" is slightly lower than the true max because of the step size.
    # We can do a binary search between last_failed_r and best_r.
    # But given time constraints, the linear scan with small steps should be close enough.
    # The step is 0.001. The best known is ~0.1258.
    # If we reach 0.125, that's very good.
    
    # Ensure points are valid
    # Check if best_points is valid for best_r
    e_check, _ = compute_energy_and_grad(best_points, best_r)
    if e_check > 1e-6:
        # If not, try to find a valid r by decreasing
        # This shouldn't happen if we accepted only valid packs
        pass
        
    return best_points

# EVOLVE-BLOCK-END
