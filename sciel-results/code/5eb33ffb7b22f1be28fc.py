"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _energy_and_grad_flat(x):
    """
    Compute energy and gradient for flat vector x of shape (3N,).
    Energy = sum_{i<j} log(1 / ||x_i - x_j||)
    Gradient = sum_{i<j} - (x_i - x_j) / ||x_i - x_j||^2  (for point i)
              = sum_{i<j}   (x_i - x_j) / ||x_i - x_j||^2  (for point j)
    """
    pts = x.reshape(N, 3)
    
    # Compute pairwise distances
    # diff[i, j, k] = pts[i, k] - pts[j, k]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal to avoid division by zero
    np.fill_diagonal(dist_sq, 1.0)
    
    # Check for too-close points
    # We need i < j
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_sq_pairs = dist_sq[i_idx, j_idx]
    
    if np.any(d_sq_pairs < 1e-18):
        # Return large energy and zero gradient to push away? 
        # Or just inf. L-BFGS might struggle with inf.
        # Let's return a large finite number and a gradient pointing away.
        # Actually, if they are too close, it's invalid. 
        # For optimization, we want to avoid this.
        # Let's return inf.
        return float('inf'), np.zeros_like(x)
    
    d_pairs = np.sqrt(d_sq_pairs)
    
    # Energy: -sum(log(d))
    energy = -np.sum(np.log(d_pairs))
    
    # Gradient
    # For each pair (i, j), d = ||p_i - p_j||
    # dE/dd = -1/d
    # dd/dp_i = (p_i - p_j) / d
    # dd/dp_j = (p_j - p_i) / d = -(p_i - p_j) / d
    # dE/dp_i = (-1/d) * (p_i - p_j) / d = -(p_i - p_j) / d^2
    # dE/dp_j = (-1/d) * (-(p_i - p_j)) / d = (p_i - p_j) / d^2
    
    grad = np.zeros_like(pts)
    
    # Vectorized gradient accumulation
    # diff[i, j] is p_i - p_j
    # We need to sum contributions.
    # Contribution to grad[i] from pair (i, j) is -diff[i, j] / dist_sq[i, j]
    # Contribution to grad[j] from pair (i, j) is  diff[i, j] / dist_sq[i, j]
    
    # Let's use broadcasting to compute the term (p_i - p_j) / ||p_i - p_j||^2
    # term[i, j, k] = diff[i, j, k] / dist_sq[i, j]
    # dist_sq has shape (N, N). diff has shape (N, N, 3).
    # We need to divide diff by dist_sq[..., np.newaxis]
    # But we only want i != j.
    
    # Create a mask for i != j
    mask = ~np.eye(N, dtype=bool)
    
    # Compute inv_dist_sq
    inv_dist_sq = np.zeros_like(dist_sq)
    # Avoid division by zero on diagonal, though we set it to 1.0 above.
    # We only care about off-diagonal.
    np.divide(1.0, dist_sq, out=inv_dist_sq, where=mask)
    
    # term[i, j, k] = diff[i, j, k] * inv_dist_sq[i, j]
    term = diff * inv_dist_sq[:, :, np.newaxis]
    
    # grad[i, k] = sum_j term[i, j, k] * (-1)  ? 
    # Wait.
    # dE/dp_i = - sum_{j != i} (p_i - p_j) / ||p_i - p_j||^2
    # term[i, j] = (p_i - p_j) / ||p_i - p_j||^2
    # So grad[i] = - sum_{j != i} term[i, j]
    
    # Sum over j (axis=1)
    grad = -np.sum(term, axis=1)
    
    # Reshape to flat
    grad_flat = grad.flatten()
    
    return energy, grad_flat

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    best_energy = _energy_and_grad_flat(best_pts.flatten())[0]
    
    # 2. Stage 1: Global Refinement with L-BFGS-B
    # We don't use constraints directly, but we can project back to sphere after each step?
    # L-BFGS-B doesn't support projection easily. 
    # Alternative: Use scipy.optimize.minimize with method='L-BFGS-B' on the flat vector.
    # The sphere constraint is non-linear.
    # A common trick is to parameterize points using 2 coordinates and derive the 3rd,
    # but that has singularities at poles.
    # Another trick: Use the fact that for small steps, we can just minimize and then normalize?
    # No, L-BFGS assumes Euclidean space.
    
    # Better approach: Use scipy.optimize.minimize with a custom function that 
    # implicitly handles the sphere? No.
    
    # Let's stick to a simpler, robust method: 
    # Since N=100 is small, we can use a more direct gradient descent with projection,
    # or use L-BFGS on the full 3D coordinates and accept that it might drift off the sphere?
    # No, energy is defined on the sphere. If points leave the sphere, the energy calculation
    # in `_energy_and_grad_flat` assumes they are on the sphere (chordal distance).
    # If they drift, the gradient is wrong.
    
    # Let's use a different optimizer that can handle equality constraints? 
    # Or, simpler: Use the "unconstrained" L-BFGS but project the points back to the sphere
    # after every function evaluation? This breaks the gradient consistency.
    
    # Alternative Strategy: 
    # Use `scipy.optimize.minimize` with method='SLSQP' or 'trust-constr' with constraints?
    # SLSQP is slow.
    
    # Let's try a different angle. 
    # The current seed uses Simulated Annealing. 
    # A deterministic approach: 
    # 1. Start with Fibonacci.
    # 2. Perform "Local Search" using gradient descent on the manifold.
    #    Step: p_new = p - eta * grad_manifold(p)
    #    Project p_new to sphere.
    #    Repeat until convergence.
    
    # Let's implement a simple Riemannian Gradient Descent.
    
    def _rgd_step(pts, lr=0.01, max_iter=1000, tol=1e-10):
        current_pts = pts.copy()
        current_energy = _energy_and_grad_flat(current_pts.flatten())[0]
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            e, g_flat = _energy_and_grad_flat(current_pts.flatten())
            if not np.isfinite(e):
                break
            g = g_flat.reshape(N, 3)
            
            # Project gradient to tangent space
            # g_tangent = g - (g . p) p
            dot = np.sum(g * current_pts, axis=1, keepdims=True)
            g_tan = g - dot * current_pts
            
            # Update
            new_pts = current_pts - lr * g_tan
            
            # Normalize
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            new_pts = new_pts / norms
            
            # Check convergence
            if np.max(np.abs(new_pts - current_pts)) < tol:
                current_pts = new_pts
                break
                
            current_pts = new_pts
            current_energy = e # Approximation, or recalc? Recalc is safer for energy tracking.
        
        final_energy = _energy_and_grad_flat(current_pts.flatten())[0]
        return current_pts, final_energy

    # Stage 1: Refine Fibonacci
    best_pts, best_energy = _rgd_step(best_pts, lr=0.05, max_iter=2000)
    
    # Stage 2: Multi-start Local Search
    # Generate several random perturbations of the best configuration and refine them.
    # This helps escape shallow local minima.
    
    n_restarts = 10
    rng = np.random.default_rng(42)
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Create a perturbed version
        # Option A: Random rotation? No, that doesn't change relative distances.
        # Option B: Add noise to points and re-normalize.
        noise = rng.normal(0, 0.1, size=(N, 3))
        perturbed_pts = best_pts + noise
        norms = np.linalg.norm(perturbed_pts, axis=1, keepdims=True)
        perturbed_pts = perturbed_pts / norms
        
        # Refine
        p, e = _rgd_step(perturbed_pts, lr=0.05, max_iter=2000)
        
        if e < best_energy:
            best_energy = e
            best_pts = p
    
    # Final check and normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_pts = best_pts / norms
    
    # Verify validity
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    i_idx, j_idx = np.triu_indices(N, k=1)
    if np.any(dist_sq[i_idx, j_idx] < 1e-18):
        # Fallback to Fibonacci if something went wrong
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        best_pts = best_pts / norms

    return best_pts
# EVOLVE-BLOCK-END
