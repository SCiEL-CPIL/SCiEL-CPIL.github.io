"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import time

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize


def _energy(points):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    # points: (N, 3) array on unit sphere
    # Use pairwise distances
    # To avoid O(N^2) memory for large N, do it in chunks or just use broadcasting for N=150
    # 150x150 is small enough
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Avoid log(0) and self-interactions
    np.fill_diagonal(dists, 1.0)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    # We only want i < j
    # log(1/d) = -log(d)
    # Sum over i<j
    # Create upper triangle indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dists[i_idx, j_idx]
    # Filter out any zero distances (shouldn't happen if points are distinct)
    valid = d_ij > 1e-12
    if not np.all(valid):
        d_ij = d_ij[valid]
    return -np.sum(np.log(d_ij))


def _energy_grad(points):
    """Compute energy and gradient."""
    # Energy
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dists[i_idx, j_idx]
    valid = d_ij > 1e-12
    d_ij = d_ij[valid]
    energy = -np.sum(np.log(d_ij))
    
    # Gradient
    # E = sum_{i<j} -log(||x_i - x_j||)
    # dE/dx_i = sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    # But on the sphere, we project the gradient to the tangent space.
    # The unconstrained gradient is grad_u.
    # The constrained gradient is (I - x_i x_i^T) grad_u
    
    grad = np.zeros_like(points)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            v = points[i] - points[j]
            d2 = np.dot(v, v)
            if d2 < 1e-24:
                continue
            # Contribution to grad[i] from pair (i,j)
            # d/dx_i (-log(||x_i - x_j||)) = -(x_i - x_j) / ||x_i - x_j||^2
            # Wait, d/dx log(||x-y||) = (x-y)/||x-y||^2
            # So d/dx (-log(||x-y||)) = -(x-y)/||x-y||^2
            # But we have sum_{i<j}.
            # For a fixed i, it appears in pairs (i,j) where j>i and (j,i) where j<i.
            # Let's just sum over all j != i.
            # Term for pair (i,j) in the sum is -log(||x_i - x_j||).
            # Gradient w.r.t x_i is -(x_i - x_j)/||x_i - x_j||^2.
            # Gradient w.r.t x_j is +(x_i - x_j)/||x_i - x_j||^2.
            # So if we sum over all j != i, we get the total gradient for x_i.
            # Let's verify:
            # E = sum_{i<j} f(x_i, x_j)
            # grad_i E = sum_{j>i} f_x1 + sum_{j<i} f_x2
            # f = -log(||x_i - x_j||)
            # f_x1 = -(x_i - x_j)/||x_i - x_j||^2
            # f_x2 = +(x_i - x_j)/||x_i - x_j||^2
            # So grad_i E = sum_{j>i} [-(x_i - x_j)/d2] + sum_{j<i} [+(x_i - x_j)/d2]
            # This is equivalent to sum_{j!=i} [-(x_i - x_j)/d2] ?
            # Let's check j<i term: +(x_i - x_j)/d2 = -(x_j - x_i)/d2.
            # And for j>i term: -(x_i - x_j)/d2.
            # So yes, grad_i E = sum_{j!=i} [-(x_i - x_j)/||x_i - x_j||^2]
            
            grad[i] += -v / d2
        
        # Project to tangent space: grad_i = (I - x_i x_i^T) grad_i
        grad[i] -= np.dot(grad[i], points[i]) * points[i]
    
    return energy, grad


def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for safety
    
    # 1. Generate initial points using Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_energy = _energy(best_pts)
    
    # 2. Optimize using L-BFGS-B with projected gradient
    # We need to handle the constraint ||x_i|| = 1.
    # L-BFGS-B works on unconstrained variables.
    # Strategy: Parameterize points using 2D angles (theta, phi) or use a projection step.
    # A common trick is to use the gradient of the energy on the sphere and use an optimizer that allows custom gradients.
    # However, standard scipy minimizers don't directly support manifold constraints.
    # We can use the "retraction" method:
    # 1. Compute gradient on the sphere (tangent space).
    # 2. Use an optimizer on the tangent space? No, L-BFGS-B expects Euclidean space.
    # Alternative: Use the fact that we can flatten the manifold locally.
    # Or, simpler: Use `scipy.optimize.minimize` with `method='L-BFGS-B'` and a custom function that:
    #    a) Takes a flat vector of 2N parameters (e.g., theta, phi for each point).
    #    b) Maps them to the sphere.
    #    c) Computes energy.
    #    d) Computes gradient w.r.t the 2N parameters.
    
    # Let's use spherical coordinates (theta, phi) for each point.
    # x = sin(theta)cos(phi), y = sin(theta)sin(phi), z = cos(theta)
    # theta in [0, pi], phi in [0, 2pi]
    
    def flat_to_points(flat):
        # flat: (2N,) -> (N, 3)
        theta = flat[0:N]
        phi = flat[N:2*N]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])
    
    def points_to_flat(pts):
        z = pts[:, 2]
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = np.arctan2(pts[:, 1], pts[:, 0])
        # Handle phi wrap-around if needed, but arctan2 gives [-pi, pi]
        # Ensure continuity? L-BFGS-B might struggle with wrap-around.
        # For now, let's see if it works.
        return np.concatenate([theta, phi])
    
    def obj_grad(flat):
        pts = flat_to_points(flat)
        e, g_sphere = _energy_grad(pts)
        
        # Compute gradient w.r.t flat parameters (theta, phi)
        # g_flat = J^T g_sphere
        # J is the Jacobian of the map (theta, phi) -> (x, y, z)
        # dx/dtheta = cos(theta)cos(phi)
        # dx/dphi = -sin(theta)sin(phi)
        # dy/dtheta = cos(theta)sin(phi)
        # dy/dphi = sin(theta)cos(phi)
        # dz/dtheta = -sin(theta)
        # dz/dphi = 0
        
        theta = flat[0:N]
        phi = flat[N:2*N]
        
        g_flat = np.zeros_like(flat)
        
        for i in range(N):
            t = theta[i]
            p = phi[i]
            
            # Jacobian row for point i
            # J_i = [[cos(t)cos(p), -sin(t)sin(p)],
            #        [cos(t)sin(p),  sin(t)cos(p)],
            #        [-sin(t),         0        ]]
            
            # g_flat[i] = dot(g_sphere[i], J_i[:, 0])
            # g_flat[i+N] = dot(g_sphere[i], J_i[:, 1])
            
            g_flat[i] = (g_sphere[i, 0] * np.cos(t) * np.cos(p) +
                         g_sphere[i, 1] * np.cos(t) * np.sin(p) +
                         g_sphere[i, 2] * (-np.sin(t)))
            
            g_flat[i + N] = (g_sphere[i, 0] * (-np.sin(t) * np.sin(p)) +
                             g_sphere[i, 1] * (np.sin(t) * np.cos(p)))
        
        return e, g_flat
    
    # Initial flat parameters
    flat0 = points_to_flat(pts)
    
    # Use L-BFGS-B
    # We need to be careful with the phi wrap-around.
    # If phi crosses -pi or pi, the function is discontinuous.
    # A better parameterization might be using quaternions or SO(3) rotations, but that's complex.
    # Let's try L-BFGS-B first and see if it converges.
    
    # To handle the time limit, we'll run the optimization in a loop, checking time.
    # Since `minimize` is a single call, we can't easily interrupt it.
    # We can use `callback` to check time and raise an exception or stop.
    
    stop_flag = [False]
    
    def callback(xk):
        if time.time() - start_time > deadline - 5.0:
            stop_flag[0] = True
            raise RuntimeError("Time limit reached")
        return stop_flag[0]
    
    try:
        res = minimize(obj_grad, flat0, jac=True, method='L-BFGS-B',
                       options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 100000},
                       callback=callback)
        if res.success or hasattr(res, 'x'):
            new_pts = flat_to_points(res.x)
            new_energy = _energy(new_pts)
            if new_energy < best_energy:
                best_pts = new_pts
                best_energy = new_energy
    except RuntimeError:
        pass
    except Exception as e:
        pass
    
    # If we have time left, do a second pass with a different starting point or more iterations?
    # Or just polish the current best.
    
    # Let's try a few more iterations on the best point found so far, using a tighter tolerance.
    if time.time() - start_time < deadline - 10.0:
        flat_best = points_to_flat(best_pts)
        stop_flag[0] = False
        try:
            res2 = minimize(obj_grad, flat_best, jac=True, method='L-BFGS-B',
                            options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 100000},
                            callback=callback)
            if res2.success or hasattr(res2, 'x'):
                new_pts = flat_to_points(res2.x)
                new_energy = _energy(new_pts)
                if new_energy < best_energy:
                    best_pts = new_pts
                    best_energy = new_energy
        except RuntimeError:
            pass
        except Exception as e:
            pass
    
    # Final check: ensure points are on the unit sphere and distinct
    # Normalize
    norms = np.linalg.norm(best_pts, axis=1)
    best_pts = best_pts / norms[:, np.newaxis]
    
    return best_pts
# EVOLVE-BLOCK-END
