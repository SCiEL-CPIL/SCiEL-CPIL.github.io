"""Rewritten program: Riesz s-energy + L-BFGS-B for sphere packing in a unit cube."""
N = 100

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _pairwise_distances_flat(x):
    """x is shape (N, 3). Returns a vector of all pairwise distances (N*(N-1)/2,) and indices."""
    n = len(x)
    # Compute all pairwise differences efficiently using broadcasting
    # To save memory, we can do it in chunks, but for N=100 it's fine: 100x100x3 = 30000 floats
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (N, N)
    # Get upper triangle indices
    iu = np.triu_indices(n, k=1)
    return dists[iu], iu


def _energy_and_grad(x, s=8.0, boundary_softness=100.0):
    """
    Compute Riesz s-energy and its gradient.
    Energy = sum_{i<j} 1/d_ij^s  +  boundary_penalty
    We minimize this. Lower energy -> larger minimum distance.
    """
    n = len(x)
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, np.inf)
    dists = np.sqrt(dists2)
    
    # Riesz energy: sum over i<j of 1/d^s
    # We only need upper triangle
    iu = np.triu_indices(n, k=1)
    d_pair = dists[iu]
    
    # Avoid division by zero
    d_pair = np.maximum(d_pair, 1e-15)
    
    energy_pairs = np.sum(d_pair ** (-s))
    
    # Boundary penalty: for each coordinate, if x > 0.5 or x < -0.5, penalize
    # Use a soft penalty: sum of max(0, |x| - 0.5 + margin)^2
    # Actually, let's use a smooth penalty that pushes points inward.
    # Boundary constraint: |x_k| <= 0.5
    # Penalty: sum_i sum_k max(0, |x_ik| - 0.5 + eps)^2 * strength
    # For smoothness, use: sum (max(0, |x|-0.5+eps))^2
    
    # Let's use a simpler approach: penalize any point outside [-0.5, 0.5]
    # with a quadratic barrier
    boundary_violation = np.maximum(0.0, np.abs(x) - 0.5)
    boundary_penalty = np.sum(boundary_violation ** 2) * boundary_softness
    
    total_energy = energy_pairs + boundary_penalty
    
    # Gradient
    # d(energy_pairs)/dx_i = sum_{j != i} -s * (x_i - x_j) / |x_i - x_j|^(s+2)
    grad = np.zeros_like(x)
    
    # For each pair (i, j), the contribution to grad[i] is -s * (x_i - x_j) / d_ij^(s+2)
    # and to grad[j] is s * (x_i - x_j) / d_ij^(s+2)
    
    # We can compute this vectorized
    # diff is (N, N, 3)
    # dists is (N, N)
    
    # Gradient of sum_{i<j} 1/d_ij^s w.r.t. x_i:
    # = sum_{j != i} -s * (x_i - x_j) * d_ij^(-s-2)
    
    # Let's compute the force matrix
    # F[i, j] = -s * (x_i - x_j) * d_ij^(-s-2) for i != j
    # Then grad[i] = sum_{j != i} F[i, j]
    
    # d_ij^(-s-2) = d_ij^(-s) * d_ij^(-2)
    # We already have d_pair for upper triangle. Let's compute the full matrix version.
    
    inv_d_pow = dists ** (-(s + 2))
    np.fill_diagonal(inv_d_pow, 0.0)
    
    # Force on i from j: -s * (x_i - x_j) * inv_d_pow[i, j]
    # grad[i] = sum_j -s * (x_i - x_j) * inv_d_pow[i, j]
    
    # Vectorized:
    # diff is (N, N, 3), inv_d_pow is (N, N)
    # Multiply diff by inv_d_pow[:, :, np.newaxis]
    weighted_diff = diff * inv_d_pow[:, :, np.newaxis]
    
    # grad[i] = -s * sum_j weighted_diff[i, j]
    grad_pairs = -s * np.sum(weighted_diff, axis=1)
    
    # Boundary gradient
    # d(boundary_penalty)/dx_ik = 2 * boundary_softness * max(0, |x_ik| - 0.5) * sign(x_ik)
    # But max(0, |x|-0.5) is 0 if |x| <= 0.5, so we only need to handle violations.
    # For smoothness, this is fine since it's 0 in the interior.
    
    boundary_grad = np.zeros_like(x)
    mask = np.abs(x) > 0.5
    boundary_grad[mask] = 2.0 * boundary_softness * boundary_violation[mask] * np.sign(x[mask])
    
    total_grad = grad_pairs + boundary_grad
    
    return total_energy, total_grad


def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin before 150s limit
    
    # Initial configuration: simple cubic grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    x0 = grid[:N].copy()
    
    best_x = x0.copy()
    best_radius = _compute_radius(best_x)
    
    # Try L-BFGS-B optimization
    # We want to minimize the energy. Lower energy -> better packing.
    # But we also need to ensure points stay in the cube.
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Flatten x0
    x0_flat = x0.flatten()
    
    try:
        # Use L-BFGS-B
        res = minimize(
            _energy_flat,
            x0_flat,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000}
        )
        
        if res.success or res.fun < _energy_flat(x0_flat)[0]:
            x_opt = res.x.reshape(N, 3)
            r_opt = _compute_radius(x_opt)
            if r_opt > best_radius:
                best_x = x_opt
                best_radius = r_opt
        
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10.0:
        return best_x
    
    # Try a few different s values for Riesz energy
    for s_val in [6.0, 10.0, 12.0]:
        if time.time() > deadline - 15.0:
            break
        
        try:
            res = minimize(
                lambda x: _energy_flat_with_s(x, s_val),
                x0_flat,
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 6000}
            )
            
            if res.success or True:
                x_opt = res.x.reshape(N, 3)
                r_opt = _compute_radius(x_opt)
                if r_opt > best_radius:
                    best_x = x_opt
                    best_radius = r_opt
                    
        except Exception:
            pass
    
    # Final local refinement using a different approach: gradient descent on min-distance
    if time.time() < deadline - 20.0:
        try:
            x_curr = best_x.copy()
            for _ in range(200):
                if time.time() > deadline - 5.0:
                    break
                x_new = _local_refinement_step(x_curr, lr=0.001)
                r_new = _compute_radius(x_new)
                if r_new > best_radius:
                    best_x = x_new
                    best_radius = r_new
                    x_curr = x_new
        except Exception:
            pass
    
    return best_x


def _compute_radius(x):
    """Compute the maximum feasible radius for configuration x."""
    n = len(x)
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, np.inf)
    min_dist = np.sqrt(np.min(dists2))
    
    # Distance to boundary
    dist_to_boundary = np.min(0.5 - np.abs(x), axis=1)
    min_boundary = np.min(dist_to_boundary)
    
    return min(min_dist / 2.0, min_boundary)


def _energy_flat(x_flat, s=8.0, boundary_softness=100.0):
    """Flat version of energy function for scipy.optimize."""
    x = x_flat.reshape(N, 3)
    energy, grad = _energy_and_grad(x, s=s, boundary_softness=boundary_softness)
    return energy, grad.flatten()


def _energy_flat_with_s(x_flat, s):
    """Energy function with fixed s."""
    x = x_flat.reshape(N, 3)
    energy, grad = _energy_and_grad(x, s=s, boundary_softness=100.0)
    return energy, grad.flatten()


def _local_refinement_step(x, lr=0.001):
    """One step of gradient descent on a smoothed min-distance objective."""
    n = len(x)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :]
    dists2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists2, np.inf)
    dists = np.sqrt(dists2)
    
    # Smooth minimum: use -log-sum-exp approximation
    # min(d) ≈ -1/k * log(sum(exp(-k*d)))
    k = 50.0
    # For numerical stability, shift by min
    d_min = np.min(dists)
    log_sum = np.log(np.sum(np.exp(-k * (dists - d_min))))
    smooth_min = d_min - np.log(log_sum) / k
    
    # Gradient of smooth min w.r.t. x_i
    # This is complex, so let's use a simpler approach:
    # Move each point away from its nearest neighbor
    
    grad = np.zeros_like(x)
    for i in range(n):
        # Find nearest neighbor
        d_to_i = dists[i, :].copy()
        d_to_i[i] = np.inf
        j = np.argmin(d_to_i)
        if dists[i, j] > 1e-15:
            direction = (x[i] - x[j]) / dists[i, j]
            grad[i] += direction
    
    # Boundary force
    for i in range(n):
        for k in range(3):
            if x[i, k] > 0.45:
                grad[i, k] -= 0.1
            elif x[i, k] < -0.45:
                grad[i, k] += 0.1
    
    x_new = x - lr * grad
    
    # Project back to cube
    x_new = np.clip(x_new, -0.5, 0.5)
    
    return x_new
# EVOLVE-BLOCK-END
