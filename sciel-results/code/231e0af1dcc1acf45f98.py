"""Improved packing: multi-start L-BFGS-B on a smooth repulsion energy with deadline guard."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
# Safety margin: the evaluator kills at 150 s wall-clock; we stop at 140 s.
DEADLINE_SECONDS = 140.0


def _score(positions):
    """Compute the objective: min over all pairs of half-distance and over all
    spheres of distance to the nearest wall. This is the common radius r."""
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    # Distance to walls: for each coordinate, distance to -0.5 or +0.5
    # min_i min_k (1/2 - |c_ik|)
    wall_dists = 0.5 - np.abs(positions)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)


def _energy_and_grad(positions):
    """
    Smooth repulsion energy that is maximized when the minimum radius is large.
    We minimize the negative of a smooth approximation to the minimum radius.
    
    Using a soft-min (log-sum-exp) approximation:
    E = -log(sum(exp(-beta * d_ij))) / beta  for pairs, similarly for walls.
    
    Actually, a simpler and more robust approach: minimize the sum of inverse
    powers of distances, which pushes all distances up. But that doesn't directly
    maximize the minimum.
    
    Better: use the fact that maximizing min(d_ij) is equivalent to minimizing
    a smooth function that is dominated by the smallest distance.
    
    We'll use: f = sum over pairs of (1/d_ij)^p + sum over walls of (1/wall_i)^p
    for large p, this is dominated by the smallest distance.
    """
    p = 8  # power; higher = more focus on minimum
    # Pairwise
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    dists = np.sqrt(dists_sq)
    
    # Energy from pairs: sum of (1/d)^p
    # Gradient: d/dd (1/d)^p = -p / d^(p+1)
    # d/dx_i (1/d_ij)^p = -p * (1/d_ij)^(p+1) * (x_i - x_j) / d_ij
    #                    = -p * (x_i - x_j) / d_ij^(p+2)
    
    inv_d_pow = np.zeros_like(dists)
    mask = dists > 1e-12
    inv_d_pow[mask] = dists[mask] ** (-(p + 2))
    
    # For each i, grad_i = -p * sum_{j != i} (x_i - x_j) * inv_d_pow[i,j]
    # Note: (x_i - x_j) is the vector from j to i
    # sum_j (x_i - x_j) * inv_d_pow[i,j] = sum_j (x_i - x_j) * inv_d_pow[i,j]
    
    # (x_i - x_j) for all i,j is: positions[:,np.newaxis,:] - positions[np.newaxis,:,:]
    # We already have diff = positions[:,np.newaxis,:] - positions[np.newaxis,:,:]
    # So grad_i = -p * sum_j diff[i,j] * inv_d_pow[i,j]
    
    grad_pairs = -p * np.sum(diff * inv_d_pow[:, :, np.newaxis], axis=1)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(positions)
    # Energy from walls: sum of (1/wall_i)^p for each coordinate
    # But wall_dists can be negative if outside; we need to handle that.
    # For the gradient, if wall_dist <= 0, the sphere is outside or on boundary.
    # We want to push it inside.
    
    # Let's compute wall energy and gradient carefully.
    # For a sphere at position c, the distance to the nearest wall is min over k of (0.5 - |c_k|).
    # This is not differentiable at |c_k| = 0.5 or c_k = 0.
    
    # Simpler: use the sum of (1/(0.5 - |c_k|))^p for all k, but this blows up near walls.
    # Actually, let's use a different approach for walls: just penalize being close to walls.
    
    # Alternative: use the same repulsion idea but with virtual spheres outside the cube.
    # That's more complex. Let's stick with a direct penalty.
    
    # For walls: we want to maximize min_i min_k (0.5 - |c_ik|)
    # Equivalent to minimizing sum of (1/(0.5 - |c_ik|))^p for all i,k where 0.5 - |c_ik| > 0
    # If any is <= 0, we have a large penalty.
    
    # Let's compute:
    wall_energy = 0.0
    grad_walls = np.zeros_like(positions)
    
    for i in range(N):
        for k in range(3):
            c = positions[i, k]
            w = 0.5 - abs(c)
            if w <= 1e-10:
                # Sphere is on or outside the wall; push it back
                # Gradient direction: toward center
                if c >= 0:
                    grad_walls[i, k] -= 1e6  # push left (decrease c)
                else:
                    grad_walls[i, k] += 1e6  # push right (increase c)
                wall_energy += 1e12
            else:
                wall_energy += w ** (-p)
                # d/dc (1/w)^p where w = 0.5 - |c|
                # If c > 0: w = 0.5 - c, dw/dc = -1, d/dc (1/w)^p = -p * (1/w)^(p+1) * (-1) = p * w^(-p-1)
                # If c < 0: w = 0.5 + c, dw/dc = 1, d/dc (1/w)^p = -p * (1/w)^(p+1) * 1 = -p * w^(-p-1)
                # So grad = +/- p * w^(-p-1)
                if c >= 0:
                    grad_walls[i, k] += p * w ** (-(p + 1))
                else:
                    grad_walls[i, k] -= p * w ** (-(p + 1))
    
    total_energy = np.sum(1.0 / (dists[mask] ** p)) + wall_energy
    total_grad = grad_pairs + grad_walls
    
    return total_energy, total_grad


def _make_initialization(seed):
    """Create a random initial configuration with some structure."""
    rng = np.random.RandomState(seed)
    
    # Start with a grid and add random perturbation
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    positions = grid[:N].copy()
    
    # Add random perturbation
    positions += rng.uniform(-0.05, 0.05, size=positions.shape)
    
    # Clip to stay inside
    positions = np.clip(positions, -0.45, 0.45)
    
    return positions


def _local_optimize(positions, deadline, seed):
    """Run L-BFGS-B to minimize the repulsion energy."""
    try:
        result = minimize(
            _energy_and_grad,
            positions,
            jac=True,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (N * 3),
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.success or result.fun < 1e6:
            return result.x
    except Exception:
        pass
    return positions


def construct():
    start_time = time.time()
    
    best_positions = None
    best_score = -np.inf
    
    # Try multiple initializations
    for seed in range(50):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 10:
            break
        
        positions = _make_initialization(seed)
        score = _score(positions)
        
        if score > best_score:
            best_score = score
            best_positions = positions.copy()
        
        # Local optimization
        elapsed = time.time() - start_time
        if elapsed > DEADLINE_SECONDS - 20:
            break
        
        optimized = _local_optimize(positions, DEADLINE_SECONDS, seed)
        opt_score = _score(optimized)
        
        if opt_score > best_score:
            best_score = opt_score
            best_positions = optimized.copy()
        
        # Check if we're close to the known best
        if best_score > 0.1113:
            break
    
    # Final check
    if best_positions is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_positions = grid[:N]
    
    # Ensure finite and valid
    if not np.all(np.isfinite(best_positions)):
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_positions = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    
    return best_positions
