"""Physics-based optimization for 200 equal circles in a unit square."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
DEADLINE = 145.0  # safety margin before 150s kill


def _hexagonal_init(n_points, square_half=0.5):
    """Generate a hexagonal lattice initialization for n_points in [-0.5, 0.5]^2."""
    # Hexagonal spacing: for a hex lattice, row spacing = s*sqrt(3)/2, col spacing = s
    # We want to fit ~n_points in the square. Estimate s from area.
    # Area per hex cell = s^2 * sqrt(3)/2
    # n_points * s^2 * sqrt(3)/2 ≈ 1 (unit square area)
    s_est = np.sqrt(2.0 / (n_points * np.sqrt(3)))
    
    # Try a range of spacings and pick the one that fits best
    best_pts = None
    best_count = 0
    
    for s_mult in np.linspace(0.8, 1.3, 20):
        s = s_est * s_mult
        row_spacing = s * np.sqrt(3) / 2
        # Number of rows and columns
        n_rows = int(np.floor((2 * square_half) / row_spacing)) + 1
        n_cols = int(np.floor((2 * square_half) / s)) + 1
        
        pts = []
        for i in range(n_rows):
            y = -square_half + i * row_spacing
            offset = (s / 2) if (i % 2 == 1) else 0
            for j in range(n_cols):
                x = -square_half + j * s + offset
                if -square_half <= x <= square_half and -square_half <= y <= square_half:
                    pts.append((x, y))
        
        pts = np.array(pts)
        if len(pts) >= n_points:
            # Take first n_points (or random subset for diversity)
            if len(pts) > n_points:
                idx = np.random.choice(len(pts), n_points, replace=False)
                pts = pts[idx]
            if len(pts) > best_count:
                best_count = len(pts)
                best_pts = pts[:n_points]
        elif best_pts is None:
            best_pts = pts[:n_points] if len(pts) >= n_points else None
    
    if best_pts is None or len(best_pts) < n_points:
        # Fallback: grid
        m = int(np.ceil(np.sqrt(n_points)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:n_points]
    
    return best_pts.astype(np.float64)


def _compute_radius(pts):
    """Compute the radius for a given set of points."""
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Wall distances
    wall_dists = np.minimum(
        0.5 - np.abs(pts[:, 0]),
        0.5 - np.abs(pts[:, 1])
    )
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)


def _energy(pts_flat, n):
    """Repulsive potential energy to minimize."""
    pts = pts_flat.reshape(n, 2)
    
    # Pairwise repulsion: sum of 1/d^2 for all pairs
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)  # avoid division by zero
    
    # Use 1/d^2 potential
    pair_energy = np.sum(1.0 / dists_sq)
    
    # Wall penalty: penalize if any point is too close to walls
    # Use a smooth penalty: (0.5 - |x|)^(-2) style, but that's singular
    # Better: penalty = sum of 1/(0.5 - |x_i|)^2 for each coordinate, clamped
    x = pts[:, 0]
    y = pts[:, 1]
    
    # Distance to walls (positive means inside)
    wx = 0.5 - np.abs(x)
    wy = 0.5 - np.abs(y)
    
    # Clamp to avoid singularity
    wx = np.maximum(wx, 0.001)
    wy = np.maximum(wy, 0.001)
    
    wall_energy = np.sum(1.0 / wx**2) + np.sum(1.0 / wy**2)
    
    # Weight wall penalty relative to pair penalty
    # We want wall and pair distances to be balanced
    total_energy = pair_energy + 10.0 * wall_energy
    
    return total_energy


def _energy_grad(pts_flat, n):
    """Gradient of the energy function."""
    pts = pts_flat.reshape(n, 2)
    grad = np.zeros_like(pts)
    
    # Pairwise: d/dx_i (1/d_ij^2) = -2*(x_i - x_j) / d_ij^4
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # grad_i = sum_j -2 * diff_ij / dists_sq_ij^2
    # diff_ij = pts_i - pts_j
    inv_d4 = 1.0 / (dists_sq**2)
    np.fill_diagonal(inv_d4, 0.0)
    
    # grad_pair[i] = -2 * sum_j (pts_i - pts_j) / dists_sq_ij^2
    grad_pair = -2.0 * np.sum(diff * inv_d4[:, :, np.newaxis], axis=1)
    
    # Wall gradient
    x = pts[:, 0]
    y = pts[:, 1]
    wx = 0.5 - np.abs(x)
    wy = 0.5 - np.abs(y)
    wx = np.maximum(wx, 0.001)
    wy = np.maximum(wy, 0.001)
    
    # d/dx (1/wx^2) = -2/wx^3 * d(wx)/dx
    # wx = 0.5 - |x|, d(wx)/dx = -sign(x)
    dwx_dx = -np.sign(x)
    dwy_dy = -np.sign(y)
    
    grad_wall_x = -2.0 / wx**3 * dwx_dx
    grad_wall_y = -2.0 / wy**3 * dwy_dy
    
    grad[:, 0] += grad_pair[:, 0] + 10.0 * grad_wall_x
    grad[:, 1] += grad_pair[:, 1] + 10.0 * grad_wall_y
    
    return grad.ravel()


def construct():
    start_time = time.time()
    
    # Initial hexagonal configuration
    np.random.seed(42)
    pts = _hexagonal_init(N)
    
    # Ensure valid: clip to inside square with small margin
    pts[:, 0] = np.clip(pts[:, 0], -0.499, 0.499)
    pts[:, 1] = np.clip(pts[:, 1], -0.499, 0.499)
    
    best_radius = _compute_radius(pts)
    best_pts = pts.copy()
    
    # L-BFGS-B optimization
    x0 = pts.ravel()
    
    # Bounds: keep points inside [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Check time budget
    elapsed = time.time() - start_time
    if elapsed > 5.0:
        return best_pts
    
    try:
        # Use L-BFGS-B with our custom energy
        # maxiter and maxfun to control runtime
        result = minimize(
            fun=_energy,
            x0=x0,
            jac=lambda x: _energy_grad(x, N),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 500,
                'maxfun': 5000,
                'ftol': 1e-10,
                'gtol': 1e-8,
            }
        )
        
        if result.x is not None:
            new_pts = result.x.reshape(N, 2)
            # Clip to valid region
            new_pts[:, 0] = np.clip(new_pts[:, 0], -0.4999, 0.4999)
            new_pts[:, 1] = np.clip(new_pts[:, 1], -0.4999, 0.4999)
            
            new_radius = _compute_radius(new_pts)
            if new_radius > best_radius:
                best_radius = new_radius
                best_pts = new_pts
    except Exception:
        pass
    
    # Second pass: refine with a different potential weighting
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20.0:
        try:
            # Try with stronger wall weighting
            def _energy2(pts_flat, n):
                pts = pts_flat.reshape(n, 2)
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists_sq = np.sum(diff**2, axis=-1)
                np.fill_diagonal(dists_sq, 1.0)
                pair_energy = np.sum(1.0 / dists_sq)
                
                x = pts[:, 0]
                y = pts[:, 1]
                wx = np.maximum(0.5 - np.abs(x), 0.001)
                wy = np.maximum(0.5 - np.abs(y), 0.001)
                wall_energy = np.sum(1.0 / wx**2) + np.sum(1.0 / wy**2)
                
                return pair_energy + 50.0 * wall_energy
            
            x0 = best_pts.ravel()
            result2 = minimize(
                fun=_energy2,
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 300,
                    'maxfun': 3000,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                }
            )
            
            if result2.x is not None:
                new_pts = result2.x.reshape(N, 2)
                new_pts[:, 0] = np.clip(new_pts[:, 0], -0.4999, 0.4999)
                new_pts[:, 1] = np.clip(new_pts[:, 1], -0.4999, 0.4999)
                
                new_radius = _compute_radius(new_pts)
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_pts = new_pts
        except Exception:
            pass
    
    # Final: ensure validity
    best_pts[:, 0] = np.clip(best_pts[:, 0], -0.5, 0.5)
    best_pts[:, 1] = np.clip(best_pts[:, 1], -0.5, 0.5)
    
    return best_pts
