"""Pack 50 equal circles in unit circle using smooth L-BFGS-B with multi-seed restarts."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = time.time() + 135.0  # safety margin out of 150s


def compute_radius(points):
    """Compute the packing radius: min of boundary clearance and half nearest-neighbor distance."""
    points = np.asarray(points, dtype=np.float64)
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_pair_dist = np.sqrt(np.min(sq_dists))
    return min(boundary_dists.min(), min_pair_dist / 2.0)


def seed_hexagonal():
    """Hexagonal lattice seed, greedily select N points maximizing min spacing."""
    spacing = 0.2
    points_list = []
    for j in range(-6, 7):
        for i in range(-6, 7):
            x = i * spacing + (0.5 * spacing if j % 2 else 0)
            y = j * spacing * np.sqrt(3) / 2
            if np.sqrt(x * x + y * y) < 1.0:
                points_list.append([x, y])
    points = np.array(points_list)
    if len(points) < N:
        # Fallback: uniform random
        return np.random.uniform(-0.9, 0.9, size=(N, 2))
    # Greedy farthest-point selection
    dists_from_center = np.linalg.norm(points, axis=1)
    start_idx = np.argmin(dists_from_center)
    selected = [points[start_idx]]
    while len(selected) < N:
        sel = np.array(selected)
        min_dists = np.min(np.linalg.norm(points[:, np.newaxis, :] - sel[np.newaxis, :, :], axis=2), axis=1)
        selected.append(points[np.argmax(min_dists)])
    return np.array(selected)


def seed_ring_layers():
    """Concentric ring seed: rings at increasing radii with points spaced around."""
    # Try to fit 50 points in rings
    # Ring 0: 1 point at center
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Ring 3: 18 points
    # Ring 4: 13 points
    # Total: 50
    rings = [(0, 1), (0.35, 6), (0.6, 12), (0.82, 18), (0.95, 13)]
    points = []
    for radius, count in rings:
        for k in range(count):
            angle = 2 * np.pi * k / count
            points.append([radius * np.cos(angle), radius * np.sin(angle)])
    return np.array(points[:N])


def seed_random_valid():
    """Generate a random valid packing with minimum radius > 0.1."""
    for attempt in range(200):
        points = np.random.uniform(-1, 1, size=(N, 2))
        norms = np.linalg.norm(points, axis=1)
        max_norm = norms.max()
        if max_norm > 0:
            points *= (0.95 / max_norm)
        r = compute_radius(points)
        if r > 0.1:
            return points
    # Fallback: scaled random
    points = np.random.uniform(-1, 1, size=(N, 2))
    norms = np.linalg.norm(points, axis=1)
    max_norm = norms.max()
    if max_norm > 0:
        points *= (0.9 / max_norm)
    return points


def smooth_optimize(seed_points, beta=100.0, max_iter=3000):
    """Optimize using smooth log-sum-exp soft-min objective with L-BFGS-B."""
    
    def objective(x):
        points = x.reshape(N, 2)
        boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff * diff, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        dists = np.sqrt(sq_dists)
        
        # Smooth min for boundary
        b_min = -np.log(np.sum(np.exp(-beta * boundary_dists))) / beta
        # Smooth min for pairs (divide by 2 for radius)
        d_min = -np.log(np.sum(np.exp(-beta * dists))) / beta / 2.0
        # Soft-min of the two
        r = -np.log(np.exp(-beta * b_min) + np.exp(-beta * d_min)) / beta
        return -r
    
    def jac(x):
        points = x.reshape(N, 2)
        boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
        norms = np.linalg.norm(points, axis=1)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff * diff, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        dists = np.sqrt(sq_dists)
        
        b_min = -np.log(np.sum(np.exp(-beta * boundary_dists))) / beta
        d_min = -np.log(np.sum(np.exp(-beta * dists))) / beta / 2.0
        
        eb = np.exp(-beta * b_min)
        ed = np.exp(-beta * d_min)
        denom = eb + ed
        wb = eb / denom
        wd = ed / denom
        
        # Gradient of b_min
        gb = np.zeros_like(points)
        exp_b = np.exp(-beta * boundary_dists)
        sum_exp_b = np.sum(exp_b)
        for i in range(N):
            if norms[i] > 1e-10:
                gb[i] = wb * (-beta * exp_b[i] / sum_exp_b) * (-points[i] / norms[i])
        
        # Gradient of d_min
        gd = np.zeros_like(points)
        exp_d = np.exp(-beta * dists)
        sum_exp_d = np.sum(exp_d)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if dists[i, j] > 1e-10:
                    grad_dij = -beta * exp_d[i, j] / sum_exp_d
                    gd[i] += wd * grad_dij * (points[i] - points[j]) / dists[i, j]
        
        return (gb + gd).flatten()
    
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    result = minimize(
        objective,
        seed_points.flatten(),
        jac=jac,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'maxfun': max_iter * 3, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    
    optimized = result.x.reshape(N, 2)
    # Ensure strictly inside
    norms = np.linalg.norm(optimized, axis=1)
    too_far = norms >= 1.0
    if np.any(too_far):
        optimized[too_far] *= 0.999 / norms[too_far]
    
    return optimized


def gradient_polish(points, n_steps=500, lr=0.005):
    """Fast gradient-based polish using subgradient of the min function."""
    points = points.copy()
    current_r = compute_radius(points)
    
    for step in range(n_steps):
        if time.time() > DEADLINE - 5.0:
            break
        
        # Compute which constraints are active
        boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        sq_dists = np.sum(diff * diff, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        dists = np.sqrt(sq_dists)
        
        b_min = boundary_dists.min()
        d_min = dists.min() / 2.0
        r = min(b_min, d_min)
        
        if r <= current_r - 1e-10:
            # Move back if worse
            continue
        
        # Build gradient: push points away from active constraints
        grad = np.zeros_like(points)
        
        # Boundary constraint: push inward
        for i in range(N):
            if boundary_dists[i] < r + 1e-6:
                norm_i = np.linalg.norm(points[i])
                if norm_i > 1e-10:
                    grad[i] -= points[i] / norm_i
        
        # Pair constraints: push apart
        for i in range(N):
            for j in range(i + 1, N):
                if dists[i, j] / 2.0 < r + 1e-6:
                    d = points[i] - points[j]
                    dist = np.linalg.norm(d)
                    if dist > 1e-10:
                        d_hat = d / dist
                        grad[i] += d_hat
                        grad[j] -= d_hat
        
        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-10:
            grad /= gnorm
        
        # Try moving
        new_points = points + lr * grad
        # Ensure inside
        new_norms = np.linalg.norm(new_points, axis=1)
        outside = new_norms >= 1.0
        if np.any(outside):
            new_points[outside] *= 0.999 / new_norms[outside]
        
        new_r = compute_radius(new_points)
        if new_r > current_r:
            points = new_points
            current_r = new_r
        else:
            # Reduce learning rate
            lr *= 0.95
    
    return points, current_r


def construct():
    """Main entry point: multi-seed smooth optimization with gradient polish."""
    np.random.seed(42)
    
    # Generate diverse seeds
    seeds = []
    
    # Seed 1: Hexagonal lattice
    try:
        seeds.append(seed_hexagonal())
    except Exception:
        pass
    
    # Seed 2: Ring layers
    try:
        seeds.append(seed_ring_layers())
    except Exception:
        pass
    
    # Seed 3: Random valid
    try:
        seeds.append(seed_random_valid())
    except Exception:
        pass
    
    # Add perturbed versions of hexagonal
    hex_seed = seeds[0] if seeds else np.random.uniform(-0.9, 0.9, size=(N, 2))
    for noise_level in [0.02, 0.05, 0.1]:
        perturbed = hex_seed + np.random.normal(0, noise_level, size=(N, 2))
        norms = np.linalg.norm(perturbed, axis=1)
        outside = norms >= 1.0
        if np.any(outside):
            perturbed[outside] *= 0.999 / norms[outside]
        seeds.append(perturbed)
    
    # Track best
    best_points = seeds[0]
    best_r = compute_radius(best_points)
    
    # Optimize each seed
    for idx, seed in enumerate(seeds):
        if time.time() > DEADLINE - 20.0:
            break
        
        # Smooth L-BFGS-B optimization
        try:
            optimized = smooth_optimize(seed, beta=100.0, max_iter=2000)
        except Exception:
            optimized = seed
        
        r_opt = compute_radius(optimized)
        
        # Gradient polish
        polished, r_polished = gradient_polish(optimized, n_steps=300, lr=0.005)
        
        if r_polished > best_r:
            best_r = r_polished
            best_points = polished
        
        if r_opt > best_r:
            best_r = r_opt
            best_points = optimized
    
    # Final polish on best
    if time.time() < DEADLINE - 10.0:
        final_points, final_r = gradient_polish(best_points, n_steps=500, lr=0.003)
        if final_r > best_r:
            best_r = final_r
            best_points = final_points
    
    # Ensure validity
    norms = np.linalg.norm(best_points, axis=1)
    outside = norms >= 1.0
    if np.any(outside):
        best_points[outside] *= 0.999 / norms[outside]
    
    # Verify no coincident points
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    if np.min(sq_dists) < 1e-20:
        # Add tiny perturbation
        best_points += np.random.normal(0, 1e-6, size=(N, 2))
        norms = np.linalg.norm(best_points, axis=1)
        outside = norms >= 1.0
        if np.any(outside):
            best_points[outside] *= 0.999 / norms[outside]
    
    return best_points
