"""
Solver for packing 50 equal spheres in a unit cube [-0.5, 0.5]^3.
Strategy: Multi-stage optimization combining:
1. Structured initial configurations (FCC-inspired, grid, random, shell-core)
2. Repulsion-based simulated annealing for global search
3. Softmin gradient polishing (L-BFGS-B) for fine-tuning
4. Active-set local search on binding constraints
5. Final multi-round polish
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50

# Precompute upper triangle indices for pairwise distances
_iu, _ju = np.triu_indices(N, k=1)
_PAIR_IDX = np.column_stack([_iu, _ju])
_NUM_PAIRS = len(_iu)

def _seed_fcc():
    """Generate an FCC-inspired initial configuration for 50 spheres."""
    m = 4
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                for dx, dy, dz in [(0, 0, 0),
                                    (0.5, 0.5, 0),
                                    (0.5, 0, 0.5),
                                    (0, 0.5, 0.5)]:
                    px = x + dx * spacing
                    py = y + dy * spacing
                    pz = z + dz * spacing
                    if -0.5 <= px <= 0.5 and -0.5 <= py <= 0.5 and -0.5 <= pz <= 0.5:
                        points.append([px, py, pz])
    
    points = np.array(points)
    if len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    
    return points

def _seed_grid():
    """Generate a grid-based initial configuration."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    idx = np.random.choice(len(grid), N, replace=False)
    return grid[idx]

def _seed_random():
    """Generate a random initial configuration."""
    return np.random.uniform(-0.45, 0.45, size=(N, 3))

def _seed_bcc():
    """Generate a BCC-inspired initial configuration."""
    m = 3
    spacing = 1.0 / m
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) * spacing - 0.5
                y = (j + 0.5) * spacing - 0.5
                z = (k + 0.5) * spacing - 0.5
                points.append([x, y, z])
                points.append([x + 0.5*spacing, y + 0.5*spacing, z + 0.5*spacing])
    
    points = np.array(points)
    if len(points) < N:
        extra = np.random.uniform(-0.45, 0.45, size=(N - len(points), 3))
        points = np.vstack([points, extra])
    else:
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    
    return points

def _seed_shell_core():
    """Generate a shell-core initial configuration."""
    n_core = 20
    n_shell = N - n_core
    
    core = np.random.uniform(-0.25, 0.25, size=(n_core, 3))
    
    theta = np.random.uniform(0, 2 * np.pi, size=n_shell)
    phi = np.random.uniform(0, np.pi, size=n_shell)
    r = 0.35 + np.random.uniform(-0.05, 0.05, size=n_shell)
    shell = np.column_stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.sin(phi) * np.sin(theta),
        r * np.cos(phi)
    ])
    
    return np.vstack([core, shell])

def _seed_harmonic():
    """
    Generate positions using low-lying 3D harmonic oscillator eigenstates.
    This gives a natural non-overlapping distribution via Pauli exclusion.
    """
    # Generate positions from the first N eigenstates of 3D harmonic oscillator
    # n = nx + ny + nz, with degeneracy
    positions = []
    # Use a simple approach: sample from the ground state density
    # which is a 3D Gaussian, then jitter to avoid overlap
    np.random.seed(42)
    for n in range(N):
        # Distribute among quantum numbers
        # Use a structured pattern to ensure good coverage
        level = n // 5
        offset = n % 5
        
        # Create a structured position based on level
        if level == 0:
            # Ground state: near center
            base = [0, 0, 0]
        elif level == 1:
            # First excited: along axes
            axes = [[0.15, 0, 0], [0, 0.15, 0], [0, 0, 0.15],
                    [0.1, 0.1, 0], [0.1, 0, 0.1]]
            base = axes[offset % 5]
        elif level == 2:
            base = [0.25 * np.cos(2*np.pi*offset/5),
                   0.25 * np.sin(2*np.pi*offset/5),
                   0.1 * (offset - 2)]
        elif level == 3:
            base = [0.3 * np.cos(2*np.pi*offset/5),
                   0.3 * np.sin(2*np.pi*offset/5),
                   0.2 * (offset - 2)]
        elif level == 4:
            base = [0.35 * np.cos(2*np.pi*offset/5),
                   0.35 * np.sin(2*np.pi*offset/5),
                   0.25 * (offset - 2)]
        else:
            base = [0.4 * np.cos(2*np.pi*offset/5),
                   0.4 * np.sin(2*np.pi*offset/5),
                   0.3 * (offset - 2)]
        
        # Add small random perturbation
        pos = np.array(base) + np.random.normal(0, 0.02, size=3)
        pos = np.clip(pos, -0.45, 0.45)
        positions.append(pos)
    
    return np.array(positions)

def _compute_radius(c):
    """
    Compute the largest common radius for centers c (N, 3) in cube [-0.5, 0.5]^3.
    """
    if len(c) == 0:
        return 0.0
    
    # Pairwise distances using precomputed indices
    diff = c[_iu] - c[_ju]
    dists_sq = np.sum(diff * diff, axis=-1)
    min_pair = np.sqrt(np.min(dists_sq))
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    
    return min(min_pair / 2.0, min_wall)

def _softmin_radius_and_grad(c, t):
    """
    Compute softmin radius and its gradient.
    Returns (radius, gradient) where gradient is (N, 3).
    Uses vectorized accumulation for speed.
    """
    diff = c[_iu] - c[_ju]
    dists_sq = np.sum(diff * diff, axis=-1)
    dists = np.sqrt(dists_sq)
    pair_vals = dists / 2.0
    
    wall_vals = 0.5 - np.abs(c)
    
    # Log-sum-exp for numerical stability
    all_vals = np.concatenate([pair_vals, wall_vals.ravel()])
    m_val = np.min(all_vals)
    
    # Weights for pairs (shifted for stability)
    pair_weights = np.exp(-(pair_vals - m_val) / t)
    
    # Weights for walls (shifted for stability)
    wall_weights = np.exp(-(wall_vals - m_val) / t)
    
    total_sum = np.sum(pair_weights) + np.sum(wall_weights)
    
    if total_sum < 1e-300:
        return m_val, np.zeros_like(c)
        
    # Softmin value
    softmin_val = m_val - t * np.log(total_sum)
    
    # Gradient calculation (vectorized)
    grad_c = np.zeros_like(c)
    
    # Pair contributions
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dists = 1.0 / np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff * inv_dists[:, np.newaxis]
    
    # pair_weights / total_sum gives normalized weights
    norm_pair_w = pair_weights / total_sum
    pair_grad = 0.5 * norm_pair_w[:, np.newaxis] * unit_vecs
    
    # Vectorized accumulation using np.add.at
    np.add.at(grad_c, _iu, -pair_grad)
    np.add.at(grad_c, _ju, pair_grad)
    
    # Wall contributions
    norm_wall_w = wall_weights / total_sum
    signs = np.sign(c)
    grad_c += norm_wall_w * signs
    
    return softmin_val, grad_c

def _repulsion_anneal(c_init, deadline, t_start=2.0, t_end=0.01, iters_per_temp=50):
    """
    Repulsion-based simulated annealing with vectorized force accumulation.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), 15)
    
    for t in temps:
        if time.time() > deadline:
            break
        
        for it in range(iters_per_temp):
            if time.time() > deadline:
                break
            
            # Compute repulsive forces (vectorized)
            forces = np.zeros_like(c)
            
            # Pair repulsion
            diff = c[_iu] - c[_ju]
            dists_sq = np.sum(diff * diff, axis=-1)
            dists = np.sqrt(dists_sq)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                force_mag = 1.0 / np.where(dists_sq > 1e-12, dists_sq, 1.0)
            force_mag = np.clip(force_mag, 0, 1e6)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                inv_dists = 1.0 / np.where(dists > 1e-12, dists, 1.0)
            unit_vecs = diff * inv_dists[:, np.newaxis]
            unit_vecs = np.where(np.isfinite(unit_vecs), unit_vecs, 0.0)
            
            pair_forces = force_mag[:, np.newaxis] * unit_vecs
            
            # Vectorized accumulation
            np.add.at(forces, _iu, pair_forces)
            np.add.at(forces, _ju, -pair_forces)
            
            # Wall repulsion
            wall_dists = 0.5 - np.abs(c)
            with np.errstate(divide='ignore', invalid='ignore'):
                wall_force_mag = 1.0 / np.where(wall_dists > 1e-6, wall_dists * wall_dists, 1.0)
            wall_force_mag = np.clip(wall_force_mag, 0, 1e6)
            wall_force_mag = np.where(np.isfinite(wall_force_mag), wall_force_mag, 0)
            
            signs = np.sign(c)
            forces += wall_force_mag * signs
            
            # Add noise and update
            noise = np.random.normal(0, t * 0.1, size=c.shape)
            c_new = c + forces * 0.01 + noise
            c_new = np.clip(c_new, -0.499999, 0.499999)
            
            # Accept if better or with probability exp(-delta_r / t)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            
            r_curr = _compute_radius(c)
            delta_r = r_new - r_curr
            if delta_r > 0 or np.random.random() < np.exp(delta_r / max(t, 1e-10)):
                c = c_new
    
    return best_c, best_r

def _smooth_polish(c_init, deadline, t_start=0.05, t_end=0.0001, iters_per_temp=100):
    """
    Polish using gradient descent on the softmin radius.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    temps = np.logspace(np.log10(t_start), np.log10(t_end), 12)
    
    for t in temps:
        if time.time() > deadline:
            break
            
        def obj(x):
            r, _ = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return -r
        
        def grad(x):
            _, g = _softmin_radius_and_grad(x.reshape(N, 3), t)
            return g.ravel()
            
        x0 = c.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                obj,
                x0,
                jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            c_new = res.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
                c = c_new.copy()
                
        except Exception:
            continue
            
    return best_c, best_r

def _active_set_polish(c_init, deadline):
    """
    Active-set polish: identify binding constraints and optimize only those points.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Find binding constraints
        diff = c[_iu] - c[_ju]
        dists_sq = np.sum(diff * diff, axis=-1)
        dists = np.sqrt(dists_sq)
        
        wall_dists = 0.5 - np.abs(c)
        r = _compute_radius(c)
        
        # Find pairs closer than 2*r + epsilon
        eps = 1e-8
        tight_pairs = np.where(dists < 2*r + eps)[0]
        
        # Find walls closer than r + epsilon
        tight_walls = []
        for i in range(N):
            for k in range(3):
                if wall_dists[i, k] < r + eps:
                    tight_walls.append((i, k))
        
        # If no tight constraints, try a small random perturbation
        if len(tight_pairs) == 0 and len(tight_walls) == 0:
            c_try = c.copy()
            idx = np.random.randint(N)
            c_try[idx] += np.random.normal(0, 0.01, size=3)
            c_try = np.clip(c_try, -0.499999, 0.499999)
            r_try = _compute_radius(c_try)
            if r_try > best_r:
                best_r = r_try
                best_c = c_try.copy()
                c = c_try.copy()
            continue
        
        # Try moving points involved in tight constraints
        moved = False
        for pi in tight_pairs[:5]:  # Limit to 5 tightest pairs
            i, j = _PAIR_IDX[pi]
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                continue
            vec = vec / norm
            
            for step in [0.005, 0.01, 0.02]:
                c_try = c.copy()
                c_try[i] += step * vec
                c_try[j] -= step * vec
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    moved = True
                    break
            if moved:
                break
        
        for (i, k) in tight_walls[:5]:
            sign = 1.0 if c[i, k] >= 0 else -1.0
            for step in [0.005, 0.01, 0.02]:
                c_try = c.copy()
                c_try[i, k] += step * sign
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    moved = True
                    break
            if moved:
                break
    
    return best_c, best_r

def _local_search(c_init, deadline, max_iters=500):
    """
    Local search: Perturb points that are constraining the radius.
    Tries multiple directions and step sizes.
    """
    c = c_init.copy()
    best_c = c.copy()
    best_r = _compute_radius(c)
    steps = (0.002, 0.005, 0.01, 0.02, 0.03)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        r = _compute_radius(c)
        if r <= best_r:
             c = best_c.copy()
             r = best_r
             
        diff = c[_iu] - c[_ju]
        dists_sq = np.sum(diff * diff, axis=-1)
        dists = np.sqrt(dists_sq)
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        if min_wall_dist < min_pair_dist / 2.0:
            # Move the point closest to a wall away from it
            idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            i, k = idx
            sign = 1.0 if c[i, k] >= 0 else -1.0
            improved = False
            for step in steps:
                c_try = c.copy()
                c_try[i, k] += step * sign
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    improved = True
                    break
            if not improved:
                # Try small random perturbation
                c_try = c.copy()
                c_try[i] += np.random.normal(0, 0.01, size=3)
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
        else:
            idx = np.argmin(dists_sq)
            i, j = _PAIR_IDX[idx]
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-6:
                vec = np.random.normal(size=3)
                norm = np.linalg.norm(vec)
            vec = vec / norm
            
            improved = False
            for step in steps:
                c_try = c.copy()
                c_try[i] += step * vec
                c_try[j] -= step * vec
                c_try = np.clip(c_try, -0.499999, 0.499999)
                r_try = _compute_radius(c_try)
                if r_try > best_r:
                    best_r = r_try
                    best_c = c_try.copy()
                    c = c_try.copy()
                    improved = True
                    break
            if not improved:
                # Try perpendicular direction
                perp = np.random.normal(size=3)
                perp = perp - np.dot(perp, vec) * vec
                perp_norm = np.linalg.norm(perp)
                if perp_norm > 1e-6:
                    perp /= perp_norm
                    c_try = c.copy()
                    c_try[i] += 0.01 * perp
                    c_try[j] -= 0.01 * perp
                    c_try = np.clip(c_try, -0.499999, 0.499999)
                    r_try = _compute_radius(c_try)
                    if r_try > best_r:
                        best_r = r_try
                        best_c = c_try.copy()
                        c = c_try.copy()
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initial configurations
    seeds = []
    
    seeds.append(_seed_fcc())
    seeds.append(_seed_grid())
    seeds.append(_seed_random())
    seeds.append(_seed_bcc())
    seeds.append(_seed_shell_core())
    seeds.append(_seed_harmonic())
    seeds.append(_seed_random())
    
    best_c = seeds[0].copy()
    best_r = _compute_radius(best_c)
    
    n_seeds = len(seeds)
    # Allocate 65% of time to seed exploration, 35% to final polish
    explore_deadline = start_time + 0.65 * (deadline - start_time)
    time_per_seed = (explore_deadline - start_time) / n_seeds
    
    for i, s in enumerate(seeds):
        if time.time() > explore_deadline:
            break
        
        seed_start = time.time()
        seed_deadline = min(seed_start + time_per_seed, explore_deadline)
        
        # Stage 1: Repulsion annealing
        c_anneal, r_anneal = _repulsion_anneal(s, seed_deadline, t_start=2.0, t_end=0.05, iters_per_temp=40)
        
        # Stage 2: Smooth polish
        c_polish, r_polish = _smooth_polish(c_anneal, seed_deadline, t_start=0.05, t_end=0.005)
        
        # Stage 3: Active-set polish
        c_as, r_as = _active_set_polish(c_polish, seed_deadline)
        
        # Stage 4: Local search
        c_ls, r_ls = _local_search(c_as, seed_deadline, max_iters=150)
        
        if r_ls > best_r:
            best_r = r_ls
            best_c = c_ls.copy()
    
    # Final polish on the global best if time remains
    final_deadline = deadline - 3.0
    if time.time() < final_deadline - 5.0:
        # First: Smooth polish with tighter temperature range
        c_final, r_final = _smooth_polish(best_c, final_deadline, t_start=0.03, t_end=0.0001, iters_per_temp=100)
        if r_final > best_r:
            best_r = r_final
            best_c = c_final.copy()
        
        # Second: Active-set polish
        if time.time() < final_deadline - 2.0:
            c_as2, r_as2 = _active_set_polish(best_c, final_deadline)
            if r_as2 > best_r:
                best_r = r_as2
                best_c = c_as2.copy()
        
        # Third: Local search with more iterations
        if time.time() < final_deadline - 2.0:
            c_ls, r_ls = _local_search(best_c, final_deadline, max_iters=400)
            if r_ls > best_r:
                best_r = r_ls
                best_c = c_ls.copy()
        
        # Fourth: Another round of smooth polish on the refined result
        if time.time() < final_deadline - 2.0 and best_r > 0.13:
            c_final2, r_final2 = _smooth_polish(best_c, final_deadline, t_start=0.01, t_end=0.00005, iters_per_temp=100)
            if r_final2 > best_r:
                best_r = r_final2
                best_c = c_final2.copy()
                
    return best_c
