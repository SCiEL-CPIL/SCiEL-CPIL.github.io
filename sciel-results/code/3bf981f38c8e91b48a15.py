"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
N = 21
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

DEADLINE = 145.0  # seconds wall-clock limit (5s safety margin from 150s)


def _compute_radii_fast(centers, W, H):
    """Vectorized computation of radii using pairwise distances."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_nn = np.min(dists, axis=1) / 2.0
    cx, cy = centers[:, 0], centers[:, 1]
    boundary_dist = np.minimum.reduce([cx, W - cx, cy, H - cy])
    radii = np.maximum(np.minimum(min_nn, boundary_dist), 1e-12)
    return radii


def _compute_radii_voronoi(centers, W, H):
    """Compute inradii using Voronoi tessellation for more accurate radii."""
    n = len(centers)
    # Compute pairwise-based radii first
    radii_pair = _compute_radii_fast(centers, W, H)
    
    # Try Voronoi for potentially better estimates
    try:
        # Add boundary points to make Voronoi bounded
        margin = 0.1 * max(W, H)
        boundary_pts = np.array([
            [0, 0], [W, 0], [W, H], [0, H],
            [W/2, -margin], [W/2, H + margin],
            [-margin, H/2], [W + margin, H/2],
            [W*0.25, -margin], [W*0.75, -margin],
            [W*0.25, H + margin], [W*0.75, H + margin],
            [-margin, H*0.25], [-margin, H*0.75],
            [W + margin, H*0.25], [W + margin, H*0.75],
        ])
        all_pts = np.vstack([centers, boundary_pts])
        
        vor = Voronoi(all_pts)
        
        for i in range(n):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if len(region) == 0:
                continue
            
            # Get the vertices of this Voronoi cell
            vertices = vor.vertices[region]
            if len(vertices) < 3:
                continue
            
            # The inradius is the minimum distance from center to any edge of the cell
            # For each pair of consecutive vertices, compute distance from center to line
            min_dist = radii_pair[i]  # fallback
            for j in range(len(vertices)):
                v1 = vertices[j]
                v2 = vertices[(j + 1) % len(vertices)]
                # Distance from point to line segment
                d = _point_to_segment_dist(centers[i], v1, v2)
                min_dist = min(min_dist, d)
            
            radii_pair[i] = min(radii_pair[i], min_dist)
    except Exception:
        pass
    
    return np.maximum(radii_pair, 1e-12)


def _point_to_segment_dist(p, a, b):
    """Distance from point p to line segment ab."""
    ab = b - a
    ap = p - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-15:
        return np.linalg.norm(ap)
    t = np.clip(np.dot(ap, ab) / ab_len_sq, 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def _sum_radii(centers, W, H):
    """Objective: sum of radii."""
    return np.sum(_compute_radii_fast(centers, W, H))


def _objective_soft(x, W, H, beta):
    """Differentiable soft objective for L-BFGS-B - vectorized."""
    n = len(x) // 2
    c = x.reshape(n, 2)
    cx, cy = c[:, 0], c[:, 1]
    
    # Distance to boundaries (n,)
    d_bound = np.minimum.reduce([cx, W - cx, cy, H - cy])
    
    # Pairwise distances (n, n)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    half_dists = dists / 2.0  # (n, n)
    
    # For each circle i, compute soft-min of [d_bound[i], half_dists[i,:]]
    # Stack: (n, n+1)
    all_vals = np.column_stack([d_bound, half_dists])  # (n, n+1)
    
    # Soft-min along axis=1
    min_vals = np.min(all_vals, axis=1)  # (n,)
    exp_terms = np.exp(-beta * (all_vals - min_vals[:, np.newaxis]))  # (n, n+1)
    sum_exp = np.sum(exp_terms, axis=1)  # (n,)
    softmins = min_vals - np.log(sum_exp) / beta  # (n,)
    
    return -np.sum(softmins)


def _local_refine(centers, W, H, time_limit=10.0, max_iters=500):
    """Local refinement using L-BFGS-B with soft-min objective, multiple beta levels."""
    n = len(centers)
    if n < 2:
        return centers, _sum_radii(centers, W, H)
    
    best_x = centers.flatten().copy()
    best_val = _sum_radii(centers, W, H)
    
    start_time = time.time()
    betas = [20.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    
    bounds = []
    for i in range(n):
        bounds.append((0, W))
        bounds.append((0, H))
    
    for beta in betas:
        if time.time() - start_time > time_limit - 1:
            break
        
        try:
            res = minimize(
                _objective_soft,
                best_x,
                args=(W, H, beta),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iters, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res.fun < -best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # Final exact evaluation
    final_centers = best_x.reshape(-1, 2)
    final_val = _sum_radii(final_centers, W, H)
    
    return final_centers, final_val


def _initial_grid(N, W, H):
    """Create an initial grid placement."""
    side = 5
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) >= N:
                break
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            centers.append([cx, cy])
    return np.array(centers[:N])


def _initial_grid_7x3(N, W, H):
    """7x3 grid layout."""
    centers = []
    for i in range(3):
        for j in range(7):
            if len(centers) >= N:
                break
            cx = (j + 0.5) * W / 7
            cy = (i + 0.5) * H / 3
            centers.append([cx, cy])
    return np.array(centers[:N])


def _initial_hexagonal(N, W, H):
    """Hexagonal packing initialization."""
    centers = []
    for n_rows in range(3, 8):
        n_cols = int(np.ceil(N / n_rows))
        for i in range(n_rows):
            for j in range(n_cols):
                if len(centers) >= N:
                    break
                x_spacing = W / (n_cols + 0.5)
                y_spacing = H / (n_rows + 0.5)
                cx = (j + 0.25 + (0.25 if i % 2 == 1 else 0)) * x_spacing
                cy = (i + 0.25) * y_spacing
                cx = min(max(cx, 0.02 * W), 0.98 * W)
                cy = min(max(cy, 0.02 * H), 0.98 * H)
                centers.append([cx, cy])
        if len(centers) >= N:
            break
    return np.array(centers[:N])


def _initial_random(N, W, H, seed=None):
    """Create a random initial placement."""
    rng = np.random.default_rng(seed)
    margin = 0.03
    centers = rng.uniform(margin, 1 - margin, size=(N, 2))
    centers[:, 0] *= W
    centers[:, 1] *= H
    return centers


def _sa_optimize(centers, W, H, time_limit=15.0, seed=42):
    """Simulated annealing for circle packing."""
    n = len(centers)
    rng = np.random.default_rng(seed)
    
    best_centers = centers.copy()
    best_val = _sum_radii(best_centers, W, H)
    current_centers = centers.copy()
    current_val = best_val
    
    start_time = time.time()
    T = 0.02  # initial temperature
    T_min = 1e-6
    cooling = 0.995
    
    while time.time() - start_time < time_limit:
        # Generate perturbation
        step_size = T * 2.0
        delta = rng.normal(0, step_size, size=(n, 2))
        new_centers = current_centers + delta
        
        # Clamp to bounds
        new_centers[:, 0] = np.clip(new_centers[:, 0], 0, W)
        new_centers[:, 1] = np.clip(new_centers[:, 1], 0, H)
        
        new_val = _sum_radii(new_centers, W, H)
        delta_val = new_val - current_val
        
        # Accept if better or with probability exp(delta/T)
        if delta_val > 0 or rng.random() < np.exp(delta_val / max(T, T_min)):
            current_centers = new_centers
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best_centers = current_centers.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_centers, best_val


def _physics_relax(centers, W, H, time_limit=10.0, seed=42):
    """
    Physics-based relaxation using radii-aware repulsion forces.
    Each circle is pushed away from neighbors proportional to overlap,
    and pulled toward the center of its available space.
    """
    n = len(centers)
    rng = np.random.default_rng(seed)
    
    pos = centers.copy()
    vel = np.zeros_like(pos)
    
    # Compute initial radii for reference
    best_val = _sum_radii(pos, W, H)
    best_pos = pos.copy()
    
    start_time = time.time()
    
    # Parameters
    dt = 0.003  # time step
    damping = 0.92  # velocity damping factor
    max_force = 8.0
    
    # Adaptive parameters
    no_improve_steps = 0
    patience = 150
    eval_count = 0
    
    while time.time() - start_time < time_limit:
        # Compute current radii
        radii = _compute_radii_fast(pos, W, H)
        
        # Compute forces based on circle overlap
        forces = np.zeros_like(pos)
        
        # Pairwise: if circles overlap, push apart
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap: r_i + r_j - d_ij
        overlap = radii[:, np.newaxis] + radii[np.newaxis, :] - dists
        overlap = np.maximum(overlap, 0)
        np.fill_diagonal(overlap, 0)
        
        # Force direction: normalized diff, magnitude proportional to overlap
        with np.errstate(divide='ignore', invalid='ignore'):
            dir_vec = diff / (dists[:, :, np.newaxis] + 1e-10)
            # Stronger force for larger overlaps
            force_mag = overlap * 2.0  # scaling factor
            forces += np.sum(force_mag[:, :, np.newaxis] * dir_vec, axis=1)
        
        # Boundary forces: push circles that are too close to walls inward
        # The "effective" boundary distance for circle i is pos[i] - r[i]
        for axis in range(2):
            dim = W if axis == 0 else H
            # Effective distance to left/bottom wall
            eff_left = pos[:, axis] - radii
            too_close_left = eff_left < 0.01
            if np.any(too_close_left):
                forces[too_close_left, axis] += 3.0 * (0.01 - eff_left[too_close_left])
            # Effective distance to right/top wall
            eff_right = (dim - pos[:, axis]) - radii
            too_close_right = eff_right < 0.01
            if np.any(too_close_right):
                forces[too_close_right, axis] -= 3.0 * (0.01 - eff_right[too_close_right])
        
        # Gentle attraction toward center to prevent drift
        center = np.array([W/2, H/2])
        forces += 0.05 * (center - pos)
        
        # Clamp forces
        forces = np.clip(forces, -max_force, max_force)
        
        # Velocity-Verlet integration
        vel = vel * damping + forces * dt
        pos = pos + vel * dt
        
        # Clamp positions to bounds (with small margin)
        margin = 1e-4
        pos[:, 0] = np.clip(pos[:, 0], margin, W - margin)
        pos[:, 1] = np.clip(pos[:, 1], margin, H - margin)
        
        # Evaluate (throttle evaluations)
        eval_count += 1
        if eval_count % 5 == 0 or time.time() - start_time > time_limit - 0.5:
            current_val = _sum_radii(pos, W, H)
            
            if current_val > best_val:
                best_val = current_val
                best_pos = pos.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        
        # Adaptive: if stuck, add small random perturbation
        if no_improve_steps > patience:
            vel += rng.normal(0, 0.02, size=pos.shape)
            damping = min(damping * 0.97, 0.80)
            dt *= 0.93
            no_improve_steps = 0
            patience = max(int(patience * 0.85), 40)
        
        # Cool down over time
        elapsed_frac = (time.time() - start_time) / time_limit
        if elapsed_frac > 0.6:
            dt *= 0.998
            damping = min(damping * 0.999, 0.88)
    
    return best_pos, best_val


def _optimize_for_aspect(W, H, time_limit=20.0, seed=42):
    """Optimize for a given aspect ratio (W x H)."""
    start_time = time.time()
    best_centers = None
    best_val = -np.inf
    
    def try_optimize(init_centers):
        nonlocal best_centers, best_val
        if time.time() - start_time > time_limit - 1:
            return
        
        val_init = _sum_radii(init_centers, W, H)
        if val_init > best_val:
            best_val = val_init
            best_centers = init_centers.copy()
        
        remaining = time_limit - (time.time() - start_time) - 0.5
        if remaining < 2:
            return
        
        # Try SA first (explore), then L-BFGS-B (exploit)
        sa_time = min(remaining * 0.35, 7.0)
        sa_centers, sa_val = _sa_optimize(init_centers, W, H, time_limit=sa_time, seed=seed)
        
        if sa_val > best_val:
            best_val = sa_val
            best_centers = sa_centers.copy()
        
        remaining = time_limit - (time.time() - start_time) - 0.5
        if remaining < 2:
            return
        
        # L-BFGS-B refinement from SA result
        lbfgs_time = min(remaining * 0.7, 10.0)
        refined, val_ref = _local_refine(sa_centers, W, H, 
                                         time_limit=lbfgs_time,
                                         max_iters=800)
        if val_ref > best_val:
            best_val = val_ref
            best_centers = refined.copy()
    
    # Try multiple initial configurations
    # 1. 5x5 grid
    try_optimize(_initial_grid(N, W, H))
    
    # 2. 7x3 grid
    try_optimize(_initial_grid_7x3(N, W, H))
    
    # 3. Hexagonal packing
    try_optimize(_initial_hexagonal(N, W, H))
    
    # 4. Random starts
    n_random = 3
    for i in range(n_random):
        if time.time() - start_time > time_limit - 5:
            break
        centers_rand = _initial_random(N, W, H, seed=seed + i * 1000)
        try_optimize(centers_rand)
    
    # 5. Physics-based relaxation from best current result
    if best_centers is not None and time.time() - start_time < time_limit - 3:
        phys_time = min(time_limit - (time.time() - start_time) - 1.0, 8.0)
        if phys_time > 2:
            try:
                phys_centers, phys_val = _physics_relax(best_centers, W, H, 
                                                        time_limit=phys_time, 
                                                        seed=seed + 999)
                if phys_val > best_val:
                    best_val = phys_val
                    best_centers = phys_centers.copy()
            except Exception:
                pass
    
    # 6. Final L-BFGS-B polish from physics result
    if best_centers is not None and time.time() - start_time < time_limit - 1:
        remaining = time_limit - (time.time() - start_time) - 0.5
        if remaining > 2:
            try:
                refined, val_ref = _local_refine(best_centers, W, H,
                                                 time_limit=min(remaining, 6.0),
                                                 max_iters=500)
                if val_ref > best_val:
                    best_val = val_ref
                    best_centers = refined.copy()
            except Exception:
                pass
    
    return best_centers, best_val


def construct():
    """Main entry point: pack 21 circles in a perimeter-4 rectangle, maximize sum of radii."""
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_centers = None
    best_val = 0.0
    best_W = None
    best_H = None
    
    # Perimeter constraint: 2*(W+H) = 4 => W+H = 2
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.9, 1.1, 0.85, 1.18, 0.8, 1.25, 0.95, 1.05, 0.75, 1.33]
    
    # Allocate time: more time for square, less for others
    weights = [0.25, 0.12, 0.12, 0.10, 0.10, 0.09, 0.09, 0.07, 0.07, 0.08, 0.08]
    
    for idx, ar in enumerate(aspect_ratios):
        if time.time() > deadline - 5.0:
            break
        
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 / (1.0 + ar)
        
        remaining = deadline - time.time() - 3.0
        if remaining < 4:
            break
        
        time_alloc = min(remaining * weights[idx] * 2.5, 25.0)
        time_alloc = max(time_alloc, 3.0)
        
        centers, val = _optimize_for_aspect(W, H, time_limit=time_alloc, seed=42 + idx * 7)
        
        if centers is not None and val > best_val:
            best_val = val
            best_centers = centers.copy()
            best_W = W
            best_H = H
    
    if best_centers is None:
        W, H = 1.0, 1.0
        best_centers = _initial_grid(N, W, H)
        best_W, best_H = W, H
        best_val = _sum_radii(best_centers, best_W, best_H)
    
    # Final polish with high precision
    remaining = deadline - time.time()
    if remaining > 3:
        refined, val_ref = _local_refine(best_centers, best_W, best_H,
                                         time_limit=min(remaining - 1, 20),
                                         max_iters=1500)
        if val_ref > best_val:
            best_val = val_ref
            best_centers = refined
    
    # Physics-based relaxation as final exploration
    remaining = deadline - time.time()
    if remaining > 4:
        phys_time = min(remaining - 2, 8.0)
        if phys_time > 2:
            try:
                phys_centers, phys_val = _physics_relax(best_centers, best_W, best_H,
                                                        time_limit=phys_time,
                                                        seed=12345)
                if phys_val > best_val:
                    best_val = phys_val
                    best_centers = phys_centers.copy()
            except Exception:
                pass
    
    # One more polish pass after physics
    remaining = deadline - time.time()
    if remaining > 2:
        refined, val_ref = _local_refine(best_centers, best_W, best_H,
                                         time_limit=min(remaining - 0.5, 10),
                                         max_iters=800)
        if val_ref > best_val:
            best_val = val_ref
            best_centers = refined
    
    # Final micro-polish
    remaining = deadline - time.time()
    if remaining > 1.5:
        refined, val_ref = _local_refine(best_centers, best_W, best_H,
                                         time_limit=min(remaining - 0.5, 5),
                                         max_iters=400)
        if val_ref > best_val:
            best_val = val_ref
            best_centers = refined
    
    # Use Voronoi-based radii for final evaluation (may give tighter fit)
    radii_voro = _compute_radii_voronoi(best_centers, best_W, best_H)
    radii_pair = _compute_radii_fast(best_centers, best_W, best_H)
    # Take the minimum to ensure validity
    radii = np.minimum(radii_voro, radii_pair)
    radii = radii * (1 - 1e-7)
    
    result = np.column_stack([best_centers[:, 0], best_centers[:, 1], radii])
    
    return result


# Quick self-test
if __name__ == "__main__":
    result = construct()
    print(f"Number of circles: {len(result)}")
    print(f"Sum of radii: {np.sum(result[:, 2]):.6f}")
    print(f"First few circles: {result[:3]}")
