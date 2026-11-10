import numpy as np
import time
from scipy.spatial.distance import pdist

N = 50

def _compute_score(centers):
    """Compute the maximum radius for given centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    if N < 2:
        min_dist = np.inf
    else:
        dists = pdist(centers, metric='euclidean')
        min_dist = np.min(dists)
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_dist / 2.0, min_wall)

def _optimize_repulsion(centers, time_budget, lr_init=0.005):
    """Gradient-based optimization using smooth repulsive potential with momentum."""
    start_time = time.time()
    pts = centers.copy()
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    
    lr = lr_init
    momentum = 0.9
    velocity = np.zeros_like(pts)
    no_improve_count = 0
    
    for step in range(100000):
        if time.time() - start_time > time_budget:
            break
        
        # Compute pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1e10)
        
        # Smooth repulsive force: minimize sum 1/dist^2
        # gradient of 1/d^2 wrt c_i = -2*(c_i - c_j)/d^4
        # So force (negative gradient) = 2 * sum_j (c_i - c_j) / d^4
        factors = 1.0 / (dist_sq**2 + 1e-15)
        np.fill_diagonal(factors, 0.0)
        
        pair_forces = np.sum(factors[:, :, np.newaxis] * diff, axis=1)
        
        # Wall forces: potential U = sum 1/(0.5 - |x|)^2
        # Force = -dU/dx = -2 * sign(x) / (0.5 - |x|)^3
        wall_dist = 0.5 - np.abs(pts)
        wall_dist_safe = np.maximum(wall_dist, 1e-4)
        signs = np.sign(pts)
        wall_forces = -2.0 * signs / (wall_dist_safe**3)
        
        # Scale wall forces to be comparable
        total_forces = pair_forces + 0.5 * wall_forces
        
        # Normalize to prevent explosion
        norms = np.linalg.norm(total_forces, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = total_forces / norms
        
        # Momentum update
        velocity = momentum * velocity + (1 - momentum) * normalized
        pts = pts + lr * velocity
        
        # Clamp
        pts = np.clip(pts, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Check score periodically
        if step % 30 == 0:
            current_score = _compute_score(pts)
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
                no_improve_count = 0
                # Slightly increase LR on improvement
                lr = min(lr * 1.05, 0.02)
            else:
                no_improve_count += 1
                # Decay learning rate if no improvement
                if no_improve_count > 10:
                    lr *= 0.995
                    no_improve_count = 0
    
    # Final score check
    final_score = _compute_score(pts)
    if final_score > best_score:
        best_score = final_score
        best_pts = pts.copy()
    
    return best_pts, best_score

def _optimize_sa(centers, time_budget, target_r=0.135):
    """Simulated annealing optimization with adaptive target."""
    start_time = time.time()
    current = centers.copy()
    best = current.copy()
    best_score = _compute_score(current)
    
    # Simulated annealing parameters
    T = 0.005
    T_min = 1e-10
    cooling = 0.99995
    step_size = 0.008
    
    rng = np.random.default_rng(123)
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
        
        # Propose a move
        idx = rng.integers(N)
        delta = rng.standard_normal(3) * step_size
        new_centers = current.copy()
        new_centers[idx] += delta
        
        # Keep within bounds with small margin
        new_centers = np.clip(new_centers, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Compute energy difference using only moved point's interactions
        diffs = new_centers[idx] - current
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        dists[idx] = np.inf  # Ignore self
        
        # Overlap penalty for pairs involving idx
        overlaps = np.maximum(0, 2 * target_r - dists)
        pair_energy_new = np.sum(overlaps**2)
        
        # Wall violation for idx
        wall_viol = np.maximum(0, np.abs(new_centers[idx]) - (0.5 - target_r))
        wall_energy_new = np.sum(wall_viol**2)
        
        new_energy = pair_energy_new + 10.0 * wall_energy_new
        
        # Old energy for idx (approximate)
        diffs_old = current[idx] - current
        dists_old = np.sqrt(np.sum(diffs_old**2, axis=1))
        dists_old[idx] = np.inf
        overlaps_old = np.maximum(0, 2 * target_r - dists_old)
        pair_energy_old = np.sum(overlaps_old**2)
        wall_viol_old = np.maximum(0, np.abs(current[idx]) - (0.5 - target_r))
        wall_energy_old = np.sum(wall_viol_old**2)
        old_energy = pair_energy_old + 10.0 * wall_energy_old
        
        delta_energy = new_energy - old_energy
        
        # Accept if better or with probability exp(-deltaE/T)
        if delta_energy < 0 or rng.random() < np.exp(-delta_energy / max(T, 1e-15)):
            current = new_centers
            # Periodically check full score
            if rng.random() < 0.01:
                current_score = _compute_score(current)
                if current_score > best_score:
                    best_score = current_score
                    best = current.copy()
        
        T *= cooling
        # Gradually reduce step size
        step_size *= 0.99999
        
        # Adaptively update target_r to current best + small margin
        if best_score > 0.13:
            target_r = min(0.14, best_score * 1.05)
    
    # Final score check
    final_score = _compute_score(current)
    if final_score > best_score:
        best_score = final_score
        best = current.copy()
    
    return best, best_score

def _bisection_feasibility(centers, r_low, r_high, time_budget):
    """Bisection search on radius r with projected gradient feasibility solver."""
    start_time = time.time()
    best_r = r_low
    best_centers = centers.copy()
    
    while r_high - r_low > 1e-7 and time.time() - start_time < time_budget:
        r_mid = (r_low + r_high) / 2.0
        
        # Try to find a feasible configuration for radius r_mid
        feasible, feasible_centers = _check_feasibility(centers, r_mid, time_budget - (time.time() - start_time))
        
        if feasible:
            best_r = r_mid
            best_centers = feasible_centers
            r_low = r_mid
        else:
            r_high = r_mid
    
    return best_r, best_centers

def _check_feasibility(centers, r, time_budget):
    """Check if a feasible configuration exists for radius r using projected gradient descent."""
    start_time = time.time()
    pts = centers.copy()
    
    # Project to feasible set: all pairs must be at least 2r apart, all within [-(0.5-r), 0.5-r]
    # Use iterative projection (similar to Dykstra's algorithm or simple alternating projections)
    
    for iteration in range(5000):
        if time.time() - start_time > time_budget:
            break
        
        # Check if already feasible
        if N >= 2:
            dists = pdist(pts, metric='euclidean')
            if np.min(dists) >= 2 * r - 1e-9:
                wall_ok = np.all(np.abs(pts) <= 0.5 - r + 1e-9)
                if wall_ok:
                    return True, pts
        else:
            wall_ok = np.all(np.abs(pts) <= 0.5 - r + 1e-9)
            if wall_ok:
                return True, pts
        
        # Project pairwise: for each pair too close, push apart
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dist_sq)
        np.fill_diagonal(dists, np.inf)
        
        # For pairs with dist < 2r, push them apart
        too_close = dists < 2 * r
        if np.any(too_close):
            # Compute correction for each point
            corrections = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    if too_close[i, j]:
                        d = dists[i, j]
                        if d < 1e-10:
                            # Random separation
                            rand_dir = np.random.randn(3)
                            rand_dir /= np.linalg.norm(rand_dir)
                            correction = rand_dir * (2 * r - d) / 2.0
                        else:
                            direction = diff[i, j] / d
                            correction = direction * (2 * r - d) / 2.0
                        corrections[i] += correction
                        corrections[j] -= correction
            pts += corrections
        
        # Project to walls
        bound = 0.5 - r
        pts = np.clip(pts, -bound, bound)
    
    # Final check
    if N >= 2:
        dists = pdist(pts, metric='euclidean')
        if np.min(dists) >= 2 * r - 1e-9:
            wall_ok = np.all(np.abs(pts) <= 0.5 - r + 1e-9)
            if wall_ok:
                return True, pts
    else:
        wall_ok = np.all(np.abs(pts) <= 0.5 - r + 1e-9)
        if wall_ok:
            return True, pts
    
    return False, pts

def _hilbert_curve_3d(n_points, seed=42):
    """Generate 3D points using a Hilbert-like space-filling curve mapping."""
    rng = np.random.default_rng(seed)
    
    points = np.zeros((n_points, 3))
    
    for i in range(n_points):
        bits = i
        x = 0
        y = 0
        z = 0
        for bit in range(12):
            b = (bits >> bit) & 1
            if bit % 3 == 0:
                x |= (b << (bit // 3))
            elif bit % 3 == 1:
                y |= (b << (bit // 3))
            else:
                z |= (b << (bit // 3))
        points[i, 0] = x / 4096.0
        points[i, 1] = y / 4096.0
        points[i, 2] = z / 4096.0
    
    perm = rng.permutation(n_points)
    points = points[perm]
    
    margin = 0.02
    points = points * (1.0 - 2 * margin) + (-0.5 + margin)
    
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20 second safety margin before 150s limit
    
    best_radius = 0.0
    best_centers = None
    
    # Initialization strategies
    starts = []
    
    # Start 1: Random
    rng = np.random.default_rng(42)
    starts.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # Start 2: Grid-based
    ticks = (np.arange(4) + 0.5) / 4 - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    indices = rng.choice(len(grid), size=N, replace=False)
    starts.append(grid[indices])
    
    # Start 3: Another random
    starts.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
    
    # Start 4: Hilbert curve
    starts.append(_hilbert_curve_3d(N, seed=123))
    
    # Start 5: BCC-like lattice
    # 3x3x3 = 27 points, add face centers and body center
    bcc_pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                bcc_pts.append([(i - 1) * 0.3, (j - 1) * 0.3, (k - 1) * 0.3])
    # Add some more points
    for i in range(3):
        for j in range(3):
            bcc_pts.append([(i - 1) * 0.3, (j - 1) * 0.3, 0.15])
    bcc_pts = np.array(bcc_pts[:N])
    starts.append(bcc_pts)
    
    # Run optimization on each start
    for i, init in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Phase 1: Gradient-based repulsion
        centers, radius = _optimize_repulsion(init, time_budget=20, lr_init=0.005)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers
        
        # Phase 2: Simulated annealing polish
        if time.time() < deadline - 10:
            centers, radius = _optimize_sa(centers, time_budget=15, target_r=max(0.13, best_radius))
            
            if radius > best_radius:
                best_radius = radius
                best_centers = centers
    
    # If we have time, do bisection feasibility search on the best result
    if time.time() < deadline - 5 and best_centers is not None:
        r_low = best_radius
        r_high = best_radius * 1.05 + 0.005
        r, centers = _bisection_feasibility(best_centers, r_low, r_high, time_budget=15)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers
    
    # Final polish with gradient descent
    if time.time() < deadline - 2 and best_centers is not None:
        centers, radius = _optimize_repulsion(best_centers, time_budget=5, lr_init=0.001)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers
    
    if best_centers is None:
        # Fallback
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
