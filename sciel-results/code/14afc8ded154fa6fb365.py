"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

def _score(centers):
    """Compute the packing radius for a set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers > 0.5) or np.any(centers < -0.5):
        return 0.0
    
    # Compute pairwise distances using KDTree for efficiency
    try:
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair_dist = np.min(dists[:, 1])
    except Exception:
        min_pair_dist = 0.0
    
    # Distance to walls
    coords = np.abs(centers)
    dist_to_wall = 0.5 - np.max(coords, axis=1)
    min_wall_dist = np.min(dist_to_wall)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    
    if r <= 0:
        return 0.0
    
    return r

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin
    
    def compute_radius(c):
        """Fast radius computation using KDTree."""
        if not np.all(np.isfinite(c)):
            return 0.0
        if np.any(c > 0.5) or np.any(c < -0.5):
            return 0.0
        wall_dists = 0.5 - np.abs(c)
        r_wall = np.min(wall_dists)
        try:
            tree = cKDTree(c)
            dists, _ = tree.query(c, k=2)
            r_pair = np.min(dists[:, 1]) / 2.0
        except Exception:
            r_pair = 0.0
        return min(r_wall, r_pair)

    def force_directed_refine(centers, time_limit, step_scale=1.0):
        """Force-directed local search to improve packing."""
        centers = centers.copy()
        r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = r
        
        base_step = 0.05 * step_scale
        decay_rate = 0.999
        stagnation_count = 0
        it = 0
        
        while time.time() < time_limit:
            it += 1
            
            wall_dists = 0.5 - np.abs(centers)
            r_wall = np.min(wall_dists)
            
            try:
                tree = cKDTree(centers)
                dists, _ = tree.query(centers, k=2)
                min_pair_dist = np.min(dists[:, 1])
                r_pair = min_pair_dist / 2.0
            except Exception:
                break
                
            current_r = min(r_wall, r_pair)
            
            if current_r <= best_r * 1.000001:
                stagnation_count += 1
                if stagnation_count > 100:
                    break
            else:
                stagnation_count = 0
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            # Compute forces
            force = np.zeros_like(centers)
            target_r = current_r + 0.002
            
            # Wall forces
            for k in range(3):
                d_wall = 0.5 - np.abs(centers[:, k])
                mask = d_wall < target_r
                if np.any(mask):
                    viol = target_r - d_wall[mask]
                    dirs = -np.sign(centers[mask, k])
                    dirs[dirs == 0] = 1.0
                    force[mask, k] += viol * dirs
            
            # Pair forces
            target_dist = 2 * target_r
            try:
                pairs = tree.query_pairs(target_dist)
            except Exception:
                pairs = []
            
            for i, j in pairs:
                diff = centers[i] - centers[j]
                d = np.linalg.norm(diff)
                if d > 1e-10 and d < target_dist:
                    direction = diff / d
                    viol = target_dist - d
                    force[i] += viol * direction
                    force[j] -= viol * direction
            
            # Apply forces with adaptive step
            norm_force = np.linalg.norm(force)
            if norm_force > 1e-10:
                step = base_step * decay_rate ** it
                centers += step * force / norm_force
                centers = np.clip(centers, -0.499999, 0.499999)
        
        return best_centers, best_r

    def penalty_refine(centers, r_target, time_limit, max_iter=300):
        """Refine centers to make radius r_target feasible using penalty method."""
        centers = centers.copy()
        x0 = centers.ravel()
        
        def objective_and_grad(x):
            if time.time() > time_limit:
                return 1e10, np.zeros_like(x)
            
            c = x.reshape(N, 3)
            
            # Wall violations
            wall_dists = 0.5 - np.abs(c)
            wall_viol = np.maximum(0, r_target - wall_dists)
            
            # Pairwise violations
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, np.inf)
            dists = np.sqrt(dists_sq)
            pair_viol = np.maximum(0, 2 * r_target - dists)
            
            iu = np.triu_indices(N, k=1)
            pv = pair_viol[iu]
            
            obj = np.sum(wall_viol**2) + np.sum(pv**2)
            
            # Gradient
            grad = np.zeros_like(c)
            
            # Wall gradient
            for k in range(3):
                mask = wall_viol[:, k] > 0
                if np.any(mask):
                    grad[mask, k] += 2 * wall_viol[mask, k] * np.sign(c[mask, k])
            
            # Pair gradient
            for idx in range(len(iu[0])):
                i, j = iu[0][idx], iu[1][idx]
                if pv[idx] > 0:
                    d = dists[i, j]
                    if d > 1e-12:
                        direction = (c[i] - c[j]) / d
                        grad[i] += -2 * pv[idx] * direction
                        grad[j] += 2 * pv[idx] * direction
            
            return obj, grad.ravel()
        
        try:
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.499999, 0.499999)] * (N * 3),
                options={'maxiter': max_iter, 'maxfun': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            new_centers = result.x.reshape(N, 3)
            new_centers = np.clip(new_centers, -0.499999, 0.499999)
            new_r = compute_radius(new_centers)
            if new_r >= r_target * 0.9999:
                return new_centers, new_r, True
        except Exception:
            pass
        
        return centers, compute_radius(centers), False

    def binary_search_radius(centers, r_low, r_high, time_limit, max_iters=20):
        """Binary search for the maximum feasible radius."""
        best_centers = centers.copy()
        best_r = compute_radius(centers)
        
        for _ in range(max_iters):
            if time.time() > time_limit - 5:
                break
            
            r_mid = (r_low + r_high) / 2.0
            if r_mid <= best_r:
                r_low = r_mid
                continue
            
            it_limit = time.time() + min(5.0, (time_limit - time.time()) / 3.0)
            new_centers, new_r, feasible = penalty_refine(centers, r_mid, it_limit)
            
            if feasible and new_r >= r_mid * 0.9999:
                best_r = new_r
                best_centers = new_centers
                r_low = r_mid
            else:
                r_high = r_mid
        
        return best_centers, best_r

    # Start with a reasonable initial configuration
    # Use a body-centered cubic (BCC) lattice or similar dense arrangement
    # For 100 spheres, let's start with a perturbed grid
    
    # Initial: 5x5x4 grid with some randomization
    m = 5
    n = 4
    # Create grid points
    xs = np.linspace(-0.35, 0.35, m)
    ys = np.linspace(-0.35, 0.35, m)
    zs = np.linspace(-0.35, 0.35, n)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    centers = grid[:N].copy()
    
    # Add small random perturbation
    rng = np.random.RandomState(42)
    centers += rng.uniform(-0.02, 0.02, size=centers.shape)
    
    # Clip to valid range
    centers = np.clip(centers, -0.45, 0.45)
    
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Phase 1: Force-directed refinement
    if time.time() < deadline - 30:
        fd_limit = min(deadline - 30, time.time() + 20)
        best_centers, best_score = force_directed_refine(centers, fd_limit, step_scale=1.0)
    
    # Phase 2: Binary search with penalty refinement
    if time.time() < deadline - 10:
        r_low = best_score
        r_high = 0.12  # Upper bound estimate
        bs_limit = deadline - 5
        best_centers, best_score = binary_search_radius(best_centers, r_low, r_high, bs_limit)
    
    # Phase 3: Final force-directed polish
    if time.time() < deadline - 2:
        fd_limit = deadline - 1
        best_centers, best_score = force_directed_refine(best_centers, fd_limit, step_scale=0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
