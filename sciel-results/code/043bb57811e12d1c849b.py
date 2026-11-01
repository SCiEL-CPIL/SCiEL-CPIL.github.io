"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100

# EVOLVE-BLOCK-START

def _compute_radius(centres):
    """Compute the maximum valid radius for given centres in [-0.5, 0.5]^3."""
    if not np.all(np.isfinite(centres)):
        return 0.0
    if centres.shape != (N, 3):
        return 0.0
    wall_dists = np.min(0.5 - np.abs(centres), axis=1)
    min_wall = np.min(wall_dists)
    if min_wall < 0:
        return 0.0
    tree = cKDTree(centres)
    pairs = tree.query_pairs(1.0, output_type='ndarray')
    if len(pairs) == 0:
        return min_wall
    diffs = centres[pairs[:, 0]] - centres[pairs[:, 1]]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    min_pair = np.min(dists)
    return min(min_pair / 2.0, min_wall)


def _soft_min_objective_and_grad(x, beta=50.0):
    """
    Differentiable soft-min objective: minimize -soft_min(min_pair_dist/2, min_wall_dist).
    Uses log-sum-exp trick for smooth minimum approximation.
    Returns (objective, gradient) where objective = -soft_min_value.
    """
    c = x.reshape(N, 3)
    
    # Wall distances: (N, 3) values, take min per point
    wall_dists = 0.5 - np.abs(c)  # (N, 3)
    
    # Pair distances
    tree = cKDTree(c)
    pairs = tree.query_pairs(1.0, output_type='ndarray')
    
    if len(pairs) == 0:
        min_val = np.min(wall_dists)
        return -min_val, np.zeros_like(x)
    
    diffs = c[pairs[:, 0]] - c[pairs[:, 1]]
    pair_dists = np.sqrt(np.sum(diffs**2, axis=1))
    
    # Collect all constraint values: wall_dists (flattened) and pair_dists/2
    # We want soft_min of all these
    all_vals = np.concatenate([wall_dists.ravel(), pair_dists / 2.0])
    
    # Soft min via log-sum-exp: soft_min(x) = -log(sum(exp(-beta * x))) / beta
    # For numerical stability, shift by min
    m = np.min(all_vals)
    exp_vals = np.exp(-beta * (all_vals - m))
    s = np.sum(exp_vals)
    soft_min_val = m - np.log(s) / beta
    
    # Gradient of -soft_min = d/dx [log(sum(exp(-beta*x_i)))/beta]
    # = (1/beta) * sum_j [-beta * exp(-beta*x_j) / sum_k exp(-beta*x_k)] * dx_j/dx
    # = -sum_j [exp(-beta*x_j)/s] * dx_j/dx
    weights = exp_vals / s  # (num_constraints,)
    
    grad = np.zeros_like(c)
    
    # Wall gradient contribution
    wall_weights = weights[:N * 3].reshape(N, 3)
    for axis in range(3):
        # d/dx of wall_dist_i_axis = -sign(x_i_axis)
        grad[:, axis] += np.where(c[:, axis] >= 0, 1.0, -1.0) * wall_weights[:, axis]
    
    # Pair gradient contribution
    if len(pairs) > 0:
        pair_weights = weights[N * 3:]
        for idx in range(len(pairs)):
            i, j = pairs[idx]
            d = pair_dists[idx]
            if d < 1e-12:
                continue
            w = pair_weights[idx] * 0.5  # factor of 1/2 from pair_dists/2
            # d/dx_i of (pair_dists/2) = (c_i - c_j) / (2 * d)
            diff = diffs[idx] / (2.0 * d)
            grad[i] += w * diff
            grad[j] -= w * diff
    
    return -soft_min_val, grad.ravel()


def _optimize_soft_min(centres, beta_schedule=None, deadline=None, max_iters=300):
    """
    Optimize using soft-min objective with adaptive beta (annealing).
    Start with low beta (smooth), increase to high beta (sharp).
    """
    if beta_schedule is None:
        beta_schedule = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
    
    c = centres.copy().astype(np.float64)
    bounds = [(-0.4999, 0.4999)] * (N * 3)
    
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    for beta in beta_schedule:
        if deadline is not None and time.time() > deadline:
            break
        
        def obj_grad(x):
            return _soft_min_objective_and_grad(x, beta)
        
        try:
            res = minimize(
                obj_grad,
                c.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            
            new_c = np.clip(res.x.reshape(N, 3), -0.4999, 0.4999)
            new_r = _compute_radius(new_c)
            
            if new_r > best_r:
                best_r = new_r
                best_c = new_c.copy()
            
            c = new_c
        except Exception:
            pass
    
    return best_c, best_r


def _force_directed_refinement(centres, max_iters=200, dt_init=0.008, deadline=None):
    """Vectorized force-directed repulsion with wall constraints."""
    centres = centres.copy().astype(np.float64)
    best_centres = centres.copy()
    best_r = _compute_radius(best_centres)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        tree = cKDTree(centres)
        pairs = tree.query_pairs(0.5, output_type='ndarray')
        
        forces = np.zeros_like(centres)
        
        if len(pairs) > 0:
            diffs = centres[pairs[:, 0]] - centres[pairs[:, 1]]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            
            for idx in range(len(pairs)):
                i, j = pairs[idx]
                d = dists[idx]
                if d < 1e-12:
                    continue
                # Strong repulsion when close
                f = 1.0 / (d**3 + 1e-8)
                forces[i] += f * diffs[idx] / d
                forces[j] -= f * diffs[idx] / d
        
        # Wall forces
        for axis in range(3):
            over = centres[:, axis] - 0.5
            mask_pos = over > -0.05
            if np.any(mask_pos):
                forces[mask_pos, axis] -= 50.0 * np.clip(over[mask_pos], 0, None) + 5.0 * np.clip(0.05 + over[mask_pos], 0, None)
            under = centres[:, axis] + 0.5
            mask_neg = under < 0.05
            if np.any(mask_neg):
                forces[mask_neg, axis] += 50.0 * np.clip(-under[mask_neg], 0, None) + 5.0 * np.clip(0.05 - under[mask_neg], 0, None)
        
        dt = dt_init * (0.995 ** it)
        centres += dt * forces
        centres = np.clip(centres, -0.4999, 0.4999)
        
        r = _compute_radius(centres)
        if r > best_r:
            best_r = r
            best_centres = centres.copy()
    
    return best_centres, best_r


def _lattice_swap_local_search(centres, max_swaps=500, deadline=None):
    """
    Local search: try swapping positions of pairs of spheres to improve the minimum radius.
    Uses nearest-neighbor graph to identify candidate swaps.
    """
    c = centres.copy().astype(np.float64)
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    for swap_iter in range(max_swaps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find the pair that achieves the minimum distance
        tree = cKDTree(c)
        pairs = tree.query_pairs(0.5, output_type='ndarray')
        
        if len(pairs) == 0:
            break
        
        diffs = c[pairs[:, 0]] - c[pairs[:, 1]]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        min_idx = np.argmin(dists)
        i, j = pairs[min_idx]
        
        # Try swapping i with a random other sphere
        candidates = np.random.choice(N, size=min(20, N), replace=False)
        candidates = candidates[candidates != i]
        
        improved = False
        for k in candidates:
            if k == i or k == j:
                continue
            
            # Try swap i <-> k
            c_new = c.copy()
            c_new[i], c_new[k] = c[k].copy(), c[i].copy()
            
            r_new = _compute_radius(c_new)
            if r_new > best_r + 1e-12:
                c = c_new
                best_c = c_new.copy()
                best_r = r_new
                improved = True
                break
        
        if not improved:
            # Try moving i slightly toward j's "opposite" direction
            d = diffs[min_idx]
            norm = np.sqrt(np.sum(d**2))
            if norm > 1e-12:
                direction = d / norm
                for delta in [0.01, 0.02, 0.03]:
                    c_new = c.copy()
                    c_new[i] = c[i] + direction * delta
                    c_new[i] = np.clip(c_new[i], -0.4999, 0.4999)
                    
                    r_new = _compute_radius(c_new)
                    if r_new > best_r + 1e-12:
                        c = c_new
                        best_c = c_new.copy()
                        best_r = r_new
                        improved = True
                        break
        
        if not improved and swap_iter > 100:
            break
    
    return best_c, best_r


def _init_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _init_fcc_like():
    """FCC-like lattice initialization."""
    a = 1.0 / 4.0
    points = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                x = i * a
                y = j * a
                z = k * a
                points.append([x - 0.5, y - 0.5, z - 0.5])
                points.append([x - 0.5, y + 0.5*a - 0.5, z + 0.5*a - 0.5])
                points.append([x + 0.5*a - 0.5, y - 0.5, z + 0.5*a - 0.5])
                points.append([x + 0.5*a - 0.5, y + 0.5*a - 0.5, z - 0.5])
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    if len(points) < N:
        grid = _init_grid()
        points = np.vstack([points, grid[len(points):]])
    
    return points[:N].copy()


def _init_bcc_like():
    """BCC-like lattice initialization."""
    a = 1.0 / 4.0
    points = []
    for i in range(5):
        for j in range(5):
            for k in range(5):
                x = i * a
                y = j * a
                z = k * a
                points.append([x - 0.5, y - 0.5, z - 0.5])
                points.append([x + 0.5*a - 0.5, y + 0.5*a - 0.5, z + 0.5*a - 0.5])
    
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    
    if len(points) < N:
        grid = _init_grid()
        points = np.vstack([points, grid[len(points):]])
    
    return points[:N].copy()


def _init_random():
    """Random initialization."""
    return np.random.uniform(-0.45, 0.45, (N, 3))


def _init_halton():
    """Halton sequence quasi-random initialization."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    points = []
    for i in range(1, N + 1):
        points.append([halton(i, 2) - 0.5, halton(i, 3) - 0.5, halton(i, 5) - 0.5])
    
    return np.array(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    np.random.seed(42)
    
    initializations = [
        _init_grid,
        _init_fcc_like,
        _init_bcc_like,
        _init_halton,
        _init_random,
    ]
    
    best_centres = None
    best_r = 0.0
    
    for init_fn in initializations:
        if time.time() > deadline - 25.0:
            break
        
        init = init_fn()
        init = np.clip(init, -0.49, 0.49)
        
        # Phase 1: Force-directed refinement to get a good starting point
        c_refined, r_refined = _force_directed_refinement(
            init, max_iters=150, dt_init=0.008, deadline=deadline - 20.0
        )
        
        if r_refined > best_r:
            best_r = r_refined
            best_centres = c_refined.copy()
        
        if time.time() > deadline - 20.0:
            break
        
        # Phase 2: Soft-min differentiable optimization with beta annealing
        c_soft, r_soft = _optimize_soft_min(
            c_refined,
            beta_schedule=[20.0, 50.0, 100.0, 200.0, 400.0],
            deadline=deadline - 15.0
        )
        
        if r_soft > best_r:
            best_r = r_soft
            best_centres = c_soft.copy()
        
        if time.time() > deadline - 15.0:
            break
        
        # Phase 3: Lattice swap local search
        c_swap, r_swap = _lattice_swap_local_search(
            c_soft, max_swaps=300, deadline=deadline - 10.0
        )
        
        if r_swap > best_r:
            best_r = r_swap
            best_centres = c_swap.copy()
    
    # Final refinement from best solution
    if best_centres is not None and time.time() < deadline - 10.0:
        # Additional force refinement
        c_final, r_final = _force_directed_refinement(
            best_centres, max_iters=100, dt_init=0.004, deadline=deadline - 5.0
        )
        if r_final > best_r:
            best_r = r_final
            best_centres = c_final.copy()
        
        # Final soft-min polish
        if time.time() < deadline - 5.0:
            c_polish, r_polish = _optimize_soft_min(
                best_centres,
                beta_schedule=[100.0, 200.0, 500.0],
                deadline=deadline - 2.0
            )
            if r_polish > best_r:
                best_r = r_polish
                best_centres = c_polish.copy()
    
    # Final validation
    if best_centres is not None:
        final_r = _compute_radius(best_centres)
        if final_r > best_r:
            best_r = final_r
        return best_centres
    else:
        return _init_grid()

# EVOLVE-BLOCK-END
