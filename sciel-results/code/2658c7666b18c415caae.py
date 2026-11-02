"""
Packing 200 equal circles in a unit square.
Strategy:
1. Initialize with hexagonal lattice.
2. Global optimization using Log-Sum-Exp surrogate for max-min distance.
3. High-precision local polish focusing on active constraints.
"""

N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time
import warnings
warnings.filterwarnings('ignore')

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf
    return min(min_pair_dist / 2.0, min_boundary)

def _soft_min_energy(params, beta=200.0):
    """
    Minimize the negative log-sum-exp of distances (not squared).
    This is a smooth approximation of -min(d_i).
    We also add a wall penalty to keep points inside.
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]
    
    # Pairwise distances (not squared for better approximation of min)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, 1.0)
    
    # Upper triangle indices for pairs
    iu = np.triu_indices(N, k=1)
    d_pairs = d[iu]
    
    # Log-Sum-Exp trick for stability
    d_shifted = d_pairs - np.min(d_pairs)
    weights = np.exp(-beta * d_shifted)
    p = weights / np.sum(weights)
    
    # Energy: (1/beta) * log(sum(exp(-beta * d_i)))
    # This is LARGE when distances are small, so minimizing spreads points.
    E_pairs = np.log(np.sum(weights)) / beta
    
    # Gradient computation
    grad_x = np.zeros(N)
    grad_y = np.zeros(N)
    
    i_idx, j_idx = iu
    vec_x = x[i_idx] - x[j_idx]
    vec_y = y[i_idx] - y[j_idx]
    
    # dE/d(d_k) = -1 (from log-sum-exp form)
    # d(d_k)/d(c_i_x) = (c_i_x - c_j_x) / d_k
    # So dE/d(c_i_x) = -sum_k p_k * (c_i_x - c_j_x) / d_k
    
    inv_d = 1.0 / np.maximum(d_pairs, 1e-15)
    
    grad_x[i_idx] -= p * vec_x * inv_d
    grad_x[j_idx] += p * vec_x * inv_d
    
    grad_y[i_idx] -= p * vec_y * inv_d
    grad_y[j_idx] += p * vec_y * inv_d
    
    # Wall Penalty - use soft-min for boundaries too
    d_left = 0.5 - x
    d_right = 0.5 + x
    d_down = 0.5 - y
    d_up = 0.5 + y
    
    # Soft min of boundary distances
    d_all = np.stack([d_left, d_right, d_down, d_up], axis=1)  # (N, 4)
    d_min = np.min(d_all, axis=1, keepdims=True)
    wall_weights = np.exp(-beta * (d_all - d_min))  # (N, 4)
    wall_weights /= wall_weights.sum(axis=1, keepdims=True)
    
    # Actually, let's use a simpler boundary energy
    # Penalize points that are too close to any wall
    eps = 1e-8
    d_left_safe = np.clip(d_left, eps, None)
    d_right_safe = np.clip(d_right, eps, None)
    d_down_safe = np.clip(d_down, eps, None)
    d_up_safe = np.clip(d_up, eps, None)
    
    # Use inverse square penalty for strong repulsion from walls
    E_wall = (1.0/d_left_safe**2 + 1.0/d_right_safe**2 + 1.0/d_down_safe**2 + 1.0/d_up_safe**2).sum()
    
    # Gradient of wall part
    grad_x += -2.0 / d_left_safe**3 + 2.0 / d_right_safe**3
    grad_y += -2.0 / d_down_safe**3 + 2.0 / d_up_safe**3
    
    # Combine energies
    E_total = E_wall + E_pairs
    
    total_grad = np.concatenate([grad_x, grad_y])
    return E_total, total_grad


def _logdet_barrier_energy(params, gamma=5.0):
    """
    Log-determinant barrier: maximize log(det(G)) where G is the Gram matrix
    of centered positions. This promotes uniform spread of points.
    We minimize -log(det(G)) + wall penalties.
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]
    
    # Center positions
    cx = x - np.mean(x)
    cy = y - np.mean(y)
    
    # Build Gram matrix G = C^T C where C is (N, 2)
    # G is 2x2: [[sum(cx^2), sum(cx*cy)], [sum(cx*cy), sum(cy^2)]]
    G00 = np.sum(cx * cx)
    G01 = np.sum(cx * cy)
    G11 = np.sum(cy * cy)
    
    # log(det(G)) = log(G00*G11 - G01^2)
    det = G00 * G11 - G01 * G01
    det = max(det, 1e-30)
    
    # Energy: -gamma * log(det) + wall penalties
    E_logdet = -gamma * np.log(det)
    
    # Wall penalties (soft)
    d_left = 0.5 - x
    d_right = 0.5 + x
    d_down = 0.5 - y
    d_up = 0.5 + y
    
    eps = 1e-8
    d_left_safe = np.clip(d_left, eps, None)
    d_right_safe = np.clip(d_right, eps, None)
    d_down_safe = np.clip(d_down, eps, None)
    d_up_safe = np.clip(d_up, eps, None)
    
    E_wall = (1.0/d_left_safe**2 + 1.0/d_right_safe**2 + 1.0/d_down_safe**2 + 1.0/d_up_safe**2).sum()
    
    E_total = E_logdet + E_wall
    
    # Gradient of logdet term:
    # d(-gamma*log(det))/dx_i = -gamma * (1/det) * d(det)/dx_i
    # d(det)/dx_i = 2*cx_i*(G11) - 2*cy_i*(G01)  ... let me redo this
    # det = G00*G11 - G01^2
    # d(det)/d(cx_i) = 2*cx_i*G11 - 2*G01*cy_i
    # d(det)/d(cy_i) = 2*cy_i*G00 - 2*G01*cx_i
    
    d_det_dx = 2.0 * cx * G11 - 2.0 * G01 * cy
    d_det_dy = 2.0 * cy * G00 - 2.0 * G01 * cx
    
    grad_x = -gamma * d_det_dx / det
    grad_y = -gamma * d_det_dy / det
    
    # Wall gradient
    grad_x += -2.0 / d_left_safe**3 + 2.0 / d_right_safe**3
    grad_y += -2.0 / d_down_safe**3 + 2.0 / d_up_safe**3
    
    total_grad = np.concatenate([grad_x, grad_y])
    return E_total, total_grad

def _initialize_hexagonal():
    """Initialize with hexagonal pattern - optimized search over parameters."""
    best_c = None
    best_r = 0.0
    
    # Try different densities with finer granularity around optimal region
    for d in np.linspace(0.052, 0.075, 50):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            c = np.clip(c, -0.4999, 0.4999)
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    # Fallback grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _optimize_energy(c0, time_budget=20.0, beta=100.0):
    """Run L-BFGS-B with Log-Sum-Exp energy."""
    start = time.time()
    bounds = [(-0.49999, 0.49999)] * (2 * N)
    
    try:
        # Wrap to handle beta
        def fun_and_grad(params):
            return _soft_min_energy(params, beta=beta)
            
        res = minimize(
            fun_and_grad,
            c0.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxfun': 5000, 'maxls': 100}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
        return c_opt
    except Exception:
        return c0


def _optimize_logdet(c0, time_budget=15.0, gamma=5.0):
    """Run L-BFGS-B with log-determinant barrier energy."""
    start = time.time()
    bounds = [(-0.49999, 0.49999)] * (2 * N)
    
    try:
        def fun_and_grad(params):
            return _logdet_barrier_energy(params, gamma=gamma)
            
        res = minimize(
            fun_and_grad,
            c0.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 3000, 'maxls': 80}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
        return c_opt
    except Exception:
        return c0

def _slsqp_active_polish(c, r_target, time_limit=5.0):
    """
    Uses SLSQP to find a configuration where all active constraints are satisfied
    for a target radius r_target. Fully vectorized constraints for speed.
    """
    start_time = time.time()
    c = c.copy()
    
    # Identify active constraints with tighter margin
    tol = 1e-6
    margin = 2.1 * r_target  # tighter margin
    
    dists = pdist(c, metric='euclidean')
    dist_mat = squareform(dists)
    
    # Active pairs: distance < margin
    active_mask = dist_mat < margin
    np.fill_diagonal(active_mask, False)
    
    active_pairs = np.argwhere(active_mask)
    active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
    
    # Active bounds - only include nearby boundaries
    d_left = 0.5 - c[:, 0]
    d_right = 0.5 + c[:, 0]
    d_down = 0.5 - c[:, 1]
    d_up = 0.5 + c[:, 1]
    
    active_bounds = np.where(
        np.minimum(np.minimum(d_left, d_right), np.minimum(d_down, d_up)) < (r_target + tol)
    )[0]
    
    if len(active_pairs) == 0 and len(active_bounds) == 0:
        return c, False

    # Vectorized constraint functions
    n_pairs = len(active_pairs)
    n_bounds = len(active_bounds)
    total_cons = n_pairs + n_bounds
    
    if total_cons == 0:
        return c, False
    
    # Precompute indices
    if n_pairs > 0:
        pair_i = active_pairs[:, 0].astype(int)
        pair_j = active_pairs[:, 1].astype(int)
    else:
        pair_i = np.array([], dtype=int)
        pair_j = np.array([], dtype=int)
    
    # For bounds, determine which wall is closest
    bound_wall_type = np.zeros(n_bounds, dtype=int)  # 0=left, 1=right, 2=down, 3=up
    if n_bounds > 0:
        for idx, i in enumerate(active_bounds):
            dl = d_left[i]
            dr = d_right[i]
            dd = d_down[i]
            du = d_up[i]
            min_d = min(dl, dr, dd, du)
            if dl == min_d:
                bound_wall_type[idx] = 0
            elif dr == min_d:
                bound_wall_type[idx] = 1
            elif dd == min_d:
                bound_wall_type[idx] = 2
            else:
                bound_wall_type[idx] = 3
    
    r2 = (2 * r_target) ** 2
    
    # Precompute bound indices for vectorized evaluation
    if n_bounds > 0:
        bound_idx = active_bounds.astype(int)
        # wall_type: 0=left, 1=right, 2=down, 3=up
        # For left/right: use x coordinate (index 2*i)
        # For down/up: use y coordinate (index 2*i+1)
        coord_idx = np.where(bound_wall_type <= 1, 2 * bound_idx, 2 * bound_idx + 1)
        sign = np.where(bound_wall_type <= 1, 1, 1)  # all positive for now, handled in func
        # Actually: left: 0.5 - x - r, right: x + 0.5 - r, down: 0.5 - y - r, up: y + 0.5 - r
        # So: left/down: 0.5 - coord - r, right/up: coord + 0.5 - r
        is_left_down = (bound_wall_type == 0) | (bound_wall_type == 2)
    
    def constraint_func(x):
        """Fully vectorized constraint evaluation."""
        vals = np.empty(total_cons)
        if n_pairs > 0:
            xi = x[2*pair_i:2*pair_i+2, :].reshape(n_pairs, 2)
            xj = x[2*pair_j:2*pair_j+2, :].reshape(n_pairs, 2)
            diff = xi - xj
            vals[:n_pairs] = np.sum(diff**2, axis=1) - r2
        if n_bounds > 0:
            coords = x[coord_idx]
            # left/down: 0.5 - coord - r_target
            # right/up: coord + 0.5 - r_target
            vals[n_pairs:] = np.where(is_left_down, 0.5 - coords - r_target, coords + 0.5 - r_target)
        return vals
    
    cons = [{'type': 'ineq', 'fun': constraint_func}]
    
    bounds = [(-0.5 + r_target, 0.5 - r_target)] * (2 * N)
    
    def objective(x):
        return 0.0

    try:
        res = minimize(
            objective,
            c.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 120, 'ftol': 1e-14}
        )
        
        if res.success:
            c_new = res.x.reshape(N, 2)
            r_new = _compute_radius(c_new)
            if r_new >= r_target - 1e-10:
                return c_new, True
    except Exception:
        pass
        
    return c, False

def _local_polish(c, time_limit=30.0):
    """
    Fast local polish using active-constraint gradient ascent on radius.
    Uses a smooth penalty-based approach for speed.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    
    def _logdet_polish_step(c, start_time, time_limit, gamma=5.0):
        """
        One step of log-determinant barrier optimization to improve uniformity.
        """
        if time.time() - start_time > time_limit - 3:
            return c, _compute_radius(c)
        
        try:
            bounds = [(-0.49999, 0.49999)] * (2 * N)
            res = minimize(
                lambda p: _logdet_barrier_energy(p, gamma=gamma),
                c.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-11, 'maxfun': 1500}
            )
            c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
            r_opt = _compute_radius(c_opt)
            if r_opt > _compute_radius(c):
                return c_opt, r_opt
        except Exception:
            pass
        return c, _compute_radius(c)
    
    def _rigid_body_perturb(c, rng, radius=0.15, max_clusters=5):
        """
        Move clusters of nearby circles simultaneously to preserve local structure.
        Returns a perturbed copy of c.
        """
        c_new = c.copy()
        n = len(c_new)
        visited = np.zeros(n, dtype=bool)
        
        # Find clusters based on proximity
        dx = c_new[:, np.newaxis, 0] - c_new[np.newaxis, :, 0]
        dy = c_new[:, np.newaxis, 1] - c_new[np.newaxis, :, 1]
        dist = np.sqrt(dx**2 + dy**2)
        
        clusters = []
        for i in range(n):
            if visited[i]:
                continue
            # BFS to find connected component within radius
            cluster = [i]
            visited[i] = True
            queue = [i]
            while queue:
                j = queue.pop()
                neighbors = np.where(dist[j] < radius)[0]
                for k in neighbors:
                    if not visited[k]:
                        visited[k] = True
                        cluster.append(k)
                        queue.append(k)
            clusters.append(cluster)
        
        # Select up to max_clusters clusters to perturb
        if len(clusters) > max_clusters:
            idx = rng.choice(len(clusters), size=max_clusters, replace=False)
            selected_clusters = [clusters[i] for i in idx]
        else:
            selected_clusters = clusters
        
        for cluster in selected_clusters:
            if len(cluster) < 2:
                continue
            # Generate a random direction and magnitude
            angle = rng.uniform(0, 2 * np.pi)
            magnitude = rng.uniform(0.003, 0.015)
            dx_move = magnitude * np.cos(angle)
            dy_move = magnitude * np.sin(angle)
            
            for i in cluster:
                c_new[i, 0] += dx_move
                c_new[i, 1] += dy_move
        
        # Clip to stay inside the square
        c_new = np.clip(c_new, -0.49999, 0.49999)
        return c_new
    
    def _annealed_polish(c, start_time, time_limit, beta_start=150.0, beta_end=1200.0):
        """
        Multi-temperature annealing polish with rigid-body perturbations.
        """
        c = c.copy()
        current_r = _compute_radius(c)
        rng = np.random.RandomState(42)
        
        # Temperature schedule: more levels for gradual refinement
        n_steps = 12
        betas = np.geomspace(beta_start, beta_end, n_steps)
        
        for beta in betas:
            if time.time() - start_time > time_limit - 5:
                break
            
            # Run L-BFGS-B at this temperature
            try:
                res = minimize(
                    lambda p: _polish_energy(p, beta=beta),
                    c.flatten(),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-0.49999, 0.49999)] * (2 * N),
                    options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 800}
                )
                c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
                r_opt = _compute_radius(c_opt)
                if r_opt > current_r:
                    c = c_opt
                    current_r = r_opt
            except Exception:
                pass
            
            # Rigid-body perturbation to escape local minima
            if time.time() - start_time > time_limit - 10:
                break
            
            c_perturbed = _rigid_body_perturb(c, rng, radius=0.10, max_clusters=4)
            r_perturbed = _compute_radius(c_perturbed)
            
            # Accept if better or with some probability (simulated annealing style)
            if r_perturbed > current_r:
                c = c_perturbed
                current_r = r_perturbed
            else:
                # Accept worse solutions with decreasing probability
                delta_r = current_r - r_perturbed
                T = 0.00008 * (1.0 - (time.time() - start_time) / time_limit)
                if delta_r < T and rng.random() < np.exp(-delta_r / max(T, 1e-10)):
                    c = c_perturbed
                    current_r = r_perturbed
        
        return c, current_r
    
    def _polish_energy(params, beta=300.0):
        """Energy that maximizes the minimum distance using log-sum-exp."""
        c_ = params.reshape(N, 2)
        x = c_[:, 0]
        y = c_[:, 1]
        
        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, 1.0)
        
        iu = np.triu_indices(N, k=1)
        d_pairs = d[iu]
        
        # Boundary distances
        bdists = np.minimum(
            np.minimum(0.5 - x, 0.5 + x),
            np.minimum(0.5 - y, 0.5 + y)
        )
        
        # We want to maximize min(d_pairs/2, bdists)
        # Use log-sum-exp to get a smooth approximation
        # E = (1/beta) * log(sum(exp(-beta * d)))
        
        # For pairs (use d/2 since radius = d/2)
        d_shifted = d_pairs - np.min(d_pairs)
        w_pairs = np.exp(-beta * d_shifted)
        E_pairs = np.log(np.sum(w_pairs)) / beta
        
        # For boundaries
        bd_shifted = bdists - np.min(bdists)
        w_bounds = np.exp(-beta * bd_shifted)
        E_bounds = np.log(np.sum(w_bounds)) / beta
        
        E = E_pairs + E_bounds
        
        # Gradient
        p_pairs = w_pairs / np.sum(w_pairs)
        p_bounds = w_bounds / np.sum(w_bounds)
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        i_idx, j_idx = iu
        vec_x = x[i_idx] - x[j_idx]
        vec_y = y[i_idx] - y[j_idx]
        
        inv_d = 1.0 / np.maximum(d_pairs, 1e-15)
        
        # dE/d(d_k) = -1
        # d(d_k)/d(c_i) = (c_i - c_j) / d_k
        grad_x[i_idx] -= p_pairs * vec_x * inv_d
        grad_x[j_idx] += p_pairs * vec_x * inv_d
        grad_y[i_idx] -= p_pairs * vec_y * inv_d
        grad_y[j_idx] += p_pairs * vec_y * inv_d
        
        # Boundary gradient
        d_left = 0.5 - x
        d_right = 0.5 + x
        d_down = 0.5 - y
        d_up = 0.5 + y
        
        d_all = np.stack([d_left, d_right, d_down, d_up], axis=1)
        d_min = np.min(d_all, axis=1, keepdims=True)
        wall_weights = np.exp(-beta * (d_all - d_min))
        wall_weights /= wall_weights.sum(axis=1, keepdims=True)
        
        grad_x += p_bounds * (-wall_weights[:, 0] + wall_weights[:, 1])
        grad_y += p_bounds * (-wall_weights[:, 2] + wall_weights[:, 3])
        
        total_grad = np.concatenate([grad_x, grad_y])
        return E, total_grad
    
    # Run annealed polish with rigid-body perturbations
    if time.time() - start_time < time_limit * 0.65:
        c_annealed, r_annealed = _annealed_polish(c, start_time, time_limit * 0.65, beta_start=150.0, beta_end=1200.0)
        if r_annealed > current_r:
            c = c_annealed
            current_r = r_annealed
    
    # Log-determinant barrier step to improve uniformity
    if time.time() - start_time < time_limit * 0.8:
        c_ld, r_ld = _logdet_polish_step(c, start_time, time_limit, gamma=5.0)
        if r_ld > current_r:
            c = c_ld
            current_r = r_ld
        # Second logdet step with different gamma
        if time.time() - start_time < time_limit * 0.85:
            c_ld2, r_ld2 = _logdet_polish_step(c, start_time, time_limit, gamma=3.0)
            if r_ld2 > current_r:
                c = c_ld2
                current_r = r_ld2
    
    # Final high-precision polish with very high beta
    if time.time() - start_time < time_limit - 5:
        try:
            res = minimize(
                lambda p: _polish_energy(p, beta=2000.0),
                c.flatten(),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.49999, 0.49999)] * (2 * N),
                options={'maxiter': 1500, 'ftol': 1e-18, 'gtol': 1e-16, 'maxfun': 8000}
            )
            c_opt = np.clip(res.x.reshape(N, 2), -0.49999, 0.49999)
            r_opt = _compute_radius(c_opt)
            if r_opt > current_r:
                c = c_opt
                current_r = r_opt
        except Exception:
            pass
    
    # SLSQP Active-Set Polish - try multiple steps with decreasing sizes
    if time.time() - start_time < time_limit - 1:
        current_r = _compute_radius(c)
        # Try to increase radius in small steps using SLSQP
        for step in [2e-6, 1e-6, 5e-7, 2e-7, 1e-7, 5e-8, 2e-8]:
            if time.time() - start_time > time_limit - 1:
                break
            target_r = current_r + step
            c_new, success = _slsqp_active_polish(c, target_r, time_limit=4.0)
            if success:
                c = c_new
                current_r = _compute_radius(c)
            else:
                # If it fails, we might be at the limit or SLSQP failed.
                # Try smaller step?
                # For now, break to save time.
                break
    
    return c, current_r

def construct():
    start_time = time.time()
    deadline = start_time + 148
    
    best_c = _initialize_hexagonal()
    best_r = _compute_radius(best_c)
    
    # Run global optimization with multiple beta values
    for beta in [40.0, 80.0, 120.0, 180.0]:
        if time.time() > deadline - 95:
            break
        c_opt = _optimize_energy(best_c, time_budget=18.0, beta=beta)
        r_opt = _compute_radius(c_opt)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
    
    # Log-determinant barrier optimization phase
    if time.time() < deadline - 85:
        for gamma in [5.0, 3.0, 8.0]:
            if time.time() > deadline - 85:
                break
            c_ld = _optimize_logdet(best_c, time_budget=12.0, gamma=gamma)
            r_ld = _compute_radius(c_ld)
            if r_ld > best_r:
                best_r = r_ld
                best_c = c_ld.copy()
    
    # Try multiple perturbed restarts if time allows
    remaining = deadline - time.time() - 70
    if remaining > 25:
        rng = np.random.RandomState(42)
        for attempt in range(3):
            if time.time() > deadline - 70:
                break
            perturbation = rng.randn(N, 2) * 0.008
            c_perturbed = np.clip(best_c + perturbation, -0.4999, 0.4999)
            
            c_opt2 = _optimize_energy(c_perturbed, time_budget=12.0, beta=100.0)
            r_opt2 = _compute_radius(c_opt2)
            
            if r_opt2 > best_r:
                best_r = r_opt2
                best_c = c_opt2.copy()
    
    # Local polish - use most of the remaining time
    remaining = deadline - time.time() - 5
    if remaining > 15:
        c_polished, r_polished = _local_polish(best_c, time_limit=min(remaining, 80.0))
        if r_polished > best_r:
            best_r = r_polished
            best_c = c_polished.copy()
    
    # Second polish pass if time allows
    remaining = deadline - time.time() - 3
    if remaining > 15:
        c_polished2, r_polished2 = _local_polish(best_c, time_limit=min(remaining, 40.0))
        if r_polished2 > best_r:
            best_r = r_polished2
            best_c = c_polished2.copy()
    
    # Third polish pass with even tighter tolerances
    remaining = deadline - time.time() - 2
    if remaining > 10:
        c_polished3, r_polished3 = _local_polish(best_c, time_limit=min(remaining, 25.0))
        if r_polished3 > best_r:
            best_r = r_polished3
            best_c = c_polished3.copy()
    
    # Final ultra-precise SLSQP push if time remains
    remaining = deadline - time.time() - 1
    if remaining > 3:
        current_r = _compute_radius(best_c)
        for step in [1e-7, 5e-8, 2e-8, 1e-8]:
            if time.time() > deadline - 1:
                break
            target_r = current_r + step
            c_new, success = _slsqp_active_polish(best_c, target_r, time_limit=3.0)
            if success:
                best_c = c_new
                current_r = _compute_radius(best_c)
            else:
                break
    
    # Final safety check
    best_c = np.clip(best_c, -0.49999, 0.49999)
    
    return best_c
