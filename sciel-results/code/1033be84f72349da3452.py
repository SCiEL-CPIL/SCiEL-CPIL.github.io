"""Pack 100 equal spheres in a unit cube [-0.5, 0.5]^3 to maximize minimum radius.
Combines soft-min energy landscape with iterative radius-increase (bisection) and
Delaunay-based contact-graph refinement via rigidity-style edge optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
HALF = 0.5

def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    centers = np.asarray(centers)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    wall_dist = HALF - np.max(np.abs(centers), axis=1)
    r_wall = np.min(wall_dist)
    if r_wall < 0:
        return 0.0
    # Pairwise distances via KDTree for speed
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair_dist = np.min(dists[:, 1])
    r_pair = min_pair_dist / 2.0
    return min(r_wall, r_pair)

def _softmin_obj_and_grad(centers, beta=50.0):
    """
    Objective: soft-min of all constraint radii (pairwise/2 and wall distances).
    We MINIMIZE this, which MAXIMIZES the soft-min of radii.
    Returns (value, gradient_flat).
    """
    centers = centers.reshape(N, 3)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    dists = np.sqrt(sq_dists)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(sq_dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_r = dists[i_idx, j_idx] / 2.0
    sq_pair = sq_dists[i_idx, j_idx]
    
    # Wall distances
    wall_r = HALF - np.max(np.abs(centers), axis=1)
    
    all_r = np.concatenate([pair_r, wall_r])
    n_pairs = len(i_idx)
    
    # Soft-min: f = -(1/beta) * log(sum(exp(-beta * r)))
    # For numerical stability
    x = -beta * all_r
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    sum_exp = np.sum(exp_x)
    log_sum = x_max + np.log(sum_exp)
    softmin_val = -log_sum / beta
    
    # Weights (softmax of -beta*r)
    weights = exp_x / sum_exp
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # Pairwise contributions
    # d(pair_r_k)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|)
    # d(pair_r_k)/d(c_j) = (c_j - c_i) / (2 * |c_i - c_j|)
    for k in range(n_pairs):
        i, j = i_idx[k], j_idx[k]
        w = weights[k]
        d = centers[i] - centers[j]
        dist = dists[i, j]
        if dist < 1e-12:
            continue
        inv = 1.0 / (2.0 * dist)
        grad[i] += w * d * inv
        grad[j] -= w * d * inv
    
    # Wall contributions
    # wall_r_i = 0.5 - max_k |c_ik|
    # d(wall_r_i)/d(c_ik) = -sign(c_ik) for k = argmax|c_ik|
    abs_c = np.abs(centers)
    k_max = np.argmax(abs_c, axis=1)
    for i in range(N):
        w = weights[n_pairs + i]
        k = k_max[i]
        grad[i, k] += w * (-np.sign(centers[i, k]))
    
    return softmin_val, grad.flatten()

def _repulsion_energy_and_grad(centers, target_r, beta=30.0):
    """
    Energy that penalizes violations of radius target_r.
    Uses softplus for smoothness.
    E = sum_{pairs} softplus(beta*(2r - d_ij))^2 + sum_i softplus(beta*(r - w_i))^2
    """
    centers = centers.reshape(N, 3)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    dists = np.sqrt(sq_dists)
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(sq_dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = dists[i_idx, j_idx]
    n_pairs = len(i_idx)
    
    wall_dists = HALF - np.max(np.abs(centers), axis=1)
    
    # Violations
    pair_viol = 2.0 * target_r - pair_dists  # positive when overlapping
    wall_viol = target_r - wall_dists        # positive when too close to wall
    
    # Softplus: sp(x) = log(1 + exp(x)), stable version
    def _softplus(x):
        return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))
    
    def _dsoftplus_dx(x):
        # sigmoid(x) = 1/(1+exp(-x))
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    sp_pair = _softplus(beta * pair_viol)
    sp_wall = _softplus(beta * wall_viol)
    
    energy = np.sum(sp_pair ** 2) + np.sum(sp_wall ** 2)
    
    # Gradient
    grad = np.zeros_like(centers)
    
    # dE/d(pair_viol_k) = 2 * sp_pair_k * dsoftplus/dx * beta
    dE_dpv = 2.0 * sp_pair * _dsoftplus_dx(beta * pair_viol) * beta
    # d(pair_viol_k)/d(c_i) = -(c_i - c_j) / d_ij
    # d(pair_viol_k)/d(c_j) = +(c_i - c_j) / d_ij
    
    for k in range(n_pairs):
        i, j = i_idx[k], j_idx[k]
        coeff = dE_dpv[k]
        d = centers[i] - centers[j]
        dist = pair_dists[k]
        if dist < 1e-12:
            continue
        inv = 1.0 / dist
        grad[i] -= coeff * d * inv
        grad[j] += coeff * d * inv
    
    # Wall
    dE_dwv = 2.0 * sp_wall * _dsoftplus_dx(beta * wall_viol) * beta
    abs_c = np.abs(centers)
    k_max = np.argmax(abs_c, axis=1)
    for i in range(N):
        coeff = dE_dwv[i]
        k = k_max[i]
        # d(wall_viol_i)/d(c_ik) = sign(c_ik) for k=argmax
        grad[i, k] += coeff * np.sign(centers[i, k])
    
    return energy, grad.flatten()

def _local_optimize_softmin(centers, max_time=10.0, beta=50.0):
    """Run L-BFGS-B local optimization using soft-min objective."""
    start_time = time.time()
    
    def objective(x):
        val, _ = _softmin_obj_and_grad(x, beta)
        return val
    
    def grad(x):
        _, g = _softmin_obj_and_grad(x, beta)
        return g
    
    bounds = [(-HALF, HALF)] * (N * 3)
    
    result = minimize(
        objective,
        centers.flatten(),
        jac=grad,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    elapsed = time.time() - start_time
    best_centers = result.x.reshape(N, 3)
    best_radius = _compute_radius(best_centers)
    
    return best_centers, best_radius, elapsed

def _local_optimize_repulsion(centers, target_r, max_time=10.0, beta=30.0):
    """Run L-BFGS-B local optimization using repulsion energy for fixed target_r."""
    start_time = time.time()
    
    def objective(x):
        val, _ = _repulsion_energy_and_grad(x, target_r, beta)
        return val
    
    def grad(x):
        _, g = _repulsion_energy_and_grad(x, target_r, beta)
        return g
    
    bounds = [(-HALF, HALF)] * (N * 3)
    
    result = minimize(
        objective,
        centers.flatten(),
        jac=grad,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    elapsed = time.time() - start_time
    best_centers = result.x.reshape(N, 3)
    best_radius = _compute_radius(best_centers)
    
    return best_centers, best_radius, elapsed

def _generate_initial_points(method='grid', rng=None):
    """Generate initial center positions."""
    if rng is None:
        rng = np.random.default_rng()
    
    if method == 'grid':
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()
    elif method == 'random':
        return rng.uniform(-0.45, 0.45, size=(N, 3))
    elif method == 'bcc':
        points = []
        for ix in range(5):
            for iy in range(5):
                for iz in range(4):
                    x = (ix + 0.5) / 5.0 - 0.5
                    y = (iy + 0.5) / 5.0 - 0.5
                    z = (iz + 0.5) / 4.0 - 0.5
                    points.append([x, y, z])
                    xc = (ix + 1.0) / 5.0 - 0.5
                    yc = (iy + 1.0) / 5.0 - 0.5
                    zc = (iz + 1.0) / 4.0 - 0.5
                    points.append([xc, yc, zc])
        return np.array(points[:N])
    else:
        return rng.uniform(-0.45, 0.45, size=(N, 3))

def _contact_graph_refine(centers, radius, max_time=8.0):
    """
    Delaunay/contact-graph-based refinement: identify near-contact pairs,
    then optimize only those edges to push the minimum radius up.
    This is a rigidity-inspired approach: treat near-contact edges as
    constraints and adjust positions to satisfy them better.
    """
    start_time = time.time()
    centers = centers.copy()
    
    # Build KDTree to find near-contact pairs
    tree = cKDTree(centers)
    # Find all pairs within 2.1 * radius (slightly above contact distance)
    pairs = tree.query_pairs(2.1 * radius)
    
    if len(pairs) == 0:
        return centers, _compute_radius(centers)
    
    # Create a sub-problem: optimize only positions of spheres involved in near-contacts
    # to maximize the minimum distance among those pairs, while keeping others fixed.
    # This is a simplified rigidity step.
    
    involved = set()
    for i, j in pairs:
        involved.add(i)
        involved.add(j)
    involved = sorted(involved)
    
    if len(involved) < 2:
        return centers, _compute_radius(centers)
    
    # For the involved spheres, run a small local optimization
    # that maximizes the minimum pairwise distance among near-contact pairs
    # while keeping wall constraints.
    
    idx_array = np.array(involved)
    
    def sub_objective(x):
        c = centers.copy()
        c[idx_array] = x.reshape(len(involved), 3)
        # Compute min over near-contact pairs
        min_d = np.inf
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_d:
                min_d = d
        # Also wall constraint
        wall = HALF - np.max(np.abs(c[idx_array]), axis=1)
        return -(min_d / 2.0 + np.min(wall))
    
    def sub_grad(x):
        c = centers.copy()
        c[idx_array] = x.reshape(len(involved), 3)
        
        # Find which pair is the minimum
        min_d = np.inf
        min_pair = None
        for i, j in pairs:
            d = np.linalg.norm(c[i] - c[j])
            if d < min_d:
                min_d = d
                min_pair = (i, j)
        
        wall = HALF - np.max(np.abs(c[idx_array]), axis=1)
        min_wall_idx = np.argmin(wall)
        
        grad_full = np.zeros((N, 3))
        
        if min_pair is not None:
            i, j = min_pair
            d = c[i] - c[j]
            dist = min_d
            if dist > 1e-12:
                grad_full[i] += d / (2.0 * dist)
                grad_full[j] -= d / (2.0 * dist)
        
        # Wall gradient
        i_wall = idx_array[min_wall_idx]
        abs_c = np.abs(c[i_wall])
        k_max = np.argmax(abs_c)
        grad_full[i_wall, k_max] += -np.sign(c[i_wall, k_max])
        
        return grad_full[idx_array].flatten()
    
    x0 = centers[idx_array].flatten()
    bounds_sub = [(-HALF, HALF)] * (len(involved) * 3)
    
    try:
        result = minimize(
            sub_objective,
            x0,
            jac=sub_grad,
            method='L-BFGS-B',
            bounds=bounds_sub,
            options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-7}
        )
        new_sub = result.x.reshape(len(involved), 3)
        centers[idx_array] = new_sub
    except Exception:
        pass
    
    # Clip
    centers = np.clip(centers, -HALF, HALF)
    elapsed = time.time() - start_time
    return centers, _compute_radius(centers)

def construct():
    """Construct centers of 100 equal spheres in [-0.5, 0.5]^3 to maximize radius."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_centers = None
    best_radius = 0.0
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Try multiple initializations with soft-min optimization
    initializations = []
    
    # Grid
    grid_centers = _generate_initial_points('grid', rng)
    initializations.append(grid_centers)
    
    # BCC
    bcc_centers = _generate_initial_points('bcc', rng)
    initializations.append(bcc_centers)
    
    # Perturbed grids
    for i in range(6):
        if time.time() > deadline - 40.0:
            break
        base = _generate_initial_points('grid', rng)
        noise_scale = 0.03 + 0.015 * i
        perturbed = base + rng.normal(0, noise_scale, size=(N, 3))
        perturbed = np.clip(perturbed, -0.48, 0.48)
        initializations.append(perturbed)
    
    # Random
    for i in range(4):
        if time.time() > deadline - 30.0:
            break
        random_c = _generate_initial_points('random', rng)
        initializations.append(random_c)
    
    # Optimize each initialization
    for init in initializations:
        if time.time() > deadline - 30.0:
            break
        
        # First pass: soft-min with moderate beta
        centers, radius, elapsed = _local_optimize_softmin(init, max_time=12.0, beta=40.0)
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
        
        # Second pass: higher beta for precision
        if time.time() > deadline - 25.0:
            break
        centers2, radius2, elapsed2 = _local_optimize_softmin(centers, max_time=8.0, beta=80.0)
        if radius2 > best_radius:
            best_radius = radius2
            best_centers = centers2.copy()
    
    # Phase 2: Iterative radius increase via bisection + repulsion energy
    if best_centers is not None and time.time() < deadline - 20.0:
        # We have a good starting point. Try to push radius higher.
        r_low = best_radius
        r_high = best_radius * 1.15  # Try 15% higher
        
        for bisection_iter in range(8):
            if time.time() > deadline - 10.0:
                break
            
            r_mid = (r_low + r_high) / 2.0
            
            # Try to find a configuration with radius >= r_mid
            # Use repulsion energy minimization starting from best_centers
            centers_test, radius_test, elapsed = _local_optimize_repulsion(
                best_centers, r_mid, max_time=10.0, beta=40.0
            )
            
            actual_r = _compute_radius(centers_test)
            if actual_r >= r_mid * 0.999:
                # Success: we can achieve at least r_mid
                r_low = r_mid
                if actual_r > best_radius:
                    best_radius = actual_r
                    best_centers = centers_test.copy()
            else:
                r_high = r_mid
            
            # Also try from a slightly perturbed version
            if time.time() > deadline - 10.0:
                break
            perturbed = best_centers + rng.normal(0, 0.01, size=(N, 3))
            perturbed = np.clip(perturbed, -0.48, 0.48)
            centers_test2, radius_test2, elapsed2 = _local_optimize_repulsion(
                perturbed, r_mid, max_time=8.0, beta=40.0
            )
            actual_r2 = _compute_radius(centers_test2)
            if actual_r2 >= r_mid * 0.999:
                r_low = r_mid
                if actual_r2 > best_radius:
                    best_radius = actual_r2
                    best_centers = centers_test2.copy()
    
    # Phase 3: Contact-graph refinement (rigidity-inspired)
    if best_centers is not None and time.time() < deadline - 8.0:
        for refine_iter in range(3):
            if time.time() > deadline - 5.0:
                break
            best_centers, r_refined = _contact_graph_refine(
                best_centers, best_radius, max_time=6.0
            )
            if r_refined > best_radius:
                best_radius = r_refined
            else:
                break
    
    # Phase 4: Final polish with very high beta soft-min
    if best_centers is not None and time.time() < deadline - 5.0:
        centers_final, radius_final, _ = _local_optimize_softmin(
            best_centers, max_time=5.0, beta=120.0
        )
        if radius_final > best_radius:
            best_radius = radius_final
            best_centers = centers_final.copy()
    
    # Ensure validity
    if best_centers is None:
        best_centers = _generate_initial_points('grid', rng)
    
    best_centers = np.clip(best_centers, -HALF, HALF)
    final_radius = _compute_radius(best_centers)
    if final_radius > 0:
        return best_centers
    
    return _generate_initial_points('grid', rng)
