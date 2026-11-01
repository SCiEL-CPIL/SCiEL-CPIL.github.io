import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100

def _compute_radius(centers):
    """Compute the largest common radius for the given centers in [-0.5, 0.5]^3."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 3):
        return 0.0
    dists_to_walls = 0.5 - np.abs(centers)
    min_wall_dist = np.min(dists_to_walls)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    return min(min_wall_dist, min_pair_dist / 2.0)


def _force_directed_step(centers, k, dt, r_target):
    """
    One force-directed step. For pairs closer than 2*r_target, push apart.
    For spheres too close to walls, push inward.
    Uses a repulsive force ~ 1/d^2 for close pairs.
    """
    c = centers.copy()
    forces = np.zeros_like(c)
    
    # Pair forces using KD-tree for efficiency
    tree = cKDTree(c)
    pairs = tree.query_pairs(r=2 * r_target * 1.5)
    
    for (i, j) in pairs:
        d = c[i] - c[j]
        dist = np.linalg.norm(d)
        if dist < 1e-12:
            # Random perturbation for coincident points
            d = np.random.randn(3) * 1e-6
            dist = np.linalg.norm(d)
        # Repulsive force: stronger when closer
        if dist < 2 * r_target:
            f = k / (dist * dist)
            forces[i] += f * d / dist
            forces[j] -= f * d / dist
    
    # Wall forces
    for i in range(N):
        for axis in range(3):
            # Left wall (x < -0.5)
            if c[i, axis] < -0.5 + r_target:
                forces[i, axis] += k * (r_target - (c[i, axis] + 0.5))
            # Right wall (x > 0.5)
            if c[i, axis] > 0.5 - r_target:
                forces[i, axis] -= k * (r_target - (0.5 - c[i, axis]))
    
    c += dt * forces
    c = np.clip(c, -0.4999, 0.4999)
    return c


def _simulated_annealing(centers, deadline, r_target_init=None):
    """
    Simulated annealing with force-directed moves.
    Accept moves that increase radius or with probability exp(-delta/T).
    """
    c = centers.copy()
    r = _compute_radius(c)
    best_c = c.copy()
    best_r = r
    
    if r_target_init is None:
        r_target_init = max(r, 0.05)
    
    T_start = 0.05
    T_end = 0.0001
    n_steps = 5000
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        T = T_start * (T_end / T_start) ** (step / n_steps)
        
        # Force-directed step
        c_new = _force_directed_step(c, k=1.0, dt=0.001, r_target=r_target_init)
        r_new = _compute_radius(c_new)
        
        # Accept if better or with Metropolis probability
        if r_new >= r:
            c = c_new
            r = r_new
        else:
            delta = r - r_new
            if np.random.random() < np.exp(-delta / max(T, 1e-10)):
                c = c_new
                r = r_new
        
        # Occasionally add random perturbation to escape local minima
        if step % 100 == 50:
            idx = np.random.randint(N)
            c[idx] += np.random.randn(3) * 0.02
            c = np.clip(c, -0.4999, 0.4999)
            r = _compute_radius(c)
        
        if r > best_r:
            best_r = r
            best_c = c.copy()
        
        # Update target radius periodically
        if step % 200 == 100:
            r_target_init = max(best_r, 0.05)
    
    return best_c, best_r


def _soft_min_objective_and_grad(x, beta, iu, active_pairs, active_walls):
    """
    Objective: soft-min approximation of the radius constraints.
    We minimize G(x) = (1/beta) * ln(sum exp(-beta * c_k(x)))
    As beta -> inf, G(x) -> -min_k c_k(x).
    So minimizing G(x) is equivalent to maximizing min_k c_k(x).
    
    Only optimize over active constraints for efficiency.
    """
    c = x.reshape(-1, 3)
    
    # Build constraint values
    constraint_vals = []
    constraint_grads_info = []
    
    # Wall constraints
    for (i, k) in active_walls:
        c_w = 0.5 - abs(c[i, k])
        constraint_vals.append(c_w)
        constraint_grads_info.append(('wall', i, k))
    
    # Pair constraints
    for (i, j) in active_pairs:
        diff_ij = c[i] - c[j]
        dist_ij = np.linalg.norm(diff_ij)
        if dist_ij > 1e-12:
            c_p = dist_ij / 2.0
            constraint_vals.append(c_p)
            constraint_grads_info.append(('pair', i, j, diff_ij, dist_ij))
    
    if len(constraint_vals) == 0:
        # No active constraints, just return a small value
        return 0.0, np.zeros_like(c).ravel()
    
    all_vals = np.array(constraint_vals)
    
    # Compute weights w_k = exp(-beta * c_k) / S
    neg_beta_c = -beta * all_vals
    max_neg_beta_c = np.max(neg_beta_c)
    exp_vals = np.exp(neg_beta_c - max_neg_beta_c)
    S = np.sum(exp_vals)
    weights = exp_vals / S
    
    # Objective value
    obj = (max_neg_beta_c + np.log(S)) / beta
    
    # Gradient
    grad = np.zeros_like(c)
    
    for idx, (w, info) in enumerate(zip(weights, constraint_grads_info)):
        if info[0] == 'wall':
            _, i, k = info
            # d(c_w)/d(c[i,k]) = -sign(c[i,k])
            # dG/dc[i,k] = -w * (-sign(c[i,k])) = w * sign(c[i,k])
            grad[i, k] += w * np.sign(c[i, k])
        else:  # pair
            _, i, j, diff_ij, dist_ij = info
            # d(c_p)/d(c_i) = (c_i - c_j) / dist_ij = diff_ij / dist_ij
            # dG/dc_i = -w * (diff_ij / dist_ij)
            # dG/dc_j = -w * (-diff_ij / dist_ij) = w * (diff_ij / dist_ij)
            unit = diff_ij / dist_ij
            grad[i] -= w * unit
            grad[j] += w * unit
    
    return obj, grad.ravel()


def _get_active_constraints(centers, r_target, n_neighbors=8):
    """
    Get the set of active constraints: pairs in near-contact and wall contacts.
    This mimics the Voronoi cell approach where only neighboring cells matter.
    Returns: list of (i, j) pairs and list of (i, k) wall contacts
    """
    c = centers
    # Find pairs within 2*r_target * 1.05 (slightly larger to capture near-contacts)
    tree = cKDTree(c)
    pairs = tree.query_pairs(r=2 * r_target * 1.05)
    
    # Wall contacts: spheres close to walls
    wall_contacts = []
    for i in range(N):
        for k in range(3):
            wall_dist = 0.5 - abs(c[i, k])
            if wall_dist < r_target * 1.05:
                wall_contacts.append((i, k))
    
    return list(pairs), wall_contacts


def _optimize_active_constraints(x0, deadline, beta_start=50.0, beta_end=5000.0, n_steps=6, max_iter_per_beta=100):
    """
    Run L-BFGS-B optimization with increasing beta, using active constraint set.
    """
    best_x = x0.copy()
    best_r = _compute_radius(x0.reshape(-1, 3))
    
    betas = np.geomspace(beta_start, beta_end, n_steps)
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        c_current = best_x.reshape(-1, 3)
        r_current = _compute_radius(c_current)
        
        # Get active constraints
        active_pairs, active_walls = _get_active_constraints(c_current, max(r_current, 0.05))
        
        # If no active constraints, use all pairs
        if len(active_pairs) == 0:
            iu = np.triu_indices(N, k=1)
            active_pairs = list(zip(iu[0], iu[1]))
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                lambda x: _soft_min_objective_and_grad(x, beta, None, active_pairs, active_walls)[0],
                best_x,
                jac=lambda x: _soft_min_objective_and_grad(x, beta, None, active_pairs, active_walls)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter_per_beta, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            
            new_x = np.clip(res.x, -0.5, 0.5)
            new_r = _compute_radius(new_x.reshape(-1, 3))
            
            if new_r > best_r:
                best_r = new_r
                best_x = new_x
        except Exception:
            pass
    
    return best_x, best_r


def _high_precision_polish(centers, deadline):
    """
    Fine-tune the best configuration using very high beta soft-min
    to squeeze out the last bits of radius.
    """
    c = centers.copy()
    r = _compute_radius(c)
    
    # Use very high beta for near-exact min optimization
    for beta in [5000, 10000, 20000, 50000]:
        if time.time() > deadline:
            break
        
        active_pairs, active_walls = _get_active_constraints(c, max(r, 0.05))
        
        if len(active_pairs) == 0:
            iu = np.triu_indices(N, k=1)
            active_pairs = list(zip(iu[0], iu[1]))
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                lambda x: _soft_min_objective_and_grad(x, beta, None, active_pairs, active_walls)[0],
                c.ravel(),
                jac=lambda x: _soft_min_objective_and_grad(x, beta, None, active_pairs, active_walls)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 10}
            )
            
            new_c = np.clip(res.x.reshape(N, 3), -0.5, 0.5)
            new_r = _compute_radius(new_c)
            
            if new_r > r:
                r = new_r
                c = new_c
        except Exception:
            pass
    
    return c, r


def _init_grid():
    """5x5x4 grid initialization."""
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.375, 0.375, 4)
    return np.array([(x[i], y[j], z[k]) for i in range(5) for j in range(5) for k in range(4)])


def _init_halton():
    """Halton sequence initialization."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    centers = np.zeros((N, 3))
    for i in range(N):
        for j in range(3):
            centers[i, j] = 2 * halton(i + 1, [2, 3, 5][j]) - 1
    return centers


def _init_fcc_like():
    """FCC-like initialization."""
    centers = _init_grid()
    np.random.seed(42)
    centers += np.random.normal(0, 0.015, centers.shape)
    return centers


def _init_bcc_like():
    """BCC-like initialization: 5x5x4 grid with offset inner points."""
    centers = _init_grid()
    # Add small offsets to create BCC-like structure
    for i in range(N):
        if i % 2 == 0:
            centers[i] += 0.03
        else:
            centers[i] -= 0.03
    return np.clip(centers, -0.45, 0.45)


def _init_random(seed=0):
    np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, (N, 3))


def _topology_perturbation(centers, r_current, n_swaps=5, magnitude=0.02):
    """
    Perturb the contact graph by swapping positions of nearby spheres.
    This explores different topologies of the packing.
    """
    c = centers.copy()
    np.random.seed(int(time.time() * 1000) % 10000)
    
    tree = cKDTree(c)
    pairs = tree.query_pairs(r=2 * r_current * 1.1)
    
    for _ in range(n_swaps):
        if len(pairs) == 0:
            break
        i, j = np.random.choice(len(pairs))
        i, j = pairs[i]
        
        # Swap with small random perturbation
        swap = np.random.choice([0, 1, 2])  # which coordinate to perturb
        c[i, swap] += np.random.normal(0, magnitude)
        c[j, swap] -= np.random.normal(0, magnitude)
        c = np.clip(c, -0.5, 0.5)
    
    return c


def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin for safety
    
    # Multiple initializations
    inits = [
        _init_grid(),
        _init_fcc_like(),
        _init_bcc_like(),
        _init_halton(),
        _init_random(0),
        _init_random(1),
    ]
    
    best_r = -1
    best_centers = None
    
    for i, init_centers in enumerate(inits):
        if time.time() > deadline - 15:
            break
        
        remaining_inits = len(inits) - i
        time_per_init = min(20, (deadline - 15 - time.time()) / max(1, remaining_inits))
        
        # Phase 1: Simulated annealing to find good basin
        sa_deadline = time.time() + time_per_init * 0.5
        centers_sa, r_sa = _simulated_annealing(init_centers, sa_deadline)
        
        # Phase 2: L-BFGS-B soft-min optimization from SA result
        opt_deadline = time.time() + time_per_init * 0.35
        centers_opt, r_opt = _optimize_active_constraints(centers_sa.ravel(), opt_deadline)
        centers_opt = centers_opt.reshape(N, 3)
        
        # Phase 3: High-precision polish
        polish_deadline = min(time.time() + time_per_init * 0.15, deadline - 10)
        centers_polished, r_polished = _high_precision_polish(centers_opt, polish_deadline)
        
        # Track best
        if r_polished > best_r:
            best_r = r_polished
            best_centers = centers_polished.copy()
        
        # Try topology perturbation if we have time
        if time.time() < deadline - 12 and r_polished > 0.1:
            perturbed = _topology_perturbation(centers_polished, r_polished, n_swaps=3, magnitude=0.015)
            # Quick SA + optimize from perturbed
            sa2_deadline = time.time() + min(5, (deadline - 12 - time.time()) * 0.4)
            centers_sa2, r_sa2 = _simulated_annealing(perturbed, sa2_deadline)
            opt2_deadline = time.time() + min(5, (deadline - 12 - time.time()) * 0.4)
            centers_opt2, r_opt2 = _optimize_active_constraints(centers_sa2.ravel(), opt2_deadline)
            centers_opt2 = centers_opt2.reshape(N, 3)
            
            if r_opt2 > best_r:
                best_r = r_opt2
                best_centers = centers_opt2.copy()
    
    # Final polish with remaining time
    if best_centers is not None and time.time() < deadline - 3:
        final_deadline = deadline - 1
        best_centers, best_r = _high_precision_polish(best_centers, final_deadline)
    
    if best_centers is None:
        best_centers = _init_grid()
    
    return best_centers
