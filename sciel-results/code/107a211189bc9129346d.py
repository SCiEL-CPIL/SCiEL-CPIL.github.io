"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

DEADLINE = 150.0 - 10.0  # safety margin before the 150s kill


def _radius(centers):
    """Vectorized max valid radius. Returns 0.0 if invalid."""
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    b = 0.5 - np.abs(centers)
    min_b = float(np.min(b))
    if min_b <= 0.0:
        return 0.0
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = float(np.min(dists))
    if min_pair <= 0.0:
        return 0.0
    return min(min_pair / 2.0, min_b)


def _soft_min_radius_and_grad(centers, beta=80.0):
    """
    Compute the soft-min radius and its gradient using log-sum-exp.
    Returns (soft_min_value, gradient).
    """
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 0.0, np.zeros_like(centers)
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0, np.zeros_like(centers)

    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2) + 1e-30)
    np.fill_diagonal(dists, np.inf)

    triu_idx = np.triu_indices(N, k=1)
    pair_dists = dists[triu_idx]
    pair_terms = pair_dists / 2.0

    # Boundary terms
    boundary_terms = 0.5 - np.abs(centers)
    boundary_flat = boundary_terms.flatten()

    all_terms = np.concatenate([pair_terms, boundary_flat])

    # Soft min via log-sum-exp
    min_val = np.min(all_terms)
    if min_val <= 0:
        return 0.0, np.zeros_like(centers)

    shifted = -beta * (all_terms - min_val)
    shifted = np.clip(shifted, -700, 700)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)

    if sum_exp == 0 or not np.isfinite(sum_exp):
        weights = np.zeros_like(all_terms)
        min_indices = np.where(all_terms == min_val)[0]
        weights[min_indices] = 1.0 / len(min_indices)
    else:
        weights = exp_vals / sum_exp

    soft_min_val = min_val - (1.0 / beta) * np.log(sum_exp)
    soft_min_val = max(0.0, soft_min_val)

    # Gradient computation
    grad = np.zeros_like(centers)

    # Pairwise gradient contributions
    n_pairs = len(triu_idx)
    pair_weights = weights[:n_pairs]
    i_idx = triu_idx[0]
    j_idx = triu_idx[1]

    d_vecs = centers[i_idx] - centers[j_idx]
    d_norms = np.linalg.norm(d_vecs, axis=1)
    d_norms_safe = np.where(d_norms < 1e-15, 1e-15, d_norms)

    # d(pair_term)/d(c_i) = (c_i - c_j) / (2 * d_ij)
    # d(pair_term)/d(c_j) = (c_j - c_i) / (2 * d_ij) = -(c_i - c_j) / (2 * d_ij)
    grad_pair_i = pair_weights[:, np.newaxis] * (d_vecs / (2.0 * d_norms_safe[:, np.newaxis]))
    grad_pair_j = -grad_pair_i

    for idx in range(n_pairs):
        grad[i_idx[idx]] += grad_pair_i[idx]
        grad[j_idx[idx]] += grad_pair_j[idx]

    # Boundary gradient contributions
    # d(0.5 - |c_ik|)/d(c_ik) = -sign(c_ik)
    n_boundary = N * 3
    boundary_weights = weights[n_pairs:]
    for idx in range(n_boundary):
        i = idx // 3
        k = idx % 3
        w = boundary_weights[idx]
        sign = np.sign(centers[i, k])
        if sign == 0:
            sign = 1.0
        grad[i, k] -= w * sign

    return soft_min_val, grad


def _repulsive_relaxation(centers, time_limit=10.0, n_iters=200):
    """
    Iterative repulsive relaxation: move each sphere away from its nearest neighbors.
    This helps escape poor local minima by spreading spheres apart.
    """
    start_time = time.time()
    centers = centers.copy()
    current_r = _radius(centers)
    best_centers = centers.copy()
    best_r = current_r

    learning_rate = 0.01

    for it in range(n_iters):
        if time.time() - start_time > time_limit:
            break

        # Compute forces: repulsion from all other spheres + attraction to center
        forces = np.zeros_like(centers)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    # Random small push for coincident points
                    forces[i] += np.random.randn(3) * 0.01
                    continue
                # Repulsive force: stronger when closer
                force_mag = 1.0 / (dist * dist + 1e-6)
                forces[i] += force_mag * d / dist

        # Add wall repulsion: push spheres away from walls
        for i in range(N):
            for k in range(3):
                if centers[i, k] > 0:
                    forces[i, k] -= 0.5 * centers[i, k]
                else:
                    forces[i, k] -= 0.5 * centers[i, k]

        # Update positions
        centers += learning_rate * forces
        centers = np.clip(centers, -0.4999, 0.4999)

        new_r = _radius(centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()

        # Decay learning rate
        learning_rate *= 0.995

    return best_centers, best_r


def _construct_fcc_lattice():
    """
    Construct an FCC-like lattice for 50 spheres.
    FCC has spheres at corners and face centers of a cube.
    For 50 spheres, use multiple layers with proper offsets.
    """
    centers = []
    # Use a 4x4x4 base grid with FCC offsets
    # FCC positions in a unit cell: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    # Scale to fit in [-0.5, 0.5]^3
    
    # Try 4 layers of 13 spheres each (approximate)
    s = 0.13  # spacing parameter
    
    for layer in range(4):
        z = -0.45 + 0.9 * layer / 3.0
        offset_x = (layer % 2) * s * 0.5
        offset_y = (layer % 2) * s * 0.354  # hexagonal offset
        
        for i in range(4):
            for j in range(4):
                if len(centers) < N:
                    x = -0.45 + s * (i + 0.5) + offset_x
                    y = -0.45 + s * (j + 0.5) + offset_y
                    if -0.48 <= x <= 0.48 and -0.48 <= y <= 0.48:
                        centers.append([x, y, z])
    
    # Pad if needed
    while len(centers) < N:
        centers.append([0.0, 0.0, 0.0])
    
    return np.array(centers[:N])


def _construct_bcc_lattice():
    """
    Construct a BCC-like lattice for 50 spheres.
    BCC has spheres at corners and body center.
    """
    centers = []
    s = 0.135
    
    for layer in range(4):
        z = -0.45 + 0.9 * layer / 3.0
        for i in range(4):
            for j in range(4):
                if len(centers) < N:
                    x = -0.45 + s * (i + 0.5)
                    y = -0.45 + s * (j + 0.5)
                    if -0.48 <= x <= 0.48 and -0.48 <= y <= 0.48:
                        centers.append([x, y, z])
        
        # Add body-center positions for this layer
        if len(centers) < N:
            xc = -0.45 + s * 1.5
            yc = -0.45 + s * 1.5
            if -0.48 <= xc <= 0.48 and -0.48 <= yc <= 0.48:
                centers.append([xc, yc, z])
    
    while len(centers) < N:
        centers.append([0.0, 0.0, 0.0])
    
    return np.array(centers[:N])


def _construct_grid():
    """Simple cubic grid as baseline."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _optimize_smooth(centers_start, time_limit=30.0, beta=80.0):
    """
    Optimize using the smooth soft-min surrogate with gradient.
    """
    start_time = time.time()
    params = centers_start.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)

    def obj(p):
        c = p.reshape(N, 3)
        val, _ = _soft_min_radius_and_grad(c, beta)
        return -val

    def grad(p):
        c = p.reshape(N, 3)
        _, g = _soft_min_radius_and_grad(c, beta)
        return -g.flatten()

    best_params = params.copy()
    best_r = _radius(centers_start)

    try:
        result = minimize(
            obj,
            params,
            jac=grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        new_centers = result.x.reshape(N, 3)
        new_r = _radius(new_centers)
        if new_r > best_r:
            best_r = new_r
            best_params = result.x.copy()
    except Exception:
        pass

    elapsed = time.time() - start_time
    return best_params.reshape(N, 3), best_r, elapsed


def _optimize_exact(centers_start, time_limit=20.0):
    """
    Optimize using the exact (nonsmooth) radius function.
    """
    start_time = time.time()
    params = centers_start.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)

    def obj(p):
        c = p.reshape(N, 3)
        return -_radius(c)

    best_params = params.copy()
    best_r = _radius(centers_start)

    try:
        result = minimize(
            obj,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        new_centers = result.x.reshape(N, 3)
        new_r = _radius(new_centers)
        if new_r > best_r:
            best_r = new_r
            best_params = result.x.copy()
    except Exception:
        pass

    elapsed = time.time() - start_time
    return best_params.reshape(N, 3), best_r, elapsed


def _local_search(centers_start, time_limit=15.0, seed=None):
    """
    Local search with random perturbations and acceptance based on improvement.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    start_time = time.time()
    centers = centers_start.copy()
    current_r = _radius(centers)
    best_centers = centers.copy()
    best_r = current_r

    step_size = 0.02

    while time.time() - start_time < time_limit:
        # Move 1-3 random spheres
        n_move = rng.integers(1, 4)
        idxs = rng.choice(N, size=n_move, replace=False)
        for idx in idxs:
            delta = rng.normal(0, step_size, size=3)
            centers[idx] += delta
            centers[idx] = np.clip(centers[idx], -0.4999, 0.4999)

        new_r = _radius(centers)
        if new_r > current_r:
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
            # Keep the move
        else:
            # Revert
            for idx in idxs:
                delta = rng.normal(0, step_size, size=3)
                # We need to revert, so store original positions
            # Simpler: just revert by recomputing from best
            centers = best_centers.copy() if current_r < best_r else centers

        # Decay step size
        step_size *= 0.999

    return best_centers, best_r


def construct():
    """Construct the best packing found within the time budget."""
    global_start = time.time()

    best_centers = None
    best_r = 0.0

    # Phase 1: Try multiple initializations with repulsive relaxation
    initializations = [
        ("grid", _construct_grid()),
        ("fcc", _construct_fcc_lattice()),
        ("bcc", _construct_bcc_lattice()),
    ]

    for name, init_centers in initializations:
        if time.time() - global_start > DEADLINE - 30.0:
            break

        # Repulsive relaxation
        time_remaining = DEADLINE - (time.time() - global_start) - 25.0
        if time_remaining < 5.0:
            break

        relaxed_centers, relaxed_r = _repulsive_relaxation(
            init_centers,
            time_limit=min(8.0, time_remaining * 0.3)
        )

        if relaxed_r > best_r:
            best_r = relaxed_r
            best_centers = relaxed_centers.copy()

    # Ensure we have a valid starting point
    if best_centers is None or best_r <= 0:
        best_centers = _construct_grid()
        best_r = _radius(best_centers)

    # Phase 2: Smooth optimization from best initialization
    if time.time() - global_start < DEADLINE - 20.0:
        time_remaining = DEADLINE - (time.time() - global_start) - 15.0
        opt_centers, opt_r, elapsed = _optimize_smooth(
            best_centers,
            time_limit=min(25.0, time_remaining)
        )
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()

    # Phase 3: Local search to escape local minima
    if time.time() - global_start < DEADLINE - 15.0:
        time_remaining = DEADLINE - (time.time() - global_start) - 10.0
        ls_centers, ls_r, elapsed = _local_search(
            best_centers,
            time_limit=min(15.0, time_remaining),
            seed=42
        )
        if ls_r > best_r:
            best_r = ls_r
            best_centers = ls_centers.copy()

            # Polish with smooth optimization after local search
            if time.time() - global_start < DEADLINE - 10.0:
                time_remaining2 = DEADLINE - (time.time() - global_start) - 5.0
                opt_centers2, opt_r2, elapsed2 = _optimize_smooth(
                    best_centers,
                    time_limit=min(15.0, time_remaining2)
                )
                if opt_r2 > best_r:
                    best_r = opt_r2
                    best_centers = opt_centers2.copy()

    # Phase 4: Exact optimization polish
    if time.time() - global_start < DEADLINE - 8.0:
        time_remaining = DEADLINE - (time.time() - global_start) - 5.0
        opt_centers, opt_r, elapsed = _optimize_exact(
            best_centers,
            time_limit=min(10.0, time_remaining)
        )
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()

    # Phase 5: Try another initialization if time permits
    if time.time() - global_start < DEADLINE - 25.0:
        # Random initialization
        rng = np.random.default_rng(123)
        margin = 0.08
        centers_rand = rng.uniform(-0.5 + margin, 0.5 - margin, size=(N, 3))
        
        time_remaining = DEADLINE - (time.time() - global_start) - 20.0
        if time_remaining > 10.0:
            relaxed_centers, relaxed_r = _repulsive_relaxation(
                centers_rand,
                time_limit=min(8.0, time_remaining * 0.3)
            )
            
            if relaxed_r > best_r:
                best_r = relaxed_r
                best_centers = relaxed_centers.copy()
            
            # Smooth optimization
            time_remaining2 = DEADLINE - (time.time() - global_start) - 10.0
            if time_remaining2 > 5.0:
                opt_centers, opt_r, elapsed = _optimize_smooth(
                    best_centers,
                    time_limit=min(15.0, time_remaining2)
                )
                if opt_r > best_r:
                    best_r = opt_r
                    best_centers = opt_centers.copy()
                
                # Exact polish
                if time.time() - global_start < DEADLINE - 5.0:
                    time_remaining3 = DEADLINE - (time.time() - global_start) - 3.0
                    opt_centers2, opt_r2, elapsed2 = _optimize_exact(
                        best_centers,
                        time_limit=min(5.0, time_remaining3)
                    )
                    if opt_r2 > best_r:
                        best_r = opt_r2
                        best_centers = opt_centers2.copy()

    # Final polish
    if time.time() - global_start < DEADLINE - 3.0:
        time_remaining = DEADLINE - (time.time() - global_start) - 2.0
        opt_centers, opt_r, elapsed = _optimize_exact(
            best_centers,
            time_limit=min(5.0, time_remaining)
        )
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()

    # Ensure validity
    final_centers = np.clip(best_centers, -0.5, 0.5)
    final_r = _radius(final_centers)

    if final_r <= 0.0:
        final_centers = _construct_grid()

    return final_centers
# EVOLVE-BLOCK-END
