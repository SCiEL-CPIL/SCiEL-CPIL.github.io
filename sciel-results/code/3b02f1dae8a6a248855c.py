"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 150.0 - 10.0  # safety margin before the 150s kill


def _radius(centers):
    """Vectorized max valid radius. Returns 0.0 if invalid."""
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    # boundary distance
    b = 0.5 - np.abs(centers)
    min_b = float(np.min(b))
    if min_b <= 0.0:
        return 0.0
    # pairwise distances via broadcasting
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = float(np.min(dists))
    if min_pair <= 0.0:
        return 0.0
    return min(min_pair / 2.0, min_b)


def _soft_min_radius(centers, beta=100.0):
    """
    Smooth surrogate for the max valid radius using soft-min via log-sum-exp.
    """
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0

    # boundary distances
    b = 0.5 - np.abs(centers)

    # pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)

    # half distances for pairs
    iu = np.triu_indices(N, k=1)
    half_dists = dists[iu] / 2.0

    # Combine all constraints
    all_vals = np.concatenate([half_dists, b])

    # Soft min via log-sum-exp
    min_val = np.min(all_vals)
    if min_val <= 0:
        return 0.0

    scaled = -beta * (all_vals - min_val)
    scaled = np.clip(scaled, -100, 100)

    log_sum = np.log(np.sum(np.exp(scaled)))
    soft_min = min_val - log_sum / beta

    return max(0.0, soft_min)


def _grad_soft_min_radius(centers, beta=100.0):
    """
    Gradient of the soft-min radius w.r.t. centers.
    """
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return np.zeros_like(centers)
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return np.zeros_like(centers)

    b = 0.5 - np.abs(centers)

    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)

    iu = np.triu_indices(N, k=1)
    half_dists = dists[iu] / 2.0

    all_vals = np.concatenate([half_dists, b])
    min_val = np.min(all_vals)
    if min_val <= 0:
        return np.zeros_like(centers)

    scaled = -beta * (all_vals - min_val)
    scaled = np.clip(scaled, -100, 100)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)

    # Weights for soft-min
    weights = exp_scaled / sum_exp

    grad = np.zeros_like(centers)

    # Gradient contribution from pairwise terms
    n_pairs = len(iu)
    for idx in range(n_pairs):
        i, j = iu[0][idx], iu[1][idx]
        d = dists[i, j]
        if d < 1e-10:
            continue
        w = weights[idx]
        vec = (centers[i] - centers[j]) / d
        grad[i] += w * 0.5 * vec
        grad[j] += w * 0.5 * (-vec)

    # Gradient contribution from boundary terms
    n_boundary = N * 3
    for idx in range(n_boundary):
        i = idx // 3
        k = idx % 3
        w = weights[n_pairs + idx]
        sign = np.sign(centers[i, k])
        if sign == 0:
            sign = 1.0
        grad[i, k] += w * (-sign)

    return grad


def _repulsive_energy(centers, r_target=0.14):
    """
    Repulsive potential energy: sum of 1/d^2 for pairs closer than 2*r_target,
    plus wall repulsion.
    """
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return 1e10
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 1e10

    # Pairwise repulsion
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)

    iu = np.triu_indices(N, k=1)
    d_pairs = dists[iu]
    # Only count pairs that are too close
    mask = d_pairs < 2 * r_target
    if np.any(mask):
        e_pairs = np.sum(1.0 / (d_pairs[mask] ** 2 + 1e-10))
    else:
        e_pairs = 0.0

    # Wall repulsion
    b = 0.5 - np.abs(centers)
    mask_wall = b < r_target
    if np.any(mask_wall):
        e_wall = np.sum(1.0 / (b[mask_wall] ** 2 + 1e-10))
    else:
        e_wall = 0.0

    return e_pairs + e_wall


def _grad_repulsive_energy(centers, r_target=0.14):
    """
    Gradient of repulsive potential energy.
    """
    if centers.shape != (N, 3) or not np.all(np.isfinite(centers)):
        return np.zeros_like(centers)
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return np.zeros_like(centers)

    grad = np.zeros_like(centers)

    # Pairwise gradient
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)

    iu = np.triu_indices(N, k=1)
    d_pairs = dists[iu]
    mask = d_pairs < 2 * r_target

    for idx in range(len(iu)):
        if not mask[idx]:
            continue
        i, j = iu[0][idx], iu[1][idx]
        d = dists[i, j]
        if d < 1e-10:
            continue
        # d(1/d^2)/d(centers[i]) = -2 * (centers[i]-centers[j]) / d^4
        vec = (centers[i] - centers[j])
        factor = -2.0 / (d ** 4 + 1e-10)
        grad[i] += factor * vec
        grad[j] += -factor * vec

    # Wall gradient
    b = 0.5 - np.abs(centers)
    mask_wall = b < r_target
    for idx in range(N * 3):
        if not mask_wall[idx]:
            continue
        i = idx // 3
        k = idx % 3
        d = b[idx]
        if d < 1e-10:
            continue
        sign = np.sign(centers[i, k])
        if sign == 0:
            sign = 1.0
        # d(1/b^2)/d(centers[i,k]) = -2/b^3 * sign
        grad[i, k] += -2.0 * sign / (d ** 3 + 1e-10)

    return grad


def _construct_cubic_grid():
    """Simple cubic grid as baseline."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _construct_bcc_lattice():
    """
    BCC (Body-Centered Cubic) lattice perturbation initialization.
    BCC has higher packing density than simple cubic.
    """
    # BCC unit cell: 8 corner atoms + 1 body center = 9 atoms per cell
    # For 50 spheres, we need about 50/9 ≈ 5.56 cells
    # Use a 3x3x3 supercell of BCC = 3^3 * 9 = 243 positions, pick 50
    
    # Better approach: create a distorted BCC-like arrangement
    # 50 spheres in a cube: try 5x5x2 or similar arrangement
    
    # Let's create a grid and add body-center perturbations
    m = 5  # 5x5x2 = 50
    centers = []
    
    # Create positions on a BCC-like lattice
    # Layer approach: 2 layers of 5x5 = 50
    for layer in range(2):
        z = -0.5 + (layer + 0.5) / 2.0
        # In each layer, use a square grid with BCC offset
        for i in range(5):
            for j in range(5):
                # Add slight BCC-like offset
                offset = 0.05 * (1 if (i + j) % 2 == 0 else -1)
                x = -0.5 + (i + 0.5) / 5.0 + offset * 0.1
                y = -0.5 + (j + 0.5) / 5.0 - offset * 0.1
                centers.append([x, y, z])
    
    return np.array(centers[:N])


def _construct_fcc_like():
    """FCC-like arrangement."""
    centers = []
    s = 0.145
    for layer in range(4):
        z = -0.5 + 0.5 * (layer + 0.5) / 4
        offset = (layer % 2) * s * 0.5
        for i in range(4):
            for j in range(4):
                if len(centers) < N:
                    x = -0.5 + s * (i + 0.5) + offset
                    y = -0.5 + s * (j + 0.5)
                    if -0.45 <= x <= 0.45 and -0.45 <= y <= 0.45:
                        centers.append([x, y, z])
    while len(centers) < N:
        centers.append([0.0, 0.0, 0.0])
    return np.array(centers[:N])


def _random_initialization(seed=None):
    """Random initialization inside the cube with some margin."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    margin = 0.1
    centers = rng.uniform(-0.5 + margin, 0.5 - margin, size=(N, 3))
    return centers


def _optimize_smooth(centers_start, time_limit=30.0, beta=50.0):
    """
    Optimize using the smooth soft-min surrogate.
    """
    start_time = time.time()
    params = centers_start.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)

    def obj(p):
        c = p.reshape(N, 3)
        return -_soft_min_radius(c, beta)

    def grad(p):
        c = p.reshape(N, 3)
        return -_grad_soft_min_radius(c, beta).flatten()

    best_params = params.copy()
    best_r = _radius(centers_start)

    try:
        result = minimize(
            obj,
            params,
            jac=grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        if result.fun < -best_r:
            new_centers = result.x.reshape(N, 3)
            new_r = _radius(new_centers)
            if new_r > best_r:
                best_r = new_r
                best_params = result.x.copy()
    except Exception:
        pass

    elapsed = time.time() - start_time
    return best_params.reshape(N, 3), best_r, elapsed


def _optimize_repulsive(centers_start, time_limit=20.0, r_target=0.14):
    """
    Optimize using repulsive potential energy minimization.
    """
    start_time = time.time()
    params = centers_start.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)

    def obj(p):
        c = p.reshape(N, 3)
        return _repulsive_energy(c, r_target)

    def grad(p):
        c = p.reshape(N, 3)
        return _grad_repulsive_energy(c, r_target).flatten()

    best_params = params.copy()
    best_r = _radius(centers_start)

    try:
        result = minimize(
            obj,
            params,
            jac=grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
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


def _optimize_exact(centers_start, time_limit=30.0):
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
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        if result.fun < -best_r:
            new_centers = result.x.reshape(N, 3)
            new_r = _radius(new_centers)
            if new_r > best_r:
                best_r = new_r
                best_params = result.x.copy()
    except Exception:
        pass

    elapsed = time.time() - start_time
    return best_params.reshape(N, 3), best_r, elapsed


def _simulated_annealing(centers_start, time_limit=20.0, seed=None):
    """
    Simulated annealing to escape local minima.
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

    T = 0.005
    cooling = 0.998

    while time.time() - start_time < time_limit:
        if T < 1e-7:
            break

        n_move = rng.integers(1, 4)
        idxs = rng.choice(N, size=n_move, replace=False)
        delta_scale = T * 0.5
        for idx in idxs:
            delta = rng.normal(0, delta_scale, size=3)
            centers[idx] += delta
            centers[idx] = np.clip(centers[idx], -0.5, 0.5)

        new_r = _radius(centers)
        delta_r = new_r - current_r

        if delta_r > 0 or rng.random() < np.exp(delta_r / T):
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
        else:
            centers = best_centers.copy() if current_r < best_r else centers

        T *= cooling

    return best_centers, best_r, time.time() - start_time


def construct():
    """Construct the best packing found within the time budget."""
    global_start = time.time()

    best_centers = None
    best_r = 0.0

    # Phase 1: BCC lattice initialization + optimization
    centers_bcc = _construct_bcc_lattice()
    r_bcc = _radius(centers_bcc)
    best_centers = centers_bcc.copy()
    best_r = r_bcc

    if time.time() - global_start < DEADLINE - 30.0:
        time_remaining = DEADLINE - (time.time() - global_start) - 20.0
        opt_centers, opt_r, elapsed = _optimize_smooth(
            centers_bcc,
            time_limit=min(25.0, time_remaining)
        )
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()

    # Phase 2: Repulsive potential optimization
    if time.time() - global_start < DEADLINE - 25.0:
        time_remaining = DEADLINE - (time.time() - global_start) - 15.0
        opt_centers, opt_r, elapsed = _optimize_repulsive(
            best_centers,
            time_limit=min(20.0, time_remaining),
            r_target=0.14
        )
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()

    # Phase 3: Exact optimization
    if time.time() - global_start < DEADLINE - 20.0:
        time_remaining = DEADLINE - (time.time() - global_start) - 10.0
        opt_centers, opt_r, elapsed = _optimize_exact(
            best_centers,
            time_limit=min(20.0, time_remaining)
        )
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()

    # Phase 4: Simulated annealing
    if time.time() - global_start < DEADLINE - 15.0:
        time_remaining = DEADLINE - (time.time() - global_start) - 10.0
        sa_centers, sa_r, elapsed = _simulated_annealing(
            best_centers,
            time_limit=min(25.0, time_remaining),
            seed=12345
        )
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers.copy()

            # Polish after SA
            if time.time() - global_start < DEADLINE - 10.0:
                time_remaining2 = DEADLINE - (time.time() - global_start) - 5.0
                opt_centers2, opt_r2, elapsed2 = _optimize_exact(
                    best_centers,
                    time_limit=min(15.0, time_remaining2)
                )
                if opt_r2 > best_r:
                    best_r = opt_r2
                    best_centers = opt_centers2.copy()

    # Phase 5: Try cubic grid as alternative
    if time.time() - global_start < DEADLINE - 25.0:
        centers0 = _construct_cubic_grid()
        r0 = _radius(centers0)
        if r0 > best_r:
            best_r = r0
            best_centers = centers0.copy()

        time_remaining = DEADLINE - (time.time() - global_start) - 15.0
        if time_remaining > 10.0:
            opt_centers, opt_r, elapsed = _optimize_smooth(
                centers0,
                time_limit=min(20.0, time_remaining)
            )
            if opt_r > best_r:
                best_r = opt_r
                best_centers = opt_centers.copy()

                if time.time() - global_start < DEADLINE - 10.0:
                    time_remaining2 = DEADLINE - (time.time() - global_start) - 5.0
                    opt_centers2, opt_r2, elapsed2 = _optimize_exact(
                        best_centers,
                        time_limit=min(10.0, time_remaining2)
                    )
                    if opt_r2 > best_r:
                        best_r = opt_r2
                        best_centers = opt_centers2.copy()

    # Phase 6: Random restarts
    n_random_tries = 3
    for i in range(n_random_tries):
        if time.time() - global_start > DEADLINE - 20.0:
            break

        centers_rand = _random_initialization(seed=100 + i)
        r_rand = _radius(centers_rand)

        time_remaining = DEADLINE - (time.time() - global_start)
        if time_remaining < 15.0:
            break

        opt_centers, opt_r, elapsed = _optimize_smooth(
            centers_rand,
            time_limit=min(10.0, time_remaining - 10.0)
        )

        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()

            if time.time() - global_start < DEADLINE - 10.0:
                time_remaining2 = DEADLINE - (time.time() - global_start) - 5.0
                sa_centers, sa_r, elapsed2 = _simulated_annealing(
                    best_centers,
                    time_limit=min(10.0, time_remaining2),
                    seed=200 + i
                )
                if sa_r > best_r:
                    best_r = sa_r
                    best_centers = sa_centers.copy()

                    if time.time() - global_start < DEADLINE - 5.0:
                        time_remaining3 = DEADLINE - (time.time() - global_start) - 2.0
                        opt_centers3, opt_r3, elapsed3 = _optimize_exact(
                            best_centers,
                            time_limit=min(5.0, time_remaining3)
                        )
                        if opt_r3 > best_r:
                            best_r = opt_r3
                            best_centers = opt_centers3.copy()

    # Phase 7: Final polish
    if time.time() - global_start < DEADLINE - 5.0:
        time_remaining = DEADLINE - (time.time() - global_start)
        opt_centers, opt_r, elapsed = _optimize_exact(
            best_centers,
            time_limit=min(10.0, time_remaining - 2.0)
        )
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()

    # Ensure validity
    final_centers = np.clip(best_centers, -0.5, 0.5)
    final_r = _radius(final_centers)

    if final_r <= 0.0:
        final_centers = _construct_cubic_grid()

    return final_centers
# EVOLVE-BLOCK-END
