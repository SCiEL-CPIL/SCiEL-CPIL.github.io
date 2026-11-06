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


def _construct_cubic_grid():
    """Simple cubic grid as baseline."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _construct_fcc_like():
    """FCC-like arrangement: 4 layers of 13 spheres each (approximate)."""
    # FCC unit cell has spheres at corners and face centers
    # For 50 spheres, try 4 layers with offsets
    centers = []
    # Layer spacing and in-plane spacing for FCC
    s = 0.145  # approximate spacing
    # 4 layers, each with ~13 spheres in a hexagonal-like pattern
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
    # Pad or trim to N
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

    T = 0.005  # Initial temperature (smaller for fine-tuning)
    cooling = 0.998

    while time.time() - start_time < time_limit:
        if T < 1e-7:
            break

        # Random perturbation: move 1-3 random spheres
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
            # Revert
            centers = best_centers.copy() if current_r < best_r else centers

        T *= cooling

    return best_centers, best_r, time.time() - start_time


def construct():
    """Construct the best packing found within the time budget."""
    global_start = time.time()

    best_centers = None
    best_r = 0.0

    # Start with cubic grid
    centers0 = _construct_cubic_grid()
    r0 = _radius(centers0)
    best_centers = centers0.copy()
    best_r = r0

    # Phase 1: Smooth optimization from grid
    if time.time() - global_start < DEADLINE - 15.0:
        time_remaining = DEADLINE - (time.time() - global_start) - 10.0
        opt_centers, opt_r, elapsed = _optimize_smooth(
            centers0,
            time_limit=min(25.0, time_remaining)
        )
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()

    # Phase 2: Exact optimization from best
    if time.time() - global_start < DEADLINE - 15.0:
        time_remaining = DEADLINE - (time.time() - global_start) - 10.0
        opt_centers, opt_r, elapsed = _optimize_exact(
            best_centers,
            time_limit=min(20.0, time_remaining)
        )
        if opt_r > best_r:
            best_r = opt_r
            best_centers = opt_centers.copy()

    # Phase 3: Simulated annealing to escape local minima
    if time.time() - global_start < DEADLINE - 20.0:
        time_remaining = DEADLINE - (time.time() - global_start) - 15.0
        sa_centers, sa_r, elapsed = _simulated_annealing(
            best_centers,
            time_limit=min(30.0, time_remaining),
            seed=12345
        )
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers.copy()

            # Polish with exact optimization after SA
            if time.time() - global_start < DEADLINE - 10.0:
                time_remaining2 = DEADLINE - (time.time() - global_start) - 5.0
                opt_centers2, opt_r2, elapsed2 = _optimize_exact(
                    best_centers,
                    time_limit=min(15.0, time_remaining2)
                )
                if opt_r2 > best_r:
                    best_r = opt_r2
                    best_centers = opt_centers2.copy()

    # Phase 4: Try FCC-like initialization
    if time.time() - global_start < DEADLINE - 25.0:
        centers_fcc = _construct_fcc_like()
        r_fcc = _radius(centers_fcc)
        if r_fcc > best_r:
            best_r = r_fcc
            best_centers = centers_fcc.copy()

        time_remaining = DEADLINE - (time.time() - global_start) - 15.0
        if time_remaining > 10.0:
            opt_centers, opt_r, elapsed = _optimize_smooth(
                centers_fcc,
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

    # Phase 5: Random restarts with SA + optimization
    n_random_tries = 3
    for i in range(n_random_tries):
        if time.time() - global_start > DEADLINE - 20.0:
            break

        centers_rand = _random_initialization(seed=100 + i)
        r_rand = _radius(centers_rand)

        # Quick smooth optimization
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

            # SA polish
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

                    # Final exact polish
                    if time.time() - global_start < DEADLINE - 5.0:
                        time_remaining3 = DEADLINE - (time.time() - global_start) - 2.0
                        opt_centers3, opt_r3, elapsed3 = _optimize_exact(
                            best_centers,
                            time_limit=min(5.0, time_remaining3)
                        )
                        if opt_r3 > best_r:
                            best_r = opt_r3
                            best_centers = opt_centers3.copy()

    # Phase 6: Final polish with exact optimization
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
        # Fallback to cubic grid
        final_centers = _construct_cubic_grid()

    return final_centers
# EVOLVE-BLOCK-END
