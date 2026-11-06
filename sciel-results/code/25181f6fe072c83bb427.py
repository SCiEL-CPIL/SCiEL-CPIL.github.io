"""Packing 50 equal spheres in a unit cube - hard-core repulsive gas model."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _exact_radius(C):
    """Compute the exact packing radius."""
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)
    if n < 2:
        return np.inf
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    abs_C = np.abs(C)
    min_bnd = np.min(0.5 - abs_C)
    return min(min_pair, min_bnd)


def _soft_min_obj_and_grad(C, beta):
    """
    Hard-core repulsive gas: negative log-sum-exp of all constraint terms.
    Terms: pairwise distances/2 and boundary distances (0.5 - |c_ik|).
    Minimizing -soft_min = maximizing the smooth surrogate of the min radius.
    Returns (objective, gradient w.r.t. C).
    """
    C = np.asarray(C).reshape(-1, 3)
    n = len(C)

    # Pairwise distances
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-30)
    np.fill_diagonal(dists, 1.0)
    pair_terms = dists / 2.0

    # Boundary terms
    abs_C = np.abs(C)
    boundary_terms = 0.5 - abs_C

    triu_i, triu_j = np.triu_indices(n, k=1)
    n_pairs = n * (n - 1) // 2
    pair_vals = pair_terms[triu_i, triu_j]  # (n_pairs,)
    bnd_vals = boundary_terms.flatten()      # (3n,)

    all_vals = np.concatenate([pair_vals, bnd_vals])

    # Soft min via log-sum-exp: soft_min = min_val - (1/beta)*log(sum(exp(-beta*(x - min_val))))
    min_val = np.min(all_vals)
    shifted = -beta * (all_vals - min_val)
    # Clip for numerical stability
    shifted = np.clip(shifted, -700.0, 700.0)
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    log_sum = np.log(sum_exp)
    soft_min_val = min_val - (1.0 / beta) * log_sum

    obj = -soft_min_val  # We minimize this

    # Gradient
    grad_C = np.zeros_like(C)

    # Pairwise part
    d = C[triu_i] - C[triu_j]  # (n_pairs, 3)
    dist = np.sqrt(np.sum(d ** 2, axis=1) + 1e-30)
    valid = dist > 1e-15
    coeff = np.zeros(n_pairs)
    coeff[valid] = weights[:n_pairs][valid] / (2.0 * dist[valid])
    weighted_d = coeff[:, np.newaxis] * d
    np.add.at(grad_C, triu_i, weighted_d)
    np.add.at(grad_C, triu_j, -weighted_d)

    # Boundary part
    bnd_grad = weights[n_pairs:].reshape(n, 3)
    signs = np.sign(C)
    signs[abs_C < 1e-15] = 1.0
    grad_C += -bnd_grad * signs

    return obj, grad_C


def _bcc_lattice_init():
    """BCC lattice: 4x4x4 corners + 4x4x4 body centers = 128, take 50."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    corners = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    body = corners + 0.5 / m
    # Scale to fit: BCC cell size = 1/m, but we want it to span the cube
    # Actually just use the lattice as-is within [-0.5, 0.5]
    all_pts = np.vstack([corners, body])
    rng = np.random.default_rng(101)
    idx = rng.choice(len(all_pts), size=N, replace=False)
    C = all_pts[idx].copy()
    C += rng.uniform(-0.01, 0.01, size=C.shape)
    C = np.clip(C, -0.48, 0.48)
    return C


def _fcc_lattice_init():
    """FCC-like lattice."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = []
    for z in ticks:
        for y in ticks:
            for x in ticks:
                pts.append([x, y, z])
                # Face centers
                if x == ticks[0]:
                    pts.append([x, y + 0.5/m, z + 0.5/m])
    pts = np.array(pts)
    rng = np.random.default_rng(202)
    if len(pts) >= N:
        idx = rng.choice(len(pts), size=N, replace=False)
        C = pts[idx].copy()
    else:
        C = pts.copy()
        while len(C) < N:
            C = np.vstack([C, rng.uniform(-0.48, 0.48, size=(1, 3))])
    C = C[:N]
    C += rng.uniform(-0.01, 0.01, size=C.shape)
    C = np.clip(C, -0.48, 0.48)
    return C


def _grid_init():
    """Jittered 4x4x4 grid."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    rng = np.random.default_rng(303)
    idx = rng.choice(len(grid), size=N, replace=False)
    C = grid[idx].copy()
    C += rng.uniform(-0.015, 0.015, size=C.shape)
    C = np.clip(C, -0.48, 0.48)
    return C


def _random_init(seed):
    rng = np.random.default_rng(seed)
    C = rng.uniform(-0.45, 0.45, size=(N, 3))
    return C


def _optimize_smooth(C0, beta, deadline, maxiter=2000):
    """Run L-BFGS-B on the soft-min objective."""
    if time.time() > deadline:
        return C0, _exact_radius(C0)

    def objective(x):
        C = x.reshape(-1, 3)
        obj, _ = _soft_min_obj_and_grad(C, beta)
        return obj

    def gradient(x):
        C = x.reshape(-1, 3)
        _, g = _soft_min_obj_and_grad(C, beta)
        return g.flatten()

    try:
        res = minimize(
            objective,
            C0.flatten(),
            jac=gradient,
            method='L-BFGS-B',
            bounds=[(-0.4999, 0.4999)] * (N * 3),
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        C_new = res.x.reshape(-1, 3)
        C_new = np.clip(C_new, -0.4999, 0.4999)
        return C_new, _exact_radius(C_new)
    except Exception:
        return C0, _exact_radius(C0)


def _simulated_annealing(C0, deadline, n_iter=3000, T0=0.05, T1=0.001):
    """Simulated annealing to escape local minima."""
    C = C0.copy()
    current_r = _exact_radius(C)
    best_C = C.copy()
    best_r = current_r
    rng = np.random.default_rng(999)

    for it in range(n_iter):
        if time.time() > deadline:
            break
        T = T0 * (T1 / T0) ** (it / n_iter)

        # Random perturbation
        C_new = C.copy()
        mask = rng.random(N) < 0.3
        C_new[mask] += rng.normal(0, T * 0.5, size=(np.sum(mask), 3))
        C_new = np.clip(C_new, -0.4999, 0.4999)

        new_r = _exact_radius(C_new)

        if new_r > current_r or rng.random() < np.exp((new_r - current_r) / max(T, 1e-10)):
            C = C_new
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_C = C.copy()

    return best_C, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Generate diverse initializations
    inits = []
    inits.append(_bcc_lattice_init())
    inits.append(_fcc_lattice_init())
    inits.append(_grid_init())
    for s in [1, 2, 3, 4, 5, 6]:
        inits.append(_random_init(s))

    # Select top-6 by initial radius
    init_radii = [_exact_radius(C) for C in inits]
    top_idx = np.argsort(init_radii)[::-1][:6]
    top_inits = [inits[i] for i in top_idx]

    best_C = top_inits[0].copy()
    best_r = init_radii[top_idx[0]]

    # Beta schedule: coarse to fine
    betas = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]

    for C0 in top_inits:
        if time.time() > deadline - 10.0:
            break

        cur_C = C0.copy()
        cur_r = _exact_radius(cur_C)

        for beta in betas:
            if time.time() > deadline - 10.0:
                break
            cur_C, cur_r = _optimize_smooth(cur_C, beta, deadline, maxiter=1500)
            if cur_r > best_r:
                best_r = cur_r
                best_C = cur_C.copy()

    # Simulated annealing from the best solution
    if time.time() < deadline - 20.0:
        sa_C, sa_r = _simulated_annealing(best_C, deadline, n_iter=5000, T0=0.03, T1=0.0005)
        if sa_r > best_r:
            best_r = sa_r
            best_C = sa_C.copy()

    # Final high-precision polish
    if time.time() < deadline - 5.0:
        beta_final = 1000.0
        final_C, final_r = _optimize_smooth(best_C, beta_final, deadline, maxiter=5000)
        if final_r > best_r:
            best_r = final_r
            best_C = final_C.copy()

    # Ensure validity
    best_C = np.clip(best_C, -0.5 + 1e-12, 0.5 - 1e-12)

    return best_C
# EVOLVE-BLOCK-END
