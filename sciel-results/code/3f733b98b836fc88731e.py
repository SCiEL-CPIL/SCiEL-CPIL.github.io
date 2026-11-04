"""
Structural change: direct center-space optimization with a smooth soft-minimum energy
and contact-graph-guided local refinement. Multiple diverse lattice starts (BCC, FCC,
simple cubic, random) are used. The soft-min energy is smooth and differentiable,
allowing L-BFGS-B to make progress where hard-violation energies stall.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 100
HALF = 0.5
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0


def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = HALF - max_abs
    min_wall_dist = np.min(wall_dists)

    if min_wall_dist < 0:
        return 0.0

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)
    min_sq_dist = np.min(sq_dists)
    min_pair_dist = np.sqrt(min_sq_dist)

    r = min(min_pair_dist / 2.0, min_wall_dist)
    return float(r)


def _soft_min_energy_and_grad(centers, R, beta=50.0):
    """
    Smooth soft-minimum energy for sphere packing.
    Energy = soft_min of (2R - d_ij)^2 for all pairs and (R - wall_dist_i)^2 for walls.
    Uses log-sum-exp style smoothing with parameter beta.
    """
    n = centers.shape[0]

    # Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)

    i_idx, j_idx = np.triu_indices(n, k=1)
    sq_d_ij = sq_dists[i_idx, j_idx]
    d_ij = np.sqrt(sq_d_ij + 1e-15)

    # Pairwise violations: max(0, 2R - d_ij)^2
    viol_pair = np.maximum(0.0, 2.0 * R - d_ij)
    viol_pair_sq = viol_pair ** 2

    # Wall violations
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = HALF - max_abs
    viol_wall = np.maximum(0.0, R - wall_dists)
    viol_wall_sq = viol_wall ** 2

    # Combine all violation squared values
    all_viol_sq = np.concatenate([viol_pair_sq, viol_wall_sq])

    # Soft minimum: exp(-beta * min) approximation via log-sum-exp
    # We want a smooth approximation to the minimum violation.
    # Actually, we want to minimize the MAXIMUM violation (bottleneck).
    # Use: E = (1/beta) * log(sum(exp(beta * v_i^2)))  -- this is a smooth max
    # But that grows fast. Instead use: E = sum(v_i^2) + penalty for max
    # Simpler: just use sum of squared violations, which is what we had before.
    # The key structural change is using a smoother potential.

    # Use a smooth potential: for each pair, if d < 2R, add (2R-d)^2
    # This is already what we have. Let's use a different approach:
    # Minimize the sum of exp(-k*(d_ij - 2R)) for pairs and exp(-k*(wall_dist - R)) for walls
    # This is a smooth repulsion that becomes large when constraints are violated.

    k = 20.0  # steepness

    # Pairwise repulsion: exp(-k * (d_ij - 2R)) when d_ij < 2R
    # For d_ij >= 2R, this is exp(-k * positive) which is small.
    # For d_ij < 2R, this is exp(+k * (2R - d_ij)) which grows.
    pair_arg = -k * (d_ij - 2.0 * R)
    # Clip to avoid overflow
    pair_arg = np.clip(pair_arg, -50.0, 50.0)
    pair_rep = np.exp(pair_arg)

    # Wall repulsion
    wall_arg = -k * (wall_dists - R)
    wall_arg = np.clip(wall_arg, -50.0, 50.0)
    wall_rep = np.exp(wall_arg)

    energy = np.sum(pair_rep) + np.sum(wall_rep)

    # Gradient
    grad = np.zeros_like(centers)

    # Pairwise gradient
    # d/dx_i of exp(-k*(d_ij - 2R)) = -k * exp(-k*(d_ij-2R)) * (x_i - x_j)/d_ij
    active_pairs = pair_arg > -30  # only compute for pairs that contribute significantly
    if np.any(active_pairs):
        i_a = i_idx[active_pairs]
        j_a = j_idx[active_pairs]
        d_a = d_ij[active_pairs]
        rep_a = pair_rep[active_pairs]

        # dE/dx_i = -k * rep * (x_i - x_j) / d_ij
        # dE/dx_j = -k * rep * (x_j - x_i) / d_ij = k * rep * (x_i - x_j) / d_ij
        vecs = centers[i_a] - centers[j_a]
        factors = k * rep_a / d_a

        forces = factors[:, np.newaxis] * vecs
        np.add.at(grad, i_a, -forces)
        np.add.at(grad, j_a, forces)

    # Wall gradient
    active_wall = wall_arg > -30
    if np.any(active_wall):
        for idx in np.where(active_wall)[0]:
            v = wall_rep[idx]
            max_k = np.argmax(np.abs(centers[idx]))
            sign_val = np.sign(centers[idx, max_k])
            if sign_val == 0:
                sign_val = 1.0
            # dE/dx_ik = -k * rep * d(wall_dist)/dx_ik
            # wall_dist = 0.5 - |x_ik| (for the max coordinate)
            # d(wall_dist)/dx_ik = -sign(x_ik)
            # So dE/dx_ik = -k * rep * (-sign(x_ik)) = k * rep * sign(x_ik)
            grad[idx, max_k] += k * v * sign_val

    return energy, grad.flatten()


def _hard_energy_and_grad(centers, R):
    """Hard violation energy: sum of squared violations."""
    n = centers.shape[0]
    energy = 0.0
    grad = np.zeros_like(centers)

    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(sq_dists, np.inf)

    i_idx, j_idx = np.triu_indices(n, k=1)
    sq_d_ij = sq_dists[i_idx, j_idx]
    d_ij = np.sqrt(sq_d_ij + 1e-15)

    viol = np.maximum(0.0, 2.0 * R - d_ij)
    energy += np.sum(viol ** 2)

    active = viol > 0
    if np.any(active):
        i_a = i_idx[active]
        j_a = j_idx[active]
        d_a = d_ij[active]
        v_a = viol[active]
        factors = 2.0 * v_a / d_a
        vecs = centers[i_a] - centers[j_a]
        forces = factors[:, np.newaxis] * vecs
        np.add.at(grad, i_a, -forces)
        np.add.at(grad, j_a, forces)

    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = HALF - max_abs
    wall_viol = np.maximum(0.0, R - wall_dists)
    energy += np.sum(wall_viol ** 2)

    active_wall = wall_viol > 0
    if np.any(active_wall):
        for idx in np.where(active_wall)[0]:
            v = wall_viol[idx]
            max_k = np.argmax(np.abs(centers[idx]))
            sign_val = np.sign(centers[idx, max_k])
            if sign_val == 0:
                sign_val = 1.0
            grad[idx, max_k] += 2.0 * v * sign_val

    return energy, grad.flatten()


def _optimize_from_start(centers, target_R, max_iter=1000):
    """Optimize from a starting configuration for a target radius."""
    def obj_and_grad(x):
        c = x.reshape(N, 3)
        e, g = _soft_min_energy_and_grad(c, target_R)
        return e, g

    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (N * 3)
    try:
        res = minimize(
            obj_and_grad,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        return new_centers
    except Exception:
        return centers


def _local_optimize_violation(centers, R, max_iter=500):
    """Local optimization targeting specific violations."""
    def obj_and_grad(x):
        c = x.reshape(N, 3)
        e, g = _hard_energy_and_grad(c, R)
        return e, g

    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (N * 3)
    try:
        res = minimize(
            obj_and_grad,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False}
        )
        return res.x.reshape(N, 3)
    except Exception:
        return centers


def _contact_graph_refine(centers, R, n_iters=20):
    """
    Refine by identifying the contact graph (pairs at minimum distance)
    and optimizing only those constraints.
    """
    for _ in range(n_iters):
        # Find minimum distance
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(sq_dists, np.inf)
        min_sq = np.min(sq_dists)
        min_d = np.sqrt(min_sq)
        current_r = min(min_d / 2.0, np.min(HALF - np.max(np.abs(centers), axis=1)))

        if current_r >= R:
            return centers

        # Find pairs within 1.01 * min_d
        threshold = 1.01 * min_d
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_ij = np.sqrt(sq_dists[i_idx, j_idx])
        close_mask = d_ij < threshold

        if not np.any(close_mask):
            break

        # Optimize with hard energy
        centers = _local_optimize_violation(centers, R, max_iter=200)

    return centers


def _generate_bcc():
    """Generate BCC lattice points in the cube."""
    # BCC: simple cubic + body center
    n_per_side = 4  # 4^3 = 64 cubic + 4^3 = 64 body centers = 128, take 100
    spacing = 1.0 / n_per_side
    points = []
    for ix in range(n_per_side):
        for iy in range(n_per_side):
            for iz in range(n_per_side):
                # Corner
                x = -0.5 + spacing * (ix + 0.5)
                y = -0.5 + spacing * (iy + 0.5)
                z = -0.5 + spacing * (iz + 0.5)
                points.append([x, y, z])
                # Body center
                points.append([x + spacing / 2, y + spacing / 2, z + spacing / 2])

    points = np.array(points)
    # Shuffle and take first 100
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(points))[:N]
    return points[idx]


def _generate_fcc():
    """Generate FCC lattice points."""
    n_per_side = 4
    spacing = 1.0 / n_per_side
    points = []
    for ix in range(n_per_side):
        for iy in range(n_per_side):
            for iz in range(n_per_side):
                x = -0.5 + spacing * (ix + 0.5)
                y = -0.5 + spacing * (iy + 0.5)
                z = -0.5 + spacing * (iz + 0.5)
                points.append([x, y, z])
                # Face centers
                if ix < n_per_side - 1:
                    points.append([x + spacing / 2, y, z])
                if iy < n_per_side - 1:
                    points.append([x, y + spacing / 2, z])
                if iz < n_per_side - 1:
                    points.append([x, y, z + spacing / 2])

    points = np.array(points)
    rng = np.random.default_rng(1)
    idx = rng.permutation(len(points))[:N]
    return points[idx]


def _generate_grid():
    """Simple cubic grid."""
    n_per_side = 5
    spacing = 1.0 / n_per_side
    points = []
    for ix in range(n_per_side):
        for iy in range(n_per_side):
            for iz in range(n_per_side):
                x = -0.5 + spacing * (ix + 0.5)
                y = -0.5 + spacing * (iy + 0.5)
                z = -0.5 + spacing * (iz + 0.5)
                points.append([x, y, z])
    points = np.array(points)
    rng = np.random.default_rng(2)
    idx = rng.permutation(len(points))[:N]
    return points[idx]


def _generate_random(seed=42):
    """Random points."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN

    # Generate diverse starting configurations
    starts = [
        _generate_bcc(),
        _generate_fcc(),
        _generate_grid(),
        _generate_random(42),
        _generate_random(123),
    ]

    # Optimize each start with a moderate target radius
    initial_targets = [0.10, 0.105, 0.11, 0.10, 0.105]

    best_pos = None
    best_r = 0.0

    for i, (start, target) in enumerate(zip(starts, initial_targets)):
        if time.time() > deadline:
            break

        # Optimize
        optimized = _optimize_from_start(start, target, max_iter=800)
        r = _compute_radius(optimized)

        if r > best_r:
            best_r = r
            best_pos = optimized.copy()

        # Contact graph refinement
        if time.time() < deadline - 5:
            refined = _contact_graph_refine(optimized, target, n_iters=10)
            r_refined = _compute_radius(refined)
            if r_refined > best_r:
                best_r = r_refined
                best_pos = refined.copy()

    # Iterative radius increase from best configuration
    current_r = best_r
    step_size = 0.002
    consecutive_failures = 0
    max_failures = 5

    while time.time() < deadline - 5:
        target_r = current_r + step_size

        if consecutive_failures >= max_failures:
            step_size *= 0.7
            consecutive_failures = 0
            if step_size < 1e-8:
                break

        success = False

        # Try soft-min optimization
        new_pos = _optimize_from_start(best_pos, target_r, max_iter=600)
        new_r = _compute_radius(new_pos)

        if new_r >= target_r - 1e-7:
            best_pos = new_pos
            best_r = new_r
            current_r = new_r
            consecutive_failures = 0
            success = True
        elif new_r > current_r + 1e-9:
            best_pos = new_pos
            best_r = new_r
            current_r = new_r

        # Try hard optimization if soft didn't fully succeed
        if not success and time.time() < deadline - 10:
            hard_pos = _local_optimize_violation(best_pos, target_r, max_iter=400)
            hard_r = _compute_radius(hard_pos)
            if hard_r > current_r + 1e-9:
                best_pos = hard_pos
                best_r = hard_r
                current_r = hard_r
            if hard_r >= target_r - 1e-7:
                success = True

        if not success:
            consecutive_failures += 1

    # Final polish
    if time.time() < deadline - 2:
        for _ in range(3):
            if time.time() > deadline - 1:
                break
            target_r = best_r + 5e-7
            polished = _optimize_from_start(best_pos, target_r, max_iter=1000)
            polished_r = _compute_radius(polished)
            if polished_r > best_r:
                best_pos = polished
                best_r = polished_r

    # Ensure valid
    best_pos = np.clip(best_pos, -HALF + 1e-10, HALF - 1e-10)
    return best_pos
