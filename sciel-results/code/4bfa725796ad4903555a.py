"""Sphere packing via repulsion annealing + Metropolis MC + refinement."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centers):
    """Compute the max common radius: min of half-pair-distances and wall clearances."""
    # Wall clearances: distance from each center to nearest face
    wall = 0.5 - np.abs(centers)  # shape (N,3)
    wall_min = wall.min()
    # Pair distances
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_min = dist.min() / 2.0
    return min(wall_min, pair_min)

def _compute_radius_grad(centers):
    """Return (radius, grad) where grad is d(radius)/d(centers) via subgradient of min."""
    wall = 0.5 - np.abs(centers)  # (N,3)
    wall_min = wall.min()
    # subgradient of min over wall: pick argmin
    wall_grad = np.zeros_like(centers)
    idx = np.unravel_index(wall.argmin(), wall.shape)
    # d/dx of (0.5 - |x|) is -sign(x) at argmin
    wall_grad[idx] = -np.sign(centers[idx])
    if np.all(centers[idx] == 0):
        wall_grad[idx] = 0.0  # subgradient at 0, pick 0

    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_min = dist.min()
    d_idx = np.unravel_index(dist.argmin(), dist.shape)
    pair_grad = np.zeros_like(centers)
    i, j = d_idx
    dvec = centers[i] - centers[j]
    dnorm = np.linalg.norm(dvec)
    if dnorm > 1e-15:
        unit = dvec / dnorm
        pair_grad[i] = 0.5 * unit
        pair_grad[j] = -0.5 * unit

    if wall_min <= pair_min:
        return wall_min, wall_grad
    else:
        return pair_min, pair_grad

def _repulsion_energy(centers, r_target):
    """Energy = sum of squared violations of min-pair-dist >= 2*r and wall >= r."""
    # Wall violations
    wall = r_target - (0.5 - np.abs(centers))
    wall_viol = np.maximum(wall, 0.0)
    wall_energy = np.sum(wall_viol ** 2)
    # Pair violations
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_viol = np.maximum(2.0 * r_target - dist, 0.0)
    pair_energy = 0.5 * np.sum(pair_viol ** 2)
    return wall_energy + pair_energy

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    np.random.seed(42)

    # --- Phase 1: Vectorized repulsion annealing ---
    # Start with random points, anneal a "temperature" that controls step size
    # Objective: maximize min-distance subject to wall constraint
    # We use a soft penalty approach: minimize sum of exp(-k * (dist - 2r)) style

    best_centers = None
    best_radius = 0.0

    def _score(c):
        r = _compute_radius(c)
        if r > best_radius:
            return r
        return r

    # Initialize with a reasonable random start
    centers = np.random.uniform(-0.4, 0.4, size=(N, 3))
    current_radius = _compute_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius

    # Repulsion-based gradient ascent with annealing
    # We use a repulsive force model: each pair repels with force ~ 1/d^2
    # Walls repel with force ~ 1/wall_dist^2
    # Anneal the step size

    n_iter_phase1 = 20000
    for it in range(n_iter_phase1):
        if time.time() > deadline - 30.0:
            break

        # Compute forces
        forces = np.zeros_like(centers)

        # Wall repulsion
        wall_dist = 0.5 - np.abs(centers)  # (N,3)
        # Push inward: force direction is sign(centers) * (1/wall_dist^2)
        # But we want to push AWAY from wall, i.e., toward center
        # Force on center i in axis k: -sign(c_ik) / wall_dist^2 (pushes toward 0)
        safe_wall = np.maximum(wall_dist, 0.01)
        wall_force = -np.sign(centers) / (safe_wall ** 2)
        forces += wall_force

        # Pair repulsion
        diff = centers[:, None, :] - centers[None, :, :]  # (N,N,3)
        dist_sq = (diff ** 2).sum(axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        # Force magnitude ~ 1/d^3 (so force vector ~ diff/d^3 = unit/d^2)
        # Only consider pairs that are too close (within some cutoff)
        cutoff = 0.5
        mask = dist < cutoff
        mask.fill_diagonal(False)
        # Force vector for pair (i,j): direction from j to i, magnitude 1/dist^2
        inv_dist_sq = np.zeros_like(dist_sq)
        inv_dist_sq[mask] = 1.0 / dist_sq[mask]
        # forces[i] += sum_j diff[i,j] * inv_dist_sq[i,j]
        pair_force = (diff * inv_dist_sq[:, :, None]).sum(axis=1)
        forces += pair_force

        # Annealed step size
        progress = it / max(n_iter_phase1, 1)
        step = 0.002 * (1.0 - progress) + 0.0001

        # Update
        centers = centers + step * forces

        # Clamp to cube
        centers = np.clip(centers, -0.49, 0.49)

        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()

    # --- Phase 2: Metropolis Monte Carlo with temperature annealing ---
    # Try random perturbations, accept if improves or with probability exp(-delta/T)
    n_iter_phase2 = 50000
    T_start = 0.01
    T_end = 0.0001

    for it in range(n_iter_phase2):
        if time.time() > deadline - 10.0:
            break

        T = T_start * (T_end / T_start) ** (it / max(n_iter_phase2, 1))

        # Random perturbation
        idx = np.random.randint(N)
        delta = np.random.normal(0, 0.02, size=3)
        new_centers = best_centers.copy()
        new_centers[idx] += delta
        new_centers[idx] = np.clip(new_centers[idx], -0.49, 0.49)

        new_r = _compute_radius(new_centers)
        delta_r = new_r - best_radius

        if delta_r >= 0 or np.random.random() < np.exp(delta_r / max(T, 1e-10)):
            best_centers = new_centers
            best_radius = new_r

    # --- Phase 3: Local refinement via steepest ascent on the active constraint ---
    # Identify which constraint is active and do targeted moves
    n_iter_phase3 = 10000
    for it in range(n_iter_phase3):
        if time.time() > deadline - 2.0:
            break

        r, grad = _compute_radius_grad(best_centers)
        if r > best_radius:
            best_radius = r

        # Steepest ascent step
        step = 0.001 * (1.0 - it / max(n_iter_phase3, 1)) + 0.0001
        new_centers = best_centers + step * grad
        new_centers = np.clip(new_centers, -0.49, 0.49)

        new_r = _compute_radius(new_centers)
        if new_r >= r - 1e-12:
            best_centers = new_centers
            if new_r > best_radius:
                best_radius = new_r

    # Final clamp to ensure validity
    best_centers = np.clip(best_centers, -0.4999, 0.4999)

    # Validate
    if not np.all(np.isfinite(best_centers)):
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    return best_centers
# EVOLVE-BLOCK-END
