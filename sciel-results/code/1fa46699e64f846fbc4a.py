"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _grid_init(side, n=N):
    """Create a grid of up to n circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.48 / side
                positions.append([x, y, r])
    while len(positions) < n:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:n])


def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.15, 0.85, size=(n, 2))
    r = rng.uniform(0.02, 0.04, size=n)
    return xy, r


def _physics_relax(xy, r, deadline, n_steps=3000, dt=0.005, k_repel=50.0,
                   k_wall=200.0, damping=0.85, r_start=0.03, r_target=0.20,
                   seed=0):
    """
    Physics-based soft-body relaxation:
    - Repulsion between overlapping circles
    - Spring force pushing circles away from walls
    - Radii grow over time (annealing-like)
    - Velocity Verlet integration with damping
    Returns (xy, r) after relaxation.
    """
    n = len(xy)
    xy = xy.astype(np.float64).copy()
    r = r.astype(np.float64).copy()
    vel = np.zeros_like(xy)

    rng = np.random.RandomState(seed)

    for step in range(n_steps):
        if time.time() > deadline:
            break

        # Animate radius growth: ease from r_start toward r_target
        t = step / max(n_steps - 1, 1)
        # Smoothstep for nicer easing
        t_ease = t * t * (3.0 - 2.0 * t)
        r_current = r_start + (r_target - r_start) * t_ease
        # But cap at current r values (they can only grow, never shrink below initial)
        r_current = np.maximum(r_current, r)
        r = r_current

        # Compute forces
        forces = np.zeros_like(xy)

        # Wall forces: push circle away from walls if it would go outside
        for i in range(n):
            ri = r[i]
            # Left wall
            if xy[i, 0] < ri:
                forces[i, 0] += k_wall * (ri - xy[i, 0])
            # Right wall
            if xy[i, 0] > 1.0 - ri:
                forces[i, 0] -= k_wall * (xy[i, 0] - (1.0 - ri))
            # Bottom wall
            if xy[i, 1] < ri:
                forces[i, 1] += k_wall * (ri - xy[i, 1])
            # Top wall
            if xy[i, 1] > 1.0 - ri:
                forces[i, 1] -= k_wall * (xy[i, 1] - (1.0 - ri))

        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-6
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Force magnitude proportional to overlap
                    f_mag = k_repel * overlap
                    if dist > 1e-12:
                        fx = f_mag * dx / dist
                        fy = f_mag * dy / dist
                    else:
                        # Random direction for coincident circles
                        angle = rng.uniform(0, 2 * np.pi)
                        fx = f_mag * np.cos(angle)
                        fy = f_mag * np.sin(angle)
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy

        # Velocity Verlet integration
        # v_half = v + 0.5 * dt * F/m  (m=1)
        vel += 0.5 * dt * forces
        vel *= damping  # Apply damping to half-step velocity
        xy += dt * vel
        # Second half: recompute forces at new position? For simplicity, use same forces
        # (semi-implicit Euler is more stable here)
        vel += 0.5 * dt * forces
        vel *= damping

        # Hard-clamp positions to [r, 1-r] to prevent escape
        for i in range(n):
            ri = r[i]
            xy[i, 0] = np.clip(xy[i, 0], ri, 1.0 - ri)
            xy[i, 1] = np.clip(xy[i, 1], ri, 1.0 - ri)

    # After relaxation, set each circle's radius to its actual clearance
    for i in range(n):
        ri = r[i]
        # Wall clearance
        wall_clear = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
        # Circle clearance
        min_circle_clear = wall_clear
        for j in range(n):
            if i == j:
                continue
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_circle_clear = min(min_circle_clear, dist - r[j])
        r[i] = max(0.001, min(ri, wall_clear, min_circle_clear))

    return xy, r


def _best_fit_local_search(xy, r, deadline, grid_size=80):
    """
    Iteratively update each circle to the best position (maximizing its radius)
    given all other circles fixed. Vectorized over a grid of candidate positions.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 30
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _final_polish(xy, r, deadline, grid_size=120):
    """
    Final high-density local search polish.
    """
    return _best_fit_local_search(xy, r, deadline, grid_size=grid_size)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Physics-based relaxation from multiple initializations
    initializations = []

    # Grid inits
    for side in [4, 5, 6]:
        pos = _grid_init(side)
        initializations.append((pos[:, :2].copy(), pos[:, 2].copy(), 'grid'))

    # Random inits
    for seed in range(8):
        xy, r = _random_init(N, seed=seed)
        initializations.append((xy, r, f'rand{seed}'))

    for idx, (xy, r, label) in enumerate(initializations):
        if time.time() > deadline:
            break

        # Physics relaxation
        phys_xy, phys_r = _physics_relax(xy, r, deadline,
                                         n_steps=2000, dt=0.005,
                                         k_repel=50.0, k_wall=200.0,
                                         damping=0.85, r_start=0.03,
                                         r_target=0.18, seed=idx)

        # Local search polish after physics
        out_xy, out_r, total, valid = _best_fit_local_search(phys_xy, phys_r, deadline, grid_size=60)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: If we have a good result, do a final high-density polish
    if best_valid and best_xy is not None:
        if time.time() < deadline - 10:
            final_xy, final_r, final_total, final_valid = _final_polish(
                best_xy, best_r, deadline, grid_size=120)
            if final_valid and final_total > best_total:
                best_total = final_total
                best_xy = final_xy
                best_r = final_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
