"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time


def _eval_packing(xy, r):
    """Evaluate packing: return (total_radius, is_valid)."""
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


def _force_directed_packing(xy, r, deadline, n_iters=5000, dt=0.002, damping=0.92):
    """
    Force-directed relaxation: circles repel each other and are attracted to walls
    when they can grow. We iteratively:
      1. Compute repulsive forces between overlapping/near circles.
      2. Compute wall forces pushing circles inward.
      3. Update positions with velocity + damping.
      4. Try to grow all radii slightly if no overlaps.
    """
    n = len(xy)
    xy = xy.copy().astype(np.float64)
    r = r.copy().astype(np.float64)
    vel = np.zeros_like(xy)

    # Target radius growth schedule
    max_r = 0.45
    grow_rate = 0.00008  # per-iteration radius growth when feasible

    best_total = 0.0
    best_valid = False
    best_xy = xy.copy()
    best_r = r.copy()

    # Initial validity check
    total, valid = _eval_packing(xy, r)
    if valid:
        best_total = total
        best_valid = True

    for it in range(n_iters):
        if time.time() > deadline:
            break

        # Compute forces
        forces = np.zeros_like(xy)

        # Pairwise repulsion: if dist < r_i + r_j + margin, push apart
        margin = 0.005
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-12:
                    # Jitter
                    dx = 0.001 * (np.random.rand() - 0.5)
                    dy = 0.001 * (np.random.rand() - 0.5)
                    dist = np.sqrt(dx * dx + dy * dy)
                overlap = (r[i] + r[j] + margin) - dist
                if overlap > 0:
                    # Repulsive force proportional to overlap
                    fx = overlap * dx / dist
                    fy = overlap * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy

        # Wall forces: push inward if too close to wall
        wall_margin = 0.002
        for i in range(n):
            # Left wall
            if xy[i, 0] - r[i] < wall_margin:
                forces[i, 0] += (wall_margin - (xy[i, 0] - r[i])) * 2.0
            # Right wall
            if 1.0 - xy[i, 0] - r[i] < wall_margin:
                forces[i, 0] -= (wall_margin - (1.0 - xy[i, 0] - r[i])) * 2.0
            # Bottom wall
            if xy[i, 1] - r[i] < wall_margin:
                forces[i, 1] += (wall_margin - (xy[i, 1] - r[i])) * 2.0
            # Top wall
            if 1.0 - xy[i, 1] - r[i] < wall_margin:
                forces[i, 1] -= (wall_margin - (1.0 - xy[i, 1] - r[i])) * 2.0

        # Update velocities and positions
        vel = vel * damping + forces * dt
        xy += vel

        # Hard clip to valid region (respect current radii)
        for i in range(n):
            ri = r[i]
            xy[i, 0] = np.clip(xy[i, 0], ri, 1.0 - ri)
            xy[i, 1] = np.clip(xy[i, 1], ri, 1.0 - ri)

        # Try to grow radii if packing is valid
        total, valid = _eval_packing(xy, r)
        if valid:
            # Grow all radii slightly
            for i in range(n):
                r[i] = min(r[i] + grow_rate, max_r)
            # Re-validate after growth
            total, valid = _eval_packing(xy, r)
            if not valid:
                # Rollback growth
                for i in range(n):
                    r[i] = max(r[i] - grow_rate, 0.01)

        # Track best
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        # Periodic: if stuck, add small random kick
        if it % 500 == 0 and it > 0:
            for i in range(n):
                vel[i] += np.random.randn(2) * 0.001

    return best_xy, best_r, best_total, best_valid


def _coordinate_descent_polish(xy, r, deadline, n_sweeps=50):
    """
    Coordinate descent: for each circle, find the best position (given others fixed)
    that maximizes its radius, using a fine grid + Newton refinement.
    """
    n = len(xy)
    xy = xy.copy().astype(np.float64)
    r = r.copy().astype(np.float64)

    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if not best_valid:
        best_total = 0.0

    grid_size = 120
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    for sweep in range(n_sweeps):
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
                r_cands = np.minimum(wall_clear, circle_clear)
            else:
                r_cands = wall_clear.copy()

            idx = int(np.argmax(r_cands))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_cands[idx]

            if new_r > r[i] + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

            # Fine refinement around best
            step = 1.0 / grid_size
            fx0 = max(0.0, new_x - 2 * step)
            fx1 = min(1.0, new_x + 2 * step)
            fy0 = max(0.0, new_y - 2 * step)
            fy1 = min(1.0, new_y + 2 * step)
            fine_gx = np.linspace(fx0, fx1, 30)
            fine_gy = np.linspace(fy0, fy1, 30)
            FGX, FGY = np.meshgrid(fine_gx, fine_gy, indexing='ij')
            fine_cands = np.column_stack([FGX.ravel(), FGY.ravel()])
            FM = len(fine_cands)
            if FM > 0:
                f_wall = np.minimum(np.minimum(fine_cands[:, 0], 1.0 - fine_cands[:, 0]),
                                    np.minimum(fine_cands[:, 1], 1.0 - fine_cands[:, 1]))
                if len(others_xy) > 0:
                    fdx = fine_cands[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                    fdy = fine_cands[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                    fdists = np.sqrt(fdx * fdx + fdy * fdy)
                    fcirc = np.min(fdists - others_r[np.newaxis, :], axis=1)
                    f_r_cands = np.minimum(f_wall, fcirc)
                else:
                    f_r_cands = f_wall.copy()
                fidx = int(np.argmax(f_r_cands))
                if f_r_cands[fidx] > r[i] + 1e-10:
                    xy[i, 0] = fine_cands[fidx, 0]
                    xy[i, 1] = fine_cands[fidx, 1]
                    r[i] = f_r_cands[fidx]
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


def _grid_init(side):
    """Create a grid of up to N circles."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.48 / side
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    r = rng.uniform(0.02, 0.04, size=n)
    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Force-directed from multiple grid initializations
    for side in [5, 6, 7]:
        if time.time() > deadline:
            break
        init = _grid_init(side)
        xy = init[:, :2].copy()
        r = init[:, 2].copy()
        fd_xy, fd_r, fd_total, fd_valid = _force_directed_packing(xy, r, deadline, n_iters=4000)
        if fd_valid and fd_total > best_total:
            best_total = fd_total
            best_valid = True
            best_xy = fd_xy
            best_r = fd_r

    # Phase 2: Force-directed from random initializations
    for seed in range(10):
        if time.time() > deadline:
            break
        xy, r = _random_init(N, seed=seed)
        fd_xy, fd_r, fd_total, fd_valid = _force_directed_packing(xy, r, deadline, n_iters=3000)
        if fd_valid and fd_total > best_total:
            best_total = fd_total
            best_valid = True
            best_xy = fd_xy
            best_r = fd_r

    # Phase 3: Coordinate descent polish on best result
    if best_valid and best_xy is not None and time.time() < deadline - 5:
        cd_xy, cd_r, cd_total, cd_valid = _coordinate_descent_polish(
            best_xy.copy(), best_r.copy(), deadline, n_sweeps=80)
        if cd_valid and cd_total > best_total + 1e-10:
            best_total = cd_total
            best_xy = cd_xy
            best_r = cd_r

    # Phase 4: Another force-directed pass from the polished result (different dynamics)
    if best_valid and best_xy is not None and time.time() < deadline - 5:
        fd2_xy, fd2_r, fd2_total, fd2_valid = _force_directed_packing(
            best_xy.copy(), best_r.copy(), deadline, n_iters=3000, dt=0.001, damping=0.95)
        if fd2_valid and fd2_total > best_total + 1e-10:
            best_total = fd2_total
            best_xy = fd2_xy
            best_r = fd2_r

        # Final CD polish
        if time.time() < deadline - 2:
            cd2_xy, cd2_r, cd2_total, cd2_valid = _coordinate_descent_polish(
                best_xy.copy(), best_r.copy(), deadline, n_sweeps=50)
            if cd2_valid and cd2_total > best_total + 1e-10:
                best_total = cd2_total
                best_xy = cd2_xy
                best_r = cd2_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        init = _grid_init(side)
        xy = init[:, :2].copy()
        r = init[:, 2].copy()
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
