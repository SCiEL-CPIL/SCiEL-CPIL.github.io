"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START

def _validate_config(xs, ys, rs, W, H, tol=1e-6):
    """Check whether configuration is valid: circles inside rectangle, disjoint interiors."""
    n = len(xs)
    for i in range(n):
        if xs[i] < rs[i] - tol or xs[i] > W - rs[i] + tol:
            return False
        if ys[i] < rs[i] - tol or ys[i] > H - rs[i] + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx * dx + dy * dy
            min_dist = rs[i] + rs[j]
            if dist_sq < min_dist * min_dist - tol:
                return False
    return True


def _relax(xs, ys, rs, W, H, max_iters=5000, deadline=None, lr_init=0.02):
    """
    Physics-based relaxation using repulsion forces for overlaps and walls.
    Returns (xs, ys, converged) where converged indicates no overlap/wall violation.
    """
    n = len(xs)
    xs = xs.copy()
    ys = ys.copy()
    lr = lr_init
    best_energy = np.inf
    no_improve_count = 0
    max_no_improve = 500

    def compute_forces():
        fx = np.zeros(n)
        fy = np.zeros(n)
        # Pairwise
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx * dx + dy * dy
                min_d = rs[i] + rs[j]
                if dist_sq < min_d * min_d:
                    if dist_sq < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        nx, ny = np.cos(angle), np.sin(angle)
                    else:
                        d = np.sqrt(dist_sq)
                        nx, ny = dx / d, dy / d
                    overlap = min_d - np.sqrt(dist_sq)
                    f = 2.0 * overlap
                    fx[i] += nx * f
                    fy[i] += ny * f
                    fx[j] -= nx * f
                    fy[j] -= ny * f
        # Walls
        wall_k = 10.0
        for i in range(n):
            if xs[i] < rs[i]:
                fx[i] += wall_k * (rs[i] - xs[i])
            if xs[i] > W - rs[i]:
                fx[i] -= wall_k * (xs[i] - (W - rs[i]))
            if ys[i] < rs[i]:
                fy[i] += wall_k * (rs[i] - ys[i])
            if ys[i] > H - rs[i]:
                fy[i] -= wall_k * (ys[i] - (H - rs[i]))
        return fx, fy

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break

        fx, fy = compute_forces()

        # Adaptive lr: scale down if forces are large
        max_f = np.max(np.sqrt(fx * fx + fy * fy))
        if max_f > 0:
            step = min(lr, 0.5 / max_f)
        else:
            step = 0.0

        xs += step * fx
        ys += step * fy

        # Hard clip to bounds
        xs = np.clip(xs, rs, W - rs)
        ys = np.clip(ys, rs, H - rs)

        # Check convergence
        if _validate_config(xs, ys, rs, W, H, tol=1e-7):
            return xs, ys, True

        # Decay learning rate slowly
        if it % 100 == 0:
            lr *= 0.995

        # Track energy for early stop
        energy = np.sum(fx * fx) + np.sum(fy * fy)
        if energy > best_energy:
            no_improve_count += 1
        else:
            best_energy = energy
            no_improve_count = 0
        if no_improve_count > max_no_improve:
            break

    return xs, ys, _validate_config(xs, ys, rs, W, H, tol=1e-7)


def _grid_init(cols, rows, W, H):
    """Initialize circle centers on a grid."""
    xs = np.zeros(N)
    ys = np.zeros(N)
    for i in range(N):
        c = i % cols
        r_idx = i // cols
        xs[i] = (c + 0.5) * W / cols
        ys[i] = (r_idx + 0.5) * H / rows
    return xs, ys


def _random_init(W, H, r):
    """Random initialization within valid region."""
    xs = np.random.uniform(r, W - r, N)
    ys = np.random.uniform(r, H - r, N)
    return xs, ys


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_sum_r = 0.0
    best_result = None

    # Aspect ratios to try (d = W - 1, so W = 1+d, H = 1-d, perimeter = 4)
    # Focus on a range of aspect ratios
    aspect_ratios = [0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4, 0.5, -0.5]

    for d in aspect_ratios:
        if time.time() > deadline - 15:
            break

        W = 1.0 + d
        H = 1.0 - d
        if W <= 0.15 or H <= 0.15:
            continue

        # Theoretical max radius (area bound): N*pi*r^2 <= W*H => r <= sqrt(W*H/(N*pi))
        r_area_max = np.sqrt(W * H / (N * np.pi))
        # Geometric max: smallest dimension / 2 for a single row/col, but with N circles
        # A better estimate: pack in grid
        r_geom_max = min(W, H) / 2.0 * 0.6  # conservative
        r_max_est = min(r_area_max, r_geom_max)

        if r_max_est <= 0.01:
            continue

        # Try multiple initializations: grid + random
        inits = []

        # Grid init with various column counts
        for cols in [3, 4, 5, 6]:
            if cols > N:
                continue
            rows = int(np.ceil(N / cols))
            xs, ys = _grid_init(cols, rows, W, H)
            inits.append((xs.copy(), ys.copy()))

        # Random inits
        r_small = 0.1 * r_max_est
        for _ in range(3):
            xs, ys = _random_init(W, H, r_small)
            inits.append((xs.copy(), ys.copy()))

        for xs_init, ys_init in inits:
            if time.time() > deadline - 10:
                break

            # Binary search on radius
            r_lo = 0.05 * r_max_est
            r_hi = 1.05 * r_max_est

            # First ensure a valid config at r_lo
            rs_test = np.full(N, r_lo)
            xs_cur = xs_init.copy()
            ys_cur = ys_init.copy()
            _, _, valid = _relax(xs_cur, ys_cur, rs_test, W, H, max_iters=1000, deadline=deadline - 5, lr_init=0.03)
            if not valid:
                # Try random again
                xs_cur, ys_cur = _random_init(W, H, r_lo)
                _, _, valid = _relax(xs_cur, ys_cur, rs_test, W, H, max_iters=1000, deadline=deadline - 5, lr_init=0.03)
                if not valid:
                    continue

            # Binary search
            best_r = r_lo
            for _ in range(20):
                if time.time() > deadline - 8:
                    break
                r_mid = (r_lo + r_hi) / 2.0
                rs_mid = np.full(N, r_mid)

                # Start from current best positions
                xs_trial = xs_cur.copy()
                ys_trial = ys_cur.copy()

                # Scale positions slightly if needed (not needed for same aspect ratio)
                _, _, valid = _relax(xs_trial, ys_trial, rs_mid, W, H, max_iters=2000, deadline=deadline - 5, lr_init=0.02)

                if valid:
                    r_lo = r_mid
                    xs_cur = xs_trial.copy()
                    ys_cur = ys_trial.copy()
                    best_r = r_mid
                else:
                    r_hi = r_mid

            total_r = best_r * N
            if total_r > best_sum_r:
                best_sum_r = total_r
                best_result = np.column_stack((xs_cur, ys_cur, np.full(N, best_r)))

    # Fallback: simple grid in unit square (perimeter 4)
    if best_result is None or best_sum_r <= 0:
        # 5x5 grid, take first 21
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([x, y, r])
                if len(cells) == N:
                    break
            if len(cells) == N:
                break
        best_result = np.array(cells)
        best_sum_r = N * (0.5 / side - 1e-4)

    # Final validation
    if best_result is not None:
        xs = best_result[:, 0]
        ys = best_result[:, 1]
        rs = best_result[:, 2]
        W = 2.0  # perimeter 4 => W+H=2, but we need to know W,H
        # We need to recover W and H. Since perimeter is 2(W+H)=4, W+H=2.
        # The bounding box of the circles gives us W and H.
        # Actually, the problem says "bounding rectangle has perimeter at most 4".
        # The evaluator checks that the circles fit in some rectangle with perimeter <= 4.
        # So we just need to ensure circles are in [0,W] x [0,H] with 2(W+H)<=4.
        # Let's compute the bounding box and check.
        x_min = np.max(0, np.min(xs - rs))
        x_max = np.max(xs + rs)
        y_min = np.max(0, np.min(ys - rs))
        y_max = np.max(ys + rs)
        bbox_W = x_max - x_min
        bbox_H = y_max - y_min
        perimeter = 2 * (bbox_W + bbox_H)

        if perimeter <= 4.0 + 1e-5:
            return best_result
        else:
            # Scale down to fit
            # Need to shrink so that 2(W+H) <= 4
            # Current perimeter is P, need P' = 4
            # Scale factor s: s^2 * P <= 4 (if we scale both W and H by s)
            # Actually scaling positions and radii by s scales W and H by s, so perimeter by s.
            s = 4.0 / perimeter * 0.999
            best_result[:, 0] *= s
            best_result[:, 1] *= s
            best_result[:, 2] *= s
            return best_result

    return best_result

# EVOLVE-BLOCK-END
