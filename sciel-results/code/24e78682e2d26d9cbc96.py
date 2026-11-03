"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_init(n, r):
    # Hexagonal lattice init for n points with circle radius r in unit disk.
    # Return array (n,2) or None if not possible.
    dx = 2.0 * r
    dy = dx * np.sqrt(3.0) / 2.0
    k = int(1.0 / dx) + 3
    xs, ys = [], []
    for j in range(-k, k + 1):
        y = j * dy
        if abs(y) > 1.0 - r:
            continue
        offset = 0.5 * dx if (j % 2) else 0.0
        for i in range(-k, k + 1):
            x = i * dx + offset
            if x * x + y * y <= (1.0 - r) ** 2:
                xs.append(x)
                ys.append(y)
    pts = np.column_stack([xs, ys])
    if len(pts) < n:
        return None
    # Prefer points closer to origin for better packing.
    d = np.linalg.norm(pts, axis=1)
    idx = np.argsort(d, kind="stable")[:n]
    return pts[idx]


def _radius(p):
    # Compute the common radius: min over pair distances/2 and 1 - norm.
    d = np.linalg.norm(p, axis=1)
    min_to_border = 1.0 - d
    # Pairwise min distance / 2
    # For N=300, full matrix is 90k entries, fine.
    diff = p[:, None, :] - p[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min() / 2.0
    return min(min_to_border.min(), min_pair)


def _soft_min(a, k=10.0):
    # Soft-min of array a using log-sum-exp trick for stability.
    m = a.min()
    return m - np.log(np.sum(np.exp(-k * (a - m)))) / k


def _soft_radius(p, k=10.0):
    d = np.linalg.norm(p, axis=1)
    border = 1.0 - d
    diff = p[:, None, :] - p[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    pair = dist / 2.0
    # We want min of all constraints: border values and pair distances.
    # Use soft-min over concatenated array.
    all_vals = np.concatenate([border.ravel(), pair.ravel()])
    return _soft_min(all_vals, k=k)


def _soft_radius_grad(p, k=10.0):
    d = np.linalg.norm(p, axis=1)
    border = 1.0 - d
    diff = p[:, None, :] - p[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    pair = dist / 2.0

    # Soft-min gradient: g = -k * exp(-k*(a-m)) / sum(exp(-k*(a-m)))
    all_vals = np.concatenate([border.ravel(), pair.ravel()])
    m = all_vals.min()
    e = np.exp(-k * (all_vals - m))
    s = e.sum()
    g = -k * e / s

    grad = np.zeros_like(p)
    # Border part: border_i = 1 - ||p_i||, d(border_i)/dp_i = -p_i / ||p_i||
    gb = g[:N]
    for i in range(N):
        if d[i] > 1e-12:
            grad[i] += gb[i] * (-p[i] / d[i])

    # Pair part: pair_ij = ||p_i - p_j|| / 2
    gp = g[N:].reshape(N, N)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            # d(||p_i - p_j||)/dp_i = (p_i - p_j)/||p_i - p_j||
            if dist[i, j] > 1e-12:
                grad[i] += gp[i, j] * 0.5 * (p[i] - p[j]) / dist[i, j]
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Initial radius estimate
    r0 = 0.05
    p = _hex_init(N, r0)
    if p is None:
        # Fallback: Fermat spiral
        i = np.arange(N)
        theta = i * 2.0 * np.pi * (1.0 + np.sqrt(5.0)) / 2.0
        r = np.sqrt((i + 0.5) / N)
        p = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    best_p = p.copy()
    best_r = _radius(p)

    # Gradient ascent with adaptive step size
    k_soft = 10.0
    lr = 0.001
    it = 0
    while time.time() < deadline:
        it += 1
        # Compute gradient of soft radius
        grad = _soft_radius_grad(p, k_soft)
        # Normalize gradient to avoid huge steps
        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-12:
            break
        grad = grad / gnorm

        # Line search: try a few step sizes
        best_step = None
        best_val = _soft_radius(p, k_soft)
        for alpha in [lr, lr * 0.5, lr * 0.25, lr * 2.0]:
            p_new = p + alpha * grad
            # Quick feasibility check: ensure points stay in disk
            if np.max(np.linalg.norm(p_new, axis=1)) > 1.0:
                continue
            val_new = _soft_radius(p_new, k_soft)
            if val_new > best_val:
                best_val = val_new
                best_step = p_new

        if best_step is None:
            # No improvement, reduce learning rate
            lr *= 0.5
            if lr < 1e-6:
                break
        else:
            p = best_step
            # Increase learning rate slightly if stable
            if it % 10 == 0:
                lr *= 1.1

        # Update best
        r_curr = _radius(p)
        if r_curr > best_r:
            best_r = r_curr
            best_p = p.copy()

        # If we hit a hard constraint violation, project back
        d = np.linalg.norm(p, axis=1)
        if np.any(d > 1.0):
            p = p / d[:, None]

    # Final validation and return
    # Ensure all points are strictly inside
    d = np.linalg.norm(best_p, axis=1)
    mask = d < 1.0
    if not np.all(mask):
        # Fallback to initial hex if invalid
        return p

    return best_p
# EVOLVE-BLOCK-END
