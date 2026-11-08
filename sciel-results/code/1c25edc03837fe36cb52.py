"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import time
import numpy as np

N = 26
DEADLINE = 150.0
SAFETY_MARGIN = 8.0


def _make_loss(pos, rad, lam_ov, lam_bd):
    """Differentiable loss = -sum(rad) + lam_ov * sum(max(0, r_i+r_j-d_ij)^2)
                              + lam_bd * sum(boundary violations)^2.
    Returns (loss, grad_pos, grad_rad)."""
    n = len(rad)
    dx = pos[:, 0, None] - pos[None, :, 0]
    dy = pos[:, 1, None] - pos[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy + 1e-12)
    np.fill_diagonal(dist, 0.0)
    overlap = np.maximum(rad[:, None] + rad[None, :] - dist, 0.0)
    # gradient of overlap^2 wrt pos
    # d(ov^2)/dpos_i = 2*ov * (-d_ij/dpos_i) = 2*ov * (pos_i - pos_j)/d_ij for j
    gpos = np.zeros_like(pos)
    grad_ov = 2.0 * overlap  # (n,n)
    # divide by dist for direction
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_d = 1.0 / dist
        np.fill_diagonal(inv_d, 0.0)
    dir_x = dx * inv_d
    dir_y = dy * inv_d
    gpos[:, 0] = -np.sum(grad_ov * dir_x, axis=1)
    gpos[:, 1] = -np.sum(grad_ov * dir_y, axis=1)
    # wrt rad: d(ov^2)/drad_i = 2*ov * (1 for each j) summed over j
    grad_rad_ov = 2.0 * np.sum(grad_ov, axis=1)

    # boundary: x in [r, 1-r], y in [r, 1-r]
    viol_x_lo = np.maximum(rad - pos[:, 0], 0.0)
    viol_x_hi = np.maximum(pos[:, 0] + rad - 1.0, 0.0)
    viol_y_lo = np.maximum(rad - pos[:, 1], 0.0)
    viol_y_hi = np.maximum(pos[:, 1] + rad - 1.0, 0.0)
    bd_pen = np.sum(viol_x_lo**2 + viol_x_hi**2 + viol_y_lo**2 + viol_y_hi**2)
    gpos[:, 0] += 2.0 * (viol_x_lo - viol_x_hi)
    gpos[:, 1] += 2.0 * (viol_y_lo - viol_y_hi)
    grad_rad_bd = 2.0 * (viol_x_lo + viol_x_hi + viol_y_lo + viol_y_hi)

    loss = -np.sum(rad) + lam_ov * np.sum(overlap**2) + lam_bd * bd_pen
    gpos *= lam_ov if False else 1.0  # already scaled? recompute scaling
    # Actually apply lam to penalty grads
    gpos = -np.zeros_like(pos)  # reset, recompute cleanly
    gpos[:, 0] = -np.sum(grad_ov * dir_x, axis=1)
    gpos[:, 1] = -np.sum(grad_ov * dir_y, axis=1)
    gpos *= lam_ov
    gpos[:, 0] += 2.0 * (viol_x_lo - viol_x_hi) * lam_bd
    gpos[:, 1] += 2.0 * (viol_y_lo - viol_y_hi) * lam_bd
    grad_rad = -np.ones(n) + lam_ov * grad_rad_ov + lam_bd * grad_rad_bd
    return loss, gpos, grad_rad


def _grid_init():
    side = int(np.ceil(np.sqrt(N)))
    pos = []
    rad = []
    for i in range(side):
        for j in range(side):
            pos.append([ (i + 0.5) / side, (j + 0.5) / side ])
            rad.append(0.5 / side - 1e-3)
    pos = np.array(pos[:N], dtype=float)
    rad = np.array(rad[:N], dtype=float)
    return pos, rad


def _score(pos, rad):
    """Return sum of radii if valid, else -inf."""
    n = len(rad)
    for i in range(n):
        x, y, r = pos[i, 0], pos[i, 1], rad[i]
        if x < r - 1e-9 or y < r - 1e-9 or x + r > 1 + 1e-9 or y + r > 1 + 1e-9:
            return -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < rad[i] + rad[j] - 1e-9:
                return -np.inf
    return float(np.sum(rad))


def _optimize(pos0, rad0, t_end, lam_ov0, lam_bd0, lam_ov_end, lam_bd_end,
              lr, n_iter, rng):
    pos = pos0.copy()
    rad = rad0.copy()
    n = len(rad)
    best_loss = np.inf
    best_pos, best_rad = pos.copy(), rad.copy()
    # anneal lam
    for it in range(n_iter):
        frac = it / max(1, n_iter - 1)
        lam_ov = lam_ov0 * (lam_ov_end / lam_ov0) ** frac
        lam_bd = lam_bd0 * (lam_bd_end / lam_bd0) ** frac
        loss, gpos, grad_rad = _make_loss(pos, rad, lam_ov, lam_bd)
        # SGD step
        pos -= lr * gpos
        rad -= lr * grad_rad
        # clamp radii positive
        rad = np.maximum(rad, 1e-5)
        # small jitter to escape plateaus
        if it % 50 == 0 and it > 0:
            pos += rng.normal(0, 0.002, size=pos.shape)
            rad += rng.normal(0, 0.0005, size=rad.shape)
            rad = np.maximum(rad, 1e-5)
        if loss < best_loss:
            best_loss = loss
            best_pos, best_rad = pos.copy(), rad.copy()
    return best_pos, best_rad


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE - SAFETY_MARGIN
    rng = np.random.default_rng(42)

    pos0, rad0 = _grid_init()
    best_val = _score(pos0, rad0)
    best_pos, best_rad = pos0.copy(), rad0.copy()

    lam_ov0, lam_ov_end = 50.0, 0.5
    lam_bd0, lam_bd_end = 200.0, 2.0
    lr = 0.01
    n_iter = 600

    restart = 0
    while time.time() < deadline:
        # perturb initial config for diversity
        if restart == 0:
            p, r = pos0.copy(), rad0.copy()
        else:
            p = best_pos + rng.normal(0, 0.05, size=best_pos.shape)
            r = best_rad + rng.normal(0, 0.02, size=best_rad.shape)
            r = np.maximum(r, 1e-5)
            p = np.clip(p, 0, 1)
        p, r = _optimize(p, r, deadline, lam_ov0, lam_bd0, lam_ov_end, lam_bd_end,
                         lr, n_iter, rng)
        # polish: run a few more iters with smaller lam
        p2, r2 = _optimize(p, r, deadline, 5.0, 20.0, 0.3, 1.0, 0.005, 300, rng)
        val = _score(p2, r2)
        if np.isfinite(val) and val > best_val:
            best_val = val
            best_pos, best_rad = p2.copy(), r2.copy()
        restart += 1
        if restart >= 60 and time.time() > deadline - 20:
            break

    return np.column_stack([best_pos, best_rad])


if __name__ == "__main__":
    res = construct()
    print("sum radii:", _score(res[:, 0], res[:, 1]))
