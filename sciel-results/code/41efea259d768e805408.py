"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 130.0  # safety margin before 150s kill


def _hex_init(n, scale=1.0):
    """Place n circles on a hexagonal lattice scaled to fit in unit square."""
    # Hexagonal lattice: rows offset by half spacing
    # spacing s: horizontal distance between centers in same row
    # vertical distance between rows: s * sqrt(3)/2
    s = scale
    pts = []
    # Generate a grid of hex points, then take first n
    # We want points roughly in [0,1]^2
    # Number of columns and rows to cover the square
    cols = int(np.ceil(1.0 / s)) + 2
    rows = int(np.ceil(1.0 / (s * np.sqrt(3) / 2))) + 2
    for j in range(rows):
        y = j * s * np.sqrt(3) / 2
        offset = 0 if j % 2 == 0 else s / 2
        for i in range(cols):
            x = i * s + offset
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append((x, y))
    pts = np.array(pts)
    # If we don't have enough, add more by expanding or just use what we have
    if len(pts) < n:
        # Fallback: add random points
        extra = np.random.rand(n - len(pts), 2) * 0.8 + 0.1
        pts = np.vstack([pts, extra])
    # Take first n
    return pts[:n]


def _objective_and_grad(z, n):
    """
    z: (3n,) array = [x0, y0, r0, x1, y1, r1, ...]
    Returns (penalty, gradient)
    Penalty = -sum(r_i) + lambda * (sum of overlap violations + boundary violations)
    """
    lam = 50.0
    pos = z.reshape(n, 3)
    x = pos[:, 0]
    y = pos[:, 1]
    r = pos[:, 2]

    # Sum of radii (we maximize, so minimize negative)
    sum_r = np.sum(r)

    # Overlap penalty: for each pair, if dist < r_i + r_j, penalty
    # Use smooth approximation
    overlap = 0.0
    grad_x = np.zeros(n)
    grad_y = np.zeros(n)
    grad_r = np.zeros(n)

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                dist = 1e-12
                dx = 1e-8
                dy = 0.0
            s = r[i] + r[j]
            if dist < s:
                # Penalty: (s - dist)^2 / 2
                pen = 0.5 * (s - dist) ** 2
                overlap += pen
                # Gradient w.r.t. positions
                # d/dx_i of (s - dist)^2/2 = -(s - dist) * (x_i - x_j)/dist
                coef = -(s - dist) / dist
                grad_x[i] += coef * dx
                grad_y[i] += coef * dy
                grad_x[j] -= coef * dx
                grad_y[j] -= coef * dy
                # Gradient w.r.t. radii
                grad_r[i] += -(s - dist)
                grad_r[j] += -(s - dist)

    # Boundary penalty: circle must be inside [0,1]^2
    # r_i <= x_i, r_i <= 1 - x_i, r_i <= y_i, r_i <= 1 - y_i
    boundary = 0.0
    bgrad_x = np.zeros(n)
    bgrad_y = np.zeros(n)
    bgrad_r = np.zeros(n)

    for i in range(n):
        # x boundaries
        for val, sign in [(x[i], 1.0), (1.0 - x[i], -1.0)]:
            if r[i] > val:
                pen = 0.5 * (r[i] - val) ** 2
                boundary += pen
                bgrad_r[i] += -(r[i] - val)
                bgrad_x[i] += sign * (r[i] - val)
        # y boundaries
        for val, sign in [(y[i], 1.0), (1.0 - y[i], -1.0)]:
            if r[i] > val:
                pen = 0.5 * (r[i] - val) ** 2
                boundary += pen
                bgrad_r[i] += -(r[i] - val)
                bgrad_y[i] += sign * (r[i] - val)

    total_penalty = -sum_r + lam * (overlap + boundary)

    # Gradient
    g = np.zeros(3 * n)
    for i in range(n):
        g[3 * i] = -1.0 + lam * (grad_x[i] + bgrad_x[i])
        g[3 * i + 1] = -1.0 + lam * (grad_y[i] + bgrad_y[i])
        g[3 * i + 2] = -1.0 + lam * (grad_r[i] + bgrad_r[i])

    return total_penalty, g


def _validate(z, n):
    """Check if configuration is valid: all circles inside unit square, no overlaps."""
    pos = z.reshape(n, 3)
    x = pos[:, 0]
    y = pos[:, 1]
    r = pos[:, 2]

    # Boundary check
    if np.any(r < 0):
        return False
    if np.any(x < r) or np.any(x > 1 - r) or np.any(y < r) or np.any(y > 1 - r):
        return False

    # Overlap check
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return False
    return True


def _sum_radii(z, n):
    return np.sum(z.reshape(n, 3)[:, 2])


def construct():
    t_start = time.time()
    best_z = None
    best_score = -np.inf

    # Try multiple initializations with different scales
    for scale in [0.2, 0.15, 0.12, 0.1, 0.08]:
        if time.time() - t_start > DEADLINE - 30:
            break

        pts = _hex_init(N, scale)
        # Initialize radii small
        r_init = np.full(N, 0.01)
        z0 = np.zeros(3 * N)
        z0[0::3] = pts[:, 0]
        z0[1::3] = pts[:, 1]
        z0[2::3] = r_init

        # Bounds
        lb = np.zeros(3 * N)
        ub = np.zeros(3 * N)
        for i in range(N):
            lb[3 * i] = 0.0
            ub[3 * i] = 1.0
            lb[3 * i + 1] = 0.0
            ub[3 * i + 1] = 1.0
            lb[3 * i + 2] = 1e-6
            ub[3 * i + 2] = 0.5

        try:
            res = minimize(
                _objective_and_grad,
                z0,
                method='L-BFGS-B',
                jac=True,
                bounds=list(zip(lb, ub)),
                options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-10},
            )
        except Exception:
            continue

        z = res.x
        # Validate
        if _validate(z, N):
            score = _sum_radii(z, N)
            if score > best_score:
                best_score = score
                best_z = z.copy()

        # Also try the result even if not perfectly valid, but clip
        # Clip to valid region
        pos = z.reshape(N, 3).copy()
        pos[:, 2] = np.maximum(pos[:, 2], 1e-6)
        for i in range(N):
            pos[i, 0] = np.clip(pos[i, 0], pos[i, 2], 1.0 - pos[i, 2])
            pos[i, 1] = np.clip(pos[i, 1], pos[i, 2], 1.0 - pos[i, 2])
        z_clipped = pos.flatten()
        if _validate(z_clipped, N):
            score = _sum_radii(z_clipped, N)
            if score > best_score:
                best_score = score
                best_z = z_clipped.copy()

    # If we have a valid result, return it
    if best_z is not None:
        return best_z.reshape(N, 3)

    # Fallback: grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
