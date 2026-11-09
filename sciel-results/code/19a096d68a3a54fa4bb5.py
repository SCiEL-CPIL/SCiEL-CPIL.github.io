"""Circle packing in unit square: hex lattice init + gradient-based optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s kill

    def hex_lattice(n):
        """Generate n points in a hexagonal arrangement inside [0,1]^2."""
        # Try multiple hexagonal grid configurations
        best = None
        for rows in range(3, 8):
            cols = int(np.ceil(n / rows))
            # Hex grid: offset every other row
            pts = []
            for r in range(rows):
                for c in range(cols):
                    x = (c + (0.5 if r % 2 == 1 else 0)) / (cols + 0.5)
                    y = (r + 0.5) / (rows + 0.5)
                    pts.append((x, y))
            if len(pts) >= n:
                pts = pts[:n]
                # Compute min distance
                min_d = np.inf
                for i in range(len(pts)):
                    for j in range(i+1, len(pts)):
                        d = np.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1])
                        min_d = min(min_d, d)
                if best is None or min_d > best[0]:
                    best = (min_d, pts)
        if best is None:
            # Fallback to grid
            side = int(np.ceil(np.sqrt(n)))
            pts = [((i+0.5)/side, (j+0.5)/side) for i in range(side) for j in range(side)][:n]
        else:
            pts = best[1]
        return np.array(pts)

    def make_obj(pos, rad, lam=50.0):
        """Objective: -sum(radii) + penalty for overlaps and boundary violations."""
        n = len(rad)
        def f(v):
            p = v[:2*n].reshape(n, 2)
            r = v[2*n:]
            # Penalty for overlaps: sum of (r_i + r_j - dist_ij)^2 for dist < r_i+r_j
            penalty = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    d = np.hypot(p[i,0]-p[j,0], p[i,1]-p[j,1])
                    overlap = r[i] + r[j] - d
                    if overlap > 0:
                        penalty += overlap * overlap
            # Boundary penalty: circle must be inside [0,1]^2
            for i in range(n):
                if p[i,0] < r[i]:
                    penalty += (r[i] - p[i,0]) ** 2
                if p[i,0] > 1 - r[i]:
                    penalty += (p[i,0] - (1 - r[i])) ** 2
                if p[i,1] < r[i]:
                    penalty += (r[i] - p[i,1]) ** 2
                if p[i,1] > 1 - r[i]:
                    penalty += (p[i,1] - (1 - r[i])) ** 2
            return -np.sum(r) + lam * penalty
        return f

    def grad_obj(pos, rad, lam=50.0):
        """Gradient of the objective."""
        n = len(rad)
        def g(v):
            p = v[:2*n].reshape(n, 2)
            r = v[2*n:]
            grad = np.zeros_like(v)
            # For each pair, compute gradient of overlap penalty
            for i in range(n):
                for j in range(i+1, n):
                    dx = p[i,0] - p[j,0]
                    dy = p[i,1] - p[j,1]
                    d = np.hypot(dx, dy)
                    overlap = r[i] + r[j] - d
                    if overlap > 0 and d > 1e-10:
                        # d/dp_i of (r_i+r_j-d)^2 = -2*overlap * (d/dp_i d)
                        # d/dp_i d = (p_i - p_j)/d
                        factor = 2 * lam * overlap
                        grad[2*i] += factor * dx / d
                        grad[2*i+1] += factor * dy / d
                        grad[2*j] -= factor * dx / d
                        grad[2*j+1] -= factor * dy / d
                        # d/dr_i of (r_i+r_j-d)^2 = 2*overlap
                        grad[2*n+i] += 2 * lam * overlap
                        grad[2*n+j] += 2 * lam * overlap
            # Boundary gradients
            for i in range(n):
                if p[i,0] < r[i]:
                    grad[2*i] += 2 * lam * (r[i] - p[i,0])
                    grad[2*n+i] -= 2 * lam * (r[i] - p[i,0])
                if p[i,0] > 1 - r[i]:
                    grad[2*i] -= 2 * lam * (p[i,0] - (1 - r[i]))
                    grad[2*n+i] += 2 * lam * (p[i,0] - (1 - r[i]))
                if p[i,1] < r[i]:
                    grad[2*i+1] += 2 * lam * (r[i] - p[i,1])
                    grad[2*n+i] -= 2 * lam * (r[i] - p[i,1])
                if p[i,1] > 1 - r[i]:
                    grad[2*i+1] -= 2 * lam * (p[i,1] - (1 - r[i]))
                    grad[2*n+i] += 2 * lam * (p[i,1] - (1 - r[i]))
            # Gradient of -sum(r)
            grad[2*n:] -= 1.0
            return grad
        return g

    # Initialize from hex lattice
    pts = hex_lattice(N)
    # Initial radii: small, based on local spacing
    min_d = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(pts[i,0]-pts[j,0], pts[i,1]-pts[j,1])
            min_d = min(min_d, d)
    init_r = np.full(N, min_d / 2 * 0.9)

    # Build initial vector
    v0 = np.concatenate([pts.ravel(), init_r])

    # Bounds: positions in [0,1], radii in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
    for i in range(N):
        bounds.append((1e-4, 0.5))  # radius

    best_result = None
    best_score = -np.inf

    # Run L-BFGS-B with increasing lambda for better convergence
    for lam in [5.0, 20.0, 50.0, 100.0]:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                make_obj(pts, init_r, lam),
                v0,
                jac=grad_obj(pts, init_r, lam),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            if res.success or res.fun < best_score:
                best_score = res.fun
                best_result = res.x
                v0 = res.x  # warm start for next lambda
        except Exception:
            pass

    if best_result is None:
        # Fallback: return the hex lattice with small radii
        result = np.zeros((N, 3))
        result[:, 0] = pts[:, 0]
        result[:, 1] = pts[:, 1]
        result[:, 2] = init_r
        return result

    # Extract final positions and radii
    p = best_result[:2*N].reshape(N, 2)
    r = best_result[2*N:]

    # Clamp to ensure validity
    for i in range(N):
        r[i] = max(r[i], 1e-4)
        p[i, 0] = np.clip(p[i, 0], r[i], 1 - r[i])
        p[i, 1] = np.clip(p[i, 1], r[i], 1 - r[i])

    # Final check: resolve any remaining overlaps by shrinking radii
    for iteration in range(100):
        if time.time() > deadline:
            break
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(p[i,0]-p[j,0], p[i,1]-p[j,1])
                if d < r[i] + r[j] - 1e-9:
                    # Shrink the larger radius slightly
                    if r[i] >= r[j]:
                        r[i] = max(1e-4, (d - r[j]) * 0.99)
                    else:
                        r[j] = max(1e-4, (d - r[i]) * 0.99)
                    changed = True
        if not changed:
            break

    # Final boundary clamp
    for i in range(N):
        r[i] = min(r[i], p[i, 0], 1 - p[i, 0], p[i, 1], 1 - p[i, 1])
        r[i] = max(r[i], 1e-4)
        p[i, 0] = np.clip(p[i, 0], r[i], 1 - r[i])
        p[i, 1] = np.clip(p[i, 1], r[i], 1 - r[i])

    result = np.zeros((N, 3))
    result[:, 0] = p[:, 0]
    result[:, 1] = p[:, 1]
    result[:, 2] = r
    return result
# EVOLVE-BLOCK-END
