"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
MAX_TIME = 110.0  # safety margin before 120s limit


def _is_valid(circles, tol=1e-9):
    """Check if circles are disjoint and inside unit square."""
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if x < -tol or x > 1 + tol or y < -tol or y > 1 + tol or r < -tol:
            return False
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = circles[i]
            xj, yj, rj = circles[j]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if dist < ri + rj - tol:
                return False
    return True


def _sum_radii(circles):
    return sum(c[2] for c in circles)


def _objective_and_grad(params, n):
    """
    params: array of shape (3n,) containing (x1,y1,r1, x2,y2,r2, ..., xn,yn,rn)
    Returns: negative sum of radii (to minimize), and gradient.
    We use a penalty for constraint violations to make it differentiable.
    """
    params = np.asarray(params)
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    # Objective: maximize sum of radii -> minimize -sum(r)
    obj = -np.sum(r)

    # Penalty for constraints
    penalty = 0.0
    grad_penalty = np.zeros_like(params)

    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r, r > 0
    # Use smooth approximation: penalty = sum of max(0, violation)^2
    # For x >= r: violation = r - x
    # For x <= 1-r: violation = x + r - 1
    # For y >= r: violation = r - y
    # For y <= 1-r: violation = y + r - 1
    # For r > 0: violation = -r

    # We'll use a large penalty coefficient
    lam = 1000.0

    # Boundary penalties
    viol_x_low = np.maximum(0, r - x)
    viol_x_high = np.maximum(0, x + r - 1)
    viol_y_low = np.maximum(0, r - y)
    viol_y_high = np.maximum(0, y + r - 1)
    viol_r = np.maximum(0, -r)

    penalty += lam * (np.sum(viol_x_low**2) + np.sum(viol_x_high**2) +
                      np.sum(viol_y_low**2) + np.sum(viol_y_high**2) +
                      np.sum(viol_r**2))

    # Pairwise disjointness: dist(i,j) >= ri + rj
    # violation = ri + rj - dist(i,j)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
            viol = r[i] + r[j] - dist
            if viol > 0:
                penalty += lam * viol**2
                # Gradient w.r.t. params
                # d(penalty)/d(x_i) = 2*lam*viol * (-dx/dist)
                # d(penalty)/d(y_i) = 2*lam*viol * (-dy/dist)
                # d(penalty)/d(r_i) = 2*lam*viol * (1 + dx*dx/(dist^2)) ... wait
                # Actually: viol = ri + rj - sqrt((xi-xj)^2 + (yi-yj)^2)
                # d(viol)/d(xi) = -(xi-xj)/dist = -dx/dist
                # d(viol)/d(xj) = (xi-xj)/dist = dx/dist
                # d(viol)/d(ri) = 1
                # d(viol)/d(rj) = 1
                # d(penalty)/d(param) = 2*lam*viol * d(viol)/d(param)
                
                coeff = 2 * lam * viol
                idx_i = i * 3
                idx_j = j * 3
                grad_penalty[idx_i] += coeff * (-dx / dist)
                grad_penalty[idx_i + 1] += coeff * (-dy / dist)
                grad_penalty[idx_i + 2] += coeff * 1
                grad_penalty[idx_j] += coeff * (dx / dist)
                grad_penalty[idx_j + 1] += coeff * (dy / dist)
                grad_penalty[idx_j + 2] += coeff * 1

    # Boundary gradients
    # For viol_x_low = max(0, r-x): if viol > 0, d/dx = -1, d/dr = 1
    mask = viol_x_low > 0
    if np.any(mask):
        coeff = 2 * lam * viol_x_low[mask]
        idx = np.where(mask)[0]
        grad_penalty[idx * 3] += -coeff  # d/dx
        grad_penalty[idx * 3 + 2] += coeff  # d/dr

    mask = viol_x_high > 0
    if np.any(mask):
        coeff = 2 * lam * viol_x_high[mask]
        idx = np.where(mask)[0]
        grad_penalty[idx * 3] += coeff  # d/dx
        grad_penalty[idx * 3 + 2] += coeff  # d/dr

    mask = viol_y_low > 0
    if np.any(mask):
        coeff = 2 * lam * viol_y_low[mask]
        idx = np.where(mask)[0]
        grad_penalty[idx * 3 + 1] += -coeff  # d/dy
        grad_penalty[idx * 3 + 2] += coeff  # d/dr

    mask = viol_y_high > 0
    if np.any(mask):
        coeff = 2 * lam * viol_y_high[mask]
        idx = np.where(mask)[0]
        grad_penalty[idx * 3 + 1] += coeff  # d/dy
        grad_penalty[idx * 3 + 2] += coeff  # d/dr

    mask = viol_r > 0
    if np.any(mask):
        coeff = 2 * lam * viol_r[mask]
        idx = np.where(mask)[0]
        grad_penalty[idx * 3 + 2] += -coeff  # d/dr

    # Total objective and gradient
    total_obj = obj + penalty
    grad_obj = -np.ones_like(params)  # d(-sum(r))/d(params) = -1 for r components
    # Actually, d(-sum(r))/d(x) = 0, d(-sum(r))/d(y) = 0, d(-sum(r))/d(r) = -1
    grad_obj = np.zeros_like(params)
    for i in range(n):
        grad_obj[i * 3 + 2] = -1.0

    total_grad = grad_obj + grad_penalty

    return total_obj, total_grad


def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME

    # Initial configuration: grid-based
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append([(i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3])
    initial_circles = np.array(cells[:N])

    best_circles = initial_circles.copy()
    best_score = _sum_radii(best_circles)

    # Try to improve with gradient-based optimization
    # Use multiple restarts with different initializations if time permits

    def _optimize_from(init_circles, max_iter=500):
        """Run gradient-based optimization from a given initial configuration."""
        nonlocal deadline, best_circles, best_score

        params = init_circles.flatten()
        n = N

        # Use L-BFGS-B with bounds
        bounds = []
        for i in range(n):
            bounds.append((0, 1))  # x
            bounds.append((0, 1))  # y
            bounds.append((1e-6, 0.5))  # r (max radius is 0.5 for a single circle in unit square)

        # Define objective and gradient for scipy
        def obj_func(p):
            return _objective_and_grad(p, n)[0]

        def grad_func(p):
            return _objective_and_grad(p, n)[1]

        try:
            result = minimize(
                obj_func,
                params,
                jac=grad_func,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'gtol': 1e-6, 'ftol': 1e-12}
            )
        except Exception:
            return None

        if result.success or result.fun < obj_func(params):
            new_params = result.x
            new_circles = new_params.reshape(n, 3)
            # Clamp to valid range
            new_circles[:, 0] = np.clip(new_circles[:, 0], 0, 1)
            new_circles[:, 1] = np.clip(new_circles[:, 1], 0, 1)
            new_circles[:, 2] = np.clip(new_circles[:, 2], 1e-6, 0.5)

            # Check validity
            if _is_valid(new_circles, tol=1e-6):
                score = _sum_radii(new_circles)
                if score > best_score:
                    best_circles = new_circles.copy()
                    best_score = score
                return new_circles
            else:
                # Try to fix by reducing radii slightly
                for attempt in range(10):
                    new_circles[:, 2] *= 0.95
                    if _is_valid(new_circles, tol=1e-6):
                        score = _sum_radii(new_circles)
                        if score > best_score:
                            best_circles = new_circles.copy()
                            best_score = score
                        return new_circles
                return None
        return None

    # Run optimization from initial grid
    _optimize_from(initial_circles, max_iter=300)

    # Check time
    if time.time() > deadline - 5:
        return best_circles

    # Try a few more restarts with perturbed initializations
    rng = np.random.RandomState(42)
    for restart in range(5):
        if time.time() > deadline - 5:
            break

        # Perturb the best solution
        perturbed = best_circles.copy()
        perturbed[:, 0] += rng.uniform(-0.05, 0.05, N)
        perturbed[:, 1] += rng.uniform(-0.05, 0.05, N)
        perturbed[:, 2] *= rng.uniform(0.8, 1.0, N)
        perturbed[:, 0] = np.clip(perturbed[:, 0], 0, 1)
        perturbed[:, 1] = np.clip(perturbed[:, 1], 0, 1)
        perturbed[:, 2] = np.clip(perturbed[:, 2], 1e-6, 0.5)

        _optimize_from(perturbed, max_iter=200)

    # Final polish: try to increase radii slightly
    if time.time() < deadline - 2:
        for _ in range(3):
            if time.time() > deadline - 1:
                break
            # Try increasing all radii by a small amount
            trial = best_circles.copy()
            trial[:, 2] *= 1.001
            if _is_valid(trial, tol=1e-8):
                best_circles = trial
            else:
                break

    return best_circles
