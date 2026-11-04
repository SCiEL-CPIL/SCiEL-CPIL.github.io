"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint

N = 26
DEADLINE = time.time() + 140  # safety margin before 150s kill


def _is_valid(circles, tol=1e-9):
    """Check that all circles are inside [0,1]^2 and disjoint."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol) or np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    # Check pairwise non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _objective(params, fixed_radii=None):
    """
    Objective: minimize -sum(radii) with penalty for constraint violations.
    params layout: [x1..xN, y1..yN, r1..rN]
    """
    x = params[:N]
    y = params[N:2 * N]
    r = params[2 * N:3 * N]

    if fixed_radii is not None:
        # If radii are fixed, just return 0 (positions only)
        pass
    else:
        # Penalize going out of bounds
        penalty = 0.0
        for i in range(N):
            if x[i] - r[i] < 0:
                penalty += (x[i] - r[i]) ** 2
            if x[i] + r[i] > 1:
                penalty += (x[i] + r[i] - 1) ** 2
            if y[i] - r[i] < 0:
                penalty += (y[i] - r[i]) ** 2
            if y[i] + r[i] > 1:
                penalty += (y[i] + r[i] - 1) ** 2
        # Penalize overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    overlap = min_dist - dist
                    penalty += overlap * overlap
        return -np.sum(r) + 1000.0 * penalty
    return 0.0


def _objective_pos(params, radii):
    """Minimize penalty for positions given fixed radii."""
    x = params[:N]
    y = params[N:2 * N]
    penalty = 0.0
    for i in range(N):
        if x[i] - radii[i] < 0:
            penalty += (x[i] - radii[i]) ** 2
        if x[i] + radii[i] > 1:
            penalty += (x[i] + radii[i] - 1) ** 2
        if y[i] - radii[i] < 0:
            penalty += (y[i] - radii[i]) ** 2
        if y[i] + radii[i] > 1:
            penalty += (y[i] + radii[i] - 1) ** 2
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(dist_sq)
                overlap = min_dist - dist
                penalty += overlap * overlap
    return penalty


def _random_init(seed):
    rng = np.random.RandomState(seed)
    # Start with a grid-like perturbation
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side * 0.95
            cells.append([cx, cy, r])
    cells = cells[:N]
    # Add small random perturbation
    for c in cells:
        c[0] += rng.uniform(-0.02, 0.02)
        c[1] += rng.uniform(-0.02, 0.02)
    return np.array(cells)


def _random_init2(seed):
    """Another random initialization strategy."""
    rng = np.random.RandomState(seed)
    # Try to place circles greedily
    circles = []
    attempts = 0
    while len(circles) < N and attempts < 10000:
        attempts += 1
        cx = rng.uniform(0.05, 0.95)
        cy = rng.uniform(0.05, 0.95)
        # Max possible radius given boundaries
        max_r = min(cx, 1 - cx, cy, 1 - cy)
        # Also check against existing circles
        r = max_r * rng.uniform(0.3, 0.9)
        if r < 0.01:
            continue
        valid = True
        for (ox, oy, orad) in circles:
            dist = np.sqrt((cx - ox) ** 2 + (cy - oy) ** 2)
            if dist < r + orad:
                valid = False
                break
        if valid:
            circles.append([cx, cy, r])
    if len(circles) < N:
        # Fill remaining with tiny circles
        for _ in range(N - len(circles)):
            cx = rng.uniform(0.1, 0.9)
            cy = rng.uniform(0.1, 0.9)
            circles.append([cx, cy, 0.01])
    return np.array(circles[:N])


def _optimize_single(circles, method='L-BFGS-B', maxiter=500):
    """Optimize a single configuration."""
    params = circles.flatten().copy()
    
    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0, 1))  # x
    for i in range(N):
        bounds.append((0, 1))  # y
    for i in range(N):
        bounds.append((1e-6, 0.5))  # r
    
    try:
        result = minimize(
            _objective,
            params,
            method=method,
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.success or result.fun < 0:
            opt_params = result.x
            opt_circles = opt_params.reshape(N, 3)
            # Clamp to valid range
            opt_circles[:, 0] = np.clip(opt_circles[:, 0], 0, 1)
            opt_circles[:, 1] = np.clip(opt_circles[:, 1], 0, 1)
            opt_circles[:, 2] = np.clip(opt_circles[:, 2], 1e-6, 0.5)
            return opt_circles
    except Exception:
        pass
    return circles


def _polish(circles, maxiter=2000):
    """High-precision polish using L-BFGS-B with tight tolerances."""
    params = circles.flatten().copy()
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
    for i in range(N):
        bounds.append((0, 1))
    for i in range(N):
        bounds.append((1e-8, 0.5))
    
    try:
        result = minimize(
            _objective,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        if result.fun < 0:
            opt_params = result.x
            opt_circles = opt_params.reshape(N, 3)
            opt_circles[:, 0] = np.clip(opt_circles[:, 0], 0, 1)
            opt_circles[:, 1] = np.clip(opt_circles[:, 1], 0, 1)
            opt_circles[:, 2] = np.clip(opt_circles[:, 2], 1e-8, 0.5)
            return opt_circles
    except Exception:
        pass
    return circles


def _simulate_annealing(circles, max_iter=2000, T_start=0.05, T_end=0.001):
    """Simple simulated annealing on positions and radii."""
    rng = np.random.RandomState(42)
    current = circles.copy()
    best = circles.copy()
    best_score = np.sum(best[:, 2])
    
    for it in range(max_iter):
        T = T_start * (T_end / T_start) ** (it / max_iter)
        # Perturb
        candidate = current.copy()
        idx = rng.randint(N)
        # Random direction
        delta = rng.normal(0, T, size=2)
        candidate[idx, 0] += delta[0]
        candidate[idx, 1] += delta[1]
        # Maybe adjust radius
        if rng.random() < 0.3:
            candidate[idx, 2] *= (1 + rng.normal(0, T * 0.5))
        candidate[idx, 2] = np.clip(candidate[idx, 2], 1e-6, 0.5)
        candidate[idx, 0] = np.clip(candidate[idx, 0], 0, 1)
        candidate[idx, 1] = np.clip(candidate[idx, 1], 0, 1)
        
        # Evaluate penalty
        cur_pen = _objective(current.flatten())
        cand_pen = _objective(candidate.flatten())
        delta_e = cand_pen - cur_pen
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current = candidate
            if np.sum(current[:, 2]) > best_score and _is_valid(current, tol=1e-7):
                best = current.copy()
                best_score = np.sum(best[:, 2])
    
    return best


def construct():
    start_time = time.time()
    best_circles = None
    best_score = -np.inf
    
    # Multi-start optimization
    num_starts = 12
    for s in range(num_starts):
        if time.time() > DEADLINE:
            break
        
        # Different initialization strategies
        if s % 3 == 0:
            circles = _random_init(s * 100 + 1)
        elif s % 3 == 1:
            circles = _random_init2(s * 200 + 7)
        else:
            # Grid with larger perturbation
            side = int(np.ceil(np.sqrt(N)))
            rng = np.random.RandomState(s * 300 + 13)
            cells = []
            for i in range(side):
                for j in range(side):
                    cx = (i + 0.5) / side + rng.uniform(-0.05, 0.05)
                    cy = (j + 0.5) / side + rng.uniform(-0.05, 0.05)
                    r = 0.5 / side * 0.9
                    cells.append([cx, cy, r])
            circles = np.array(cells[:N])
        
        # Optimize
        opt_circles = _optimize_single(circles, maxiter=300)
        
        # Check validity
        if _is_valid(opt_circles, tol=1e-6):
            score = np.sum(opt_circles[:, 2])
            if score > best_score:
                best_score = score
                best_circles = opt_circles.copy()
        
        # Simulated annealing refinement
        if time.time() < DEADLINE - 10:
            sa_circles = _simulate_annealing(opt_circles, max_iter=1000)
            if _is_valid(sa_circles, tol=1e-6):
                score = np.sum(sa_circles[:, 2])
                if score > best_score:
                    best_score = score
                    best_circles = sa_circles.copy()
    
    # Final polish on the best result
    if best_circles is not None and time.time() < DEADLINE - 5:
        polished = _polish(best_circles, maxiter=1000)
        if _is_valid(polished, tol=1e-6):
            score = np.sum(polished[:, 2])
            if score > best_score:
                best_score = score
                best_circles = polished
    
    # Fallback to seed if nothing valid found
    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:N])
    
    return best_circles
