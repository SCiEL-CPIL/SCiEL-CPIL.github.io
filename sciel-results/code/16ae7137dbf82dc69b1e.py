"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii.

Combines a force-based physics optimizer with diverse restarts and a final L-BFGS-B
polish pass for refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_START = time.time()
DEADLINE = _START + 116.0  # safety margin before 150s limit
RNG = np.random.default_rng(20240517)

# --- force parameters ---
K_WALL = 500.0
K_SEP = 40.0
K_R = 10.0
EPS = 1e-9
STEP = 0.03
R_MIN = 1e-6
R_MAX = 0.5


def _is_valid(c, tol=1e-6):
    """Return True if packing is inside [0,1]^2 and circles are disjoint."""
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    if r.min() < R_MIN:
        return False
    if (x.min() < -tol) or (x.max() > 1 + tol) or (y.min() < -tol) or (y.max() > 1 + tol):
        return False
    if np.any(x + r > 1 + tol) or np.any(x - r < -tol):
        return False
    if np.any(y + r > 1 + tol) or np.any(y - r < -tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - tol:
                return False
    return True


def _score(c):
    return float(np.sum(c[:, 2])) if _is_valid(c) else -np.inf


def _clamp(c):
    """Clamp positions/radii to a feasible box (keeps circles inside square)."""
    x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
    r = np.clip(r, R_MIN, R_MAX)
    x = np.clip(x, r, 1.0 - r)
    y = np.clip(y, r, 1.0 - r)
    return np.column_stack([x, y, r])


def _forces(c):
    """Compute forces on each circle: wall repulsion, separation, radius growth."""
    x = c[:, 0].copy(); y = c[:, 1].copy(); r = c[:, 2].copy()
    fx = np.zeros(N); fy = np.zeros(N); fr = np.zeros(N)

    # wall forces: push away from walls
    fx += K_WALL * np.where(x < r, 1.0, 0.0) - K_WALL * np.where(x > 1 - r, 1.0, 0.0)
    fy += K_WALL * np.where(y < r, 1.0, 0.0) - K_WALL * np.where(y > 1 - r, 1.0, 0.0)
    fr += K_WALL * (np.where(x < r, 1.0, 0.0) + np.where(x > 1 - r, 1.0, 0.0)
                    + np.where(y < r, 1.0, 0.0) + np.where(y > 1 - r, 1.0, 0.0))

    # separation forces: push overlapping circles apart
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]; dy = y[i] - y[j]
            d = np.hypot(dx, dy)
            s = r[i] + r[j]
            if d < s:
                if d < 1e-9:
                    dx = RNG.standard_normal(); dy = RNG.standard_normal()
                    d = np.hypot(dx, dy)
                ux = dx / d; uy = dy / d
                f = K_SEP * (s - d)
                fx[i] += f * ux; fy[i] += f * uy
                fx[j] -= f * ux; fy[j] -= f * uy
                fr[i] += f; fr[j] += f

    # maximize radii
    fr += K_R
    return fx, fy, fr


def _pbd_step(c):
    """One force-based step with adaptive step size and clamping."""
    c = _clamp(c)
    fx, fy, fr = _forces(c)

    # adaptive step: smaller when forces are large
    fmag = np.hypot(fx, fy)
    step = STEP * 0.5 / (1.0 + 0.02 * fmag.max())

    x = c[:, 0] + step * fx
    y = c[:, 1] + step * fy
    r = c[:, 2] + step * fr

    return np.column_stack([x, y, r])


def _optimize(c0):
    """Force-based optimizer with adaptive step size."""
    c = c0.copy()
    best = _score(c)
    best_c = c.copy()
    step = STEP
    for it in range(4000):
        c = _pbd_step(c)
        val = _score(c)
        if val > best:
            best = val
            best_c = c.copy()
            step = min(step * 1.12, 0.12)
        else:
            step *= 0.82
            if it > 5:
                c = best_c.copy()
            if step < 1e-4:
                break
        if time.time() > DEADLINE - 1.0:
            break
    return best_c, best


def _seed_grid():
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    return np.array(cells[:N])


def _seed_random():
    """Random initialization with small radii."""
    x = RNG.uniform(0.08, 0.92, N)
    y = RNG.uniform(0.08, 0.92, N)
    r = RNG.uniform(0.02, 0.08, N)
    return np.column_stack([x, y, r])


def _seed_symmetric():
    """Symmetric corner initialization."""
    # Place circles near corners and center
    positions = []
    radii = []
    # Corners
    for cx, cy in [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]:
        positions.append((cx, cy))
        radii.append(0.12)
    # Center
    positions.append((0.5, 0.5))
    radii.append(0.15)
    # Fill remaining with random
    while len(positions) < N:
        x = RNG.uniform(0.1, 0.9)
        y = RNG.uniform(0.1, 0.9)
        r = RNG.uniform(0.05, 0.1)
        positions.append((x, y))
        radii.append(r)
    
    x = np.array([p[0] for p in positions[:N]])
    y = np.array([p[1] for p in positions[:N]])
    r = np.array(radii[:N])
    return np.column_stack([x, y, r])


def _lbfgs_polish(c0):
    """Final L-BFGS-B polish pass on the best packing."""
    if time.time() > DEADLINE - 5.0:
        return c0, _score(c0)
    
    # Convert to optimization parameters
    x0 = c0.flatten()
    
    def _objective(params):
        c = params.reshape(-1, 3)
        x = c[:, 0]; y = c[:, 1]; r = c[:, 2]
        
        # Penalty for invalid configurations
        penalty = 0.0
        # Wall violations
        penalty += np.sum(np.maximum(0, r - x) ** 2)
        penalty += np.sum(np.maximum(0, x + r - 1) ** 2)
        penalty += np.sum(np.maximum(0, r - y) ** 2)
        penalty += np.sum(np.maximum(0, y + r - 1) ** 2)
        
        # Overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]; dy = y[i] - y[j]
                d = np.hypot(dx, dy)
                s = r[i] + r[j]
                if d < s:
                    penalty += (s - d) ** 2
        
        # Negative sum of radii (minimize)
        return -np.sum(r) + 10.0 * penalty
    
    try:
        result = minimize(
            _objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        c_polished = result.x.reshape(-1, 3)
        c_polished = _clamp(c_polished)
        
        # Only accept if valid
        if _is_valid(c_polished, tol=1e-5):
            score = _score(c_polished)
            if score > _score(c0):
                return c_polished, score
    except Exception:
        pass
    
    return c0, _score(c0)


def construct():
    t0 = time.time()
    
    # Start with grid initialization
    c_grid = _seed_grid()
    best_c, best_val = _optimize(c_grid)
    
    # Try diverse random starts
    n_random = 30
    for i in range(n_random):
        if time.time() > DEADLINE - 10.0:
            break
        
        if i % 10 == 0:
            c0 = _seed_symmetric()
        else:
            c0 = _seed_random()
        
        cc, val = _optimize(c0)
        if val > best_val:
            best_val, best_c = val, cc
    
    # Final polish on the best packing
    if time.time() < DEADLINE - 5.0:
        cc, val = _lbfgs_polish(best_c)
        if val > best_val:
            best_val, best_c = val, cc
    
    # Fallback: seed grid is always valid
    if best_val <= 0:
        return c_grid
    
    return best_c
