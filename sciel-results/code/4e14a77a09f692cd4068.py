"""Circle packing in unit square using iterative maximal expansion (hard constraints)."""
import numpy as np
import time

N = 26

def _max_radius_at(x, y, circles, i):
    """Maximum radius for circle i given its position (x,y) and other circles."""
    # Wall constraints
    r = min(x, 1.0 - x, y, 1.0 - y)
    # Neighbor constraints
    for j in range(len(circles)):
        if i == j:
            continue
        dx = x - circles[j, 0]
        dy = y - circles[j, 1]
        d = np.sqrt(dx * dx + dy * dy)
        # To avoid overlap: d >= r_i + r_j  =>  r_i <= d - r_j
        # But if d < r_j, circle i is inside j; still set constraint
        r = min(r, d - circles[j, 2])
    return max(0.0, r)

def _clearance(circles, i):
    """How much circle i could grow at its current position before hitting a constraint."""
    return _max_radius_at(circles[i, 0], circles[i, 1], circles, i) - circles[i, 2]

def _is_valid(circles, eps=1e-9):
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(x < -eps) or np.any(x > 1 + eps) or np.any(y < -eps) or np.any(y > 1 + eps):
        return False
    if np.any(r < -eps):
        return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - eps:
                return False
    return True

def _sum_radii(circles):
    return float(np.sum(circles[:, 2]))

def _greedy_expand(circles, max_rounds=200, deadline=None):
    """Iteratively expand the circle with the most clearance until it touches a constraint."""
    circles = circles.copy()
    n = len(circles)
    for _ in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        # Find circle with maximum positive clearance
        best_i = -1
        best_clear = 0.0
        for i in range(n):
            c = _clearance(circles, i)
            if c > best_clear:
                best_clear = c
                best_i = i
        if best_i == -1:
            break
        # Expand circle best_i to its maximum allowed radius at current position
        circles[best_i, 2] = _max_radius_at(circles[best_i, 0], circles[best_i, 1], circles, best_i)
    return circles

def _local_jiggle(circles, n_tries=50, deadline=None):
    """Try small random perturbations of positions, then re-expand greedily."""
    best = circles.copy()
    best_score = _sum_radii(best)
    n = len(circles)
    for _ in range(n_tries):
        if deadline is not None and time.time() > deadline:
            break
        trial = circles.copy()
        # Pick a random circle and nudge it slightly
        i = np.random.randint(n)
        trial[i, 0] += np.random.uniform(-0.02, 0.02)
        trial[i, 1] += np.random.uniform(-0.02, 0.02)
        # Clamp to square (keep center inside)
        trial[i, 0] = np.clip(trial[i, 0], 0.0, 1.0)
        trial[i, 1] = np.clip(trial[i, 1], 0.0, 1.0)
        # Re-expand greedily from this new configuration
        trial = _greedy_expand(trial, max_rounds=100, deadline=deadline)
        if _is_valid(trial, eps=1e-7):
            s = _sum_radii(trial)
            if s > best_score:
                best_score = s
                best = trial.copy()
    return best

def _init_grid():
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([cx, cy, r])
    return np.array(cells)

def _init_random(seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.03)
    return np.column_stack([x, y, r])

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    np.random.seed(123)
    
    best_circles = None
    best_score = -np.inf
    
    # Multiple diverse initializations
    inits = []
    inits.append(("grid", _init_grid()))
    for s in [1, 7, 42, 99, 2024]:
        inits.append((f"rand_{s}", _init_random(s)))
    
    for name, init in inits:
        if time.time() > deadline - 5:
            break
        current = init.copy()
        
        # Phase 1: Greedy expansion
        expanded = _greedy_expand(current, max_rounds=300, deadline=deadline - 5)
        
        # Phase 2: Local jiggle + re-expand
        improved = _local_jiggle(expanded, n_tries=40, deadline=deadline - 5)
        
        if _is_valid(improved, eps=1e-7):
            s = _sum_radii(improved)
            if s > best_score:
                best_score = s
                best_circles = improved.copy()
    
    # Fallback: return grid if nothing valid was found (should not happen)
    if best_circles is None:
        best_circles = _init_grid()
    
    return best_circles
