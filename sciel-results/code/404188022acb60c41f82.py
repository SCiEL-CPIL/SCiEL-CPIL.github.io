"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START

_start_time = time.time()
_TIME_LIMIT = 145  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _validate(circles, tol=1e-7):
    """Check circles are inside unit square and non-overlapping."""
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r < -tol) or np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _fix_boundaries(circles):
    """Project circles inside unit square and ensure non-overlap by shrinking if necessary."""
    circles = circles.copy()
    for i in range(N):
        r = max(0.0, circles[i, 2])
        r = min(r, 0.5)
        circles[i, 2] = r
        circles[i, 0] = np.clip(circles[i, 0], r, 1.0 - r)
        circles[i, 1] = np.clip(circles[i, 1], r, 1.0 - r)
        
    for _iter in range(10):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                if dist < min_dist:
                    overlap = min_dist - dist
                    shrink = overlap / 2.0 + 1e-5
                    circles[i, 2] = max(0.0, circles[i, 2] - shrink)
                    circles[j, 2] = max(0.0, circles[j, 2] - shrink)
                    circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1.0 - circles[i, 2])
                    circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1.0 - circles[i, 2])
                    circles[j, 0] = np.clip(circles[j, 0], circles[j, 2], 1.0 - circles[j, 2])
                    circles[j, 1] = np.clip(circles[j, 1], circles[j, 2], 1.0 - circles[j, 2])
                    changed = True
        if not changed:
            break
            
    return circles


def _energy(circles, beta):
    """
    Compute energy (penalty) for a configuration.
    Lower energy = better (more feasible, larger radii).
    Uses vectorized operations for speed.
    """
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    
    # Boundary violations
    lx = np.maximum(0, r - x)
    ux = np.maximum(0, x + r - 1.0)
    ly = np.maximum(0, r - y)
    uy = np.maximum(0, y + r - 1.0)
    boundary_penalty = np.sum(lx**2) + np.sum(ux**2) + np.sum(ly**2) + np.sum(uy**2)
    
    # Negative radius
    neg_r = np.maximum(0, -r)
    neg_penalty = np.sum(neg_r**2)
    
    # Pairwise overlaps (vectorized)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = np.maximum(0, r[idx_i] + r[idx_j] - dist)
    overlap_penalty = np.sum(overlap**2)
    
    # Total energy: negative sum of radii + weighted penalties
    return -np.sum(r) + beta * (boundary_penalty + neg_penalty + overlap_penalty)


def _initial_hex():
    """Hexagonal packing initial config."""
    cols = 5
    rows = 6
    positions = []
    r = 0.09
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5 / cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])


def _initial_grid():
    """Grid initial config."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5 / side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])


def _initial_corner_weighted(seed=0):
    """Random config with corner preference."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
               [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.08])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])


def _monte_carlo_annealing(xyz0, deadline, beta_start=10.0, beta_end=500.0, 
                           temp_start=0.1, temp_end=0.001, step_scale=0.02):
    """
    Monte Carlo simulated annealing with vectorized energy computation.
    Moves: random position jitter, radius adjustment, or both.
    Accepts worse solutions with probability exp(-dE/T) to escape local minima.
    """
    circles = xyz0.copy()
    best_circles = circles.copy()
    best_score = np.sum(circles[:, 2])
    
    # Precompute indices for pairwise
    n = N
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    # Track time
    t0 = time.time()
    total_steps = 0
    accepted = 0
    
    # Annealing schedule parameters
    n_phases = 5
    for phase in range(n_phases):
        phase_start = phase / n_phases
        phase_end = (phase + 1) / n_phases
        beta = beta_start * (beta_end / beta_start) ** phase_start
        temp = temp_start * (temp_end / temp_start) ** phase_start
        
        # Run iterations until deadline or phase time exhausted
        phase_deadline = time.time() + (deadline - time.time()) * 0.8 / n_phases
        while time.time() < phase_deadline:
            # Generate random move
            move_type = np.random.randint(0, 3)
            new_circles = circles.copy()
            
            if move_type == 0:
                # Position jitter
                i = np.random.randint(n)
                new_circles[i, 0] += np.random.normal(0, step_scale)
                new_circles[i, 1] += np.random.normal(0, step_scale)
            elif move_type == 1:
                # Radius adjustment
                i = np.random.randint(n)
                new_circles[i, 2] += np.random.normal(0, step_scale * 0.5)
            else:
                # Both position and radius
                i = np.random.randint(n)
                new_circles[i, 0] += np.random.normal(0, step_scale)
                new_circles[i, 1] += np.random.normal(0, step_scale)
                new_circles[i, 2] += np.random.normal(0, step_scale * 0.5)
            
            # Clip to valid range
            new_circles[:, 0] = np.clip(new_circles[:, 0], 0, 1)
            new_circles[:, 1] = np.clip(new_circles[:, 1], 0, 1)
            new_circles[:, 2] = np.clip(new_circles[:, 2], 0, 0.5)
            
            # Compute energy change
            e_old = _energy(circles, beta)
            e_new = _energy(new_circles, beta)
            dE = e_new - e_old
            
            # Accept if better, or with probability exp(-dE/T) if worse
            if dE < 0 or np.random.random() < np.exp(-dE / max(temp, 1e-10)):
                circles = new_circles
                accepted += 1
                # Update best
                if _validate(circles, tol=1e-6):
                    score = np.sum(circles[:, 2])
                    if score > best_score:
                        best_score = score
                        best_circles = circles.copy()
            
            total_steps += 1
            # Check deadline
            if time.time() > deadline - 5:
                break
        
        # Decay step size
        step_scale *= 0.8
    
    return best_circles, best_score


def construct():
    """
    Pack 26 circles in the unit square, maximizing the sum of radii.
    Uses Monte Carlo simulated annealing with vectorized energy computation.
    """
    deadline = _start_time + _TIME_LIMIT
    
    # Generate multiple initial configurations
    inits = [
        _initial_hex(),
        _initial_grid(),
        _initial_corner_weighted(seed=42),
        _initial_corner_weighted(seed=123),
        _initial_corner_weighted(seed=999),
    ]
    
    best_overall = None
    best_score_overall = -np.inf
    
    for i, xyz0 in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        # Allocate time: give more time to earlier starts
        alloc = min(remaining - 10, max(15, 30 - i * 3))
        local_deadline = time.time() + alloc
        
        circles, score = _monte_carlo_annealing(xyz0, local_deadline)
        
        # Validate and fix
        if _validate(circles):
            if score > best_score_overall:
                best_score_overall = score
                best_overall = circles.copy()
        else:
            circles_fixed = _fix_boundaries(circles)
            if _validate(circles_fixed):
                score_fixed = np.sum(circles_fixed[:, 2])
                if score_fixed > best_score_overall:
                    best_score_overall = score_fixed
                    best_overall = circles_fixed.copy()
    
    # Final repair
    if best_overall is not None:
        repaired = _fix_boundaries(best_overall)
        if _validate(repaired):
            best_overall = repaired
        else:
            best_overall = None
    
    if best_overall is None:
        # Fallback: grid of 26 circles with small radii
        result = np.zeros((N, 3))
        side = 5
        for idx in range(N):
            i = idx % side
            j = idx // side
            result[idx, 0] = (i + 0.5) / side
            result[idx, 1] = (j + 0.5) / side
            result[idx, 2] = 0.5 / side * 0.8
        return result
    
    return best_overall

# EVOLVE-BLOCK-END
