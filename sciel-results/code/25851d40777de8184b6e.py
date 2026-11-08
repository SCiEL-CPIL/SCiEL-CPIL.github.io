"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26
import time
_START_TIME = time.time()
_DEADLINE = 115  # safety margin within 120s limit

# EVOLVE-BLOCK-START
import numpy as np
import random
from scipy.optimize import minimize


def _check_validity(circles, tol=1e-9):
    """Check if circles are valid: inside unit square and disjoint."""
    if len(circles) != N:
        return False
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    if np.any(r < 0):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol) or np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    # Check pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _compute_max_radius(circles):
    """Compute the maximum possible radius for each circle given positions."""
    x = circles[:, 0]
    y = circles[:, 1]
    n = len(circles)
    max_r = np.zeros(n)
    
    for i in range(n):
        # Distance to boundaries
        d_bound = min(x[i], 1 - x[i], y[i], 1 - y[i])
        max_r[i] = d_bound
        
        # Distance to other circles
        for j in range(n):
            if i == j:
                continue
            dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
            # Circle i can have radius at most (dist - r_j) / 1 but we're computing max possible
            # Actually for equal radii packing, we want dist >= r_i + r_j
            # If we assume all same radius r, then r <= dist/2
            # But for variable radii, it's more complex
            # Let's just use the constraint that r_i <= (dist - r_j) for any r_j
            # This is tricky, so let's use a simpler approach
            pass
    
    return max_r


def _relaxation_step(circles, lr=0.01, iterations=1000):
    """
    Physics-based relaxation: treat circles as repelling each other and the boundaries.
    We maximize sum of radii by pushing circles apart.
    """
    current = circles.copy()
    n = len(current)
    
    for it in range(iterations):
        if time.time() > _DEADLINE - 5:
            break
        
        x = current[:, 0].copy()
        y = current[:, 1].copy()
        r = current[:, 2].copy()
        
        # Forces on each circle
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    dx = random.uniform(-0.001, 0.001)
                    dy = random.uniform(-0.001, 0.001)
                    dist = np.sqrt(dx*dx + dy*dy)
                
                # Desired distance: r_i + r_j
                desired = r[i] + r[j]
                if dist < desired:
                    # Repulsive force
                    force = (desired - dist) / dist
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
        
        # Boundary forces
        for i in range(n):
            # Left boundary
            if x[i] - r[i] < 0:
                fx[i] += (r[i] - x[i]) * 10
            # Right boundary
            if x[i] + r[i] > 1:
                fx[i] -= (x[i] + r[i] - 1) * 10
            # Bottom boundary
            if y[i] - r[i] < 0:
                fy[i] += (r[i] - y[i]) * 10
            # Top boundary
            if y[i] + r[i] > 1:
                fy[i] -= (y[i] + r[i] - 1) * 10
        
        # Update positions
        for i in range(n):
            x[i] += lr * fx[i]
            y[i] += lr * fy[i]
            
            # Clamp to stay in bounds
            x[i] = max(r[i], min(1 - r[i], x[i]))
            y[i] = max(r[i], min(1 - r[i], y[i]))
        
        current[:, 0] = x
        current[:, 1] = y
    
    return current


def _expand_radii(circles, iterations=500):
    """
    Iteratively try to expand radii while maintaining validity.
    """
    current = circles.copy()
    n = len(current)
    
    for it in range(iterations):
        if time.time() > _DEADLINE - 5:
            break
        
        improved = False
        # Try expanding each circle's radius slightly
        for i in range(n):
            if time.time() > _DEADLINE - 5:
                break
            
            # Try small expansion
            for delta in [0.001, 0.0005, 0.0001]:
                new_r = current[i, 2] + delta
                if new_r > 0.5:
                    continue
                
                # Check if this expansion is valid
                valid = True
                x, y = current[i, 0], current[i, 1]
                
                # Boundary check
                if x - new_r < -1e-9 or x + new_r > 1 + 1e-9 or y - new_r < -1e-9 or y + new_r > 1 + 1e-9:
                    valid = False
                    break
                
                # Check against other circles
                for j in range(n):
                    if i == j:
                        continue
                    dist = np.sqrt((x - current[j, 0])**2 + (y - current[j, 1])**2)
                    if dist < new_r + current[j, 2] - 1e-9:
                        valid = False
                        break
                
                if valid:
                    current[i, 2] = new_r
                    improved = True
                    break
        
        if not improved:
            break
    
    return current


def _build_initial_hex(n_circles=N):
    """Build a hexagonal packing initial guess."""
    best_circles = None
    best_score = -np.inf
    
    # Try different hexagonal spacings
    for d in np.linspace(0.18, 0.28, 15):
        positions = []
        sqrt3 = np.sqrt(3)
        for row in range(10):
            y = d * sqrt3 / 2 * row + d / 2
            if y > 1 - d / 2:
                break
            offset = (d / 2) if row % 2 == 1 else 0
            for col in range(10):
                x = d * col + offset + d / 2
                if x > 1 - d / 2:
                    break
                positions.append((x, y))
        
        if len(positions) < n_circles:
            continue
        
        # Take first n_circles positions
        positions = positions[:n_circles]
        
        # Estimate radii
        circles = []
        valid = True
        for k, (x, y) in enumerate(positions):
            d_bound = min(x, 1 - x, y, 1 - y)
            if d_bound < 0.01:
                valid = False
                break
            d_other = np.inf
            for m, (xm, ym) in enumerate(positions):
                if m == k:
                    continue
                d_other = min(d_other, np.sqrt((x - xm)**2 + (y - ym)**2))
            r = min(d_bound, d_other / 2) * 0.85
            if r < 0.01:
                valid = False
                break
            circles.append([x, y, r])
        
        if not valid:
            continue
        
        circles = np.array(circles)
        if _check_validity(circles):
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles
    
    if best_circles is not None:
        return best_circles
    
    # Fallback: simple grid
    side = int(np.ceil(np.sqrt(n_circles)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:n_circles])


def _build_initial_random(n_circles=N, seed=42):
    """Build a random initial placement."""
    rng = np.random.RandomState(seed)
    circles = []
    for _ in range(n_circles):
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        r = rng.uniform(0.03, 0.08)
        circles.append([x, y, r])
    
    circles = np.array(circles)
    # Make it valid by shrinking
    for i in range(n_circles):
        x, y = circles[i, 0], circles[i, 1]
        d_bound = min(x, 1 - x, y, 1 - y)
        circles[i, 2] = min(circles[i, 2], d_bound * 0.9)
    
    if _check_validity(circles):
        return circles
    
    # If not valid, use hex
    return _build_initial_hex(n_circles)


def _optimize_with_slsqp(circles0, max_iters=300):
    """
    Use SLSQP to optimize positions and radii.
    This is slower but can find better local optima.
    """
    n = len(circles0)
    x0 = circles0.flatten().copy()
    
    def objective(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Penalty for violations
        penalty = 0.0
        
        # Boundary constraints
        for i in range(n):
            if r[i] < 0:
                penalty += 1e4 * (-r[i])
            if x[i] - r[i] < 0:
                penalty += 1e4 * (-(x[i] - r[i]))
            if x[i] + r[i] > 1:
                penalty += 1e4 * (x[i] + r[i] - 1)
            if y[i] - r[i] < 0:
                penalty += 1e4 * (-(y[i] - r[i]))
            if y[i] + r[i] > 1:
                penalty += 1e4 * (y[i] + r[i] - 1)
        
        # Disjoint constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if dist < r[i] + r[j]:
                    penalty += 1e4 * (r[i] + r[j] - dist)
        
        return -np.sum(r) + penalty
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))  # x
        bounds.append((0, 1))  # y
        bounds.append((0, 0.5))  # r
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                         options={'maxiter': max_iters, 'ftol': 1e-8})
        if result.success:
            res = result.x.reshape(n, 3)
            if _check_validity(res):
                return res
    except Exception:
        pass
    
    return circles0


def _local_search_improved(circles, n_iter=1000, seed=42):
    """Improved local search with better perturbation strategy."""
    if time.time() > _DEADLINE:
        return circles
    
    current = circles.copy()
    current_score = np.sum(current[:, 2])
    
    best = current.copy()
    best_score = current_score
    
    rng = random.Random(seed)
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 5:
            break
        
        # Pick a random circle to perturb
        idx = rng.randint(0, N - 1)
        new_circles = current.copy()
        
        # Adaptive perturbation scale
        scale = 0.02 + 0.03 * (1 - it / n_iter)
        
        # Perturb position
        new_circles[idx, 0] += rng.gauss(0, scale)
        new_circles[idx, 1] += rng.gauss(0, scale)
        
        # Perturb radius
        new_circles[idx, 2] *= (1 + rng.gauss(0, 0.05))
        new_circles[idx, 2] = min(max(new_circles[idx, 2], 0.01), 0.5)
        
        # Check validity
        if _check_validity(new_circles):
            new_score = np.sum(new_circles[:, 2])
            if new_score > current_score:
                current = new_circles
                current_score = new_score
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
    
    return best


def construct():
    if time.time() > _DEADLINE:
        # Return a valid fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    best_result = None
    best_value = -np.inf
    
    # Try multiple strategies
    strategies = [
        ("hex", _build_initial_hex),
        ("random1", lambda: _build_initial_random(N, seed=42)),
        ("random2", lambda: _build_initial_random(N, seed=123)),
        ("random3", lambda: _build_initial_random(N, seed=999)),
    ]
    
    for name, builder in strategies:
        if time.time() > _DEADLINE - 20:
            break
        
        try:
            c = builder()
            if not _check_validity(c):
                continue
            
            # Step 1: Relaxation to separate circles
            c_relaxed = _relaxation_step(c, lr=0.005, iterations=500)
            if not _check_validity(c_relaxed):
                c_relaxed = c
            
            # Step 2: Expand radii
            c_expanded = _expand_radii(c_relaxed, iterations=200)
            
            # Step 3: Local search
            c_ls = _local_search_improved(c_expanded, n_iter=500, seed=42 + hash(name) % 100)
            
            # Step 4: More relaxation and expansion
            c_relaxed2 = _relaxation_step(c_ls, lr=0.003, iterations=300)
            if not _check_validity(c_relaxed2):
                c_relaxed2 = c_ls
            c_expanded2 = _expand_radii(c_relaxed2, iterations=150)
            
            # Step 5: Final local search
            c_final = _local_search_improved(c_expanded2, n_iter=300, seed=1000 + hash(name) % 100)
            
            if _check_validity(c_final):
                val = np.sum(c_final[:, 2])
                if val > best_value:
                    best_value = val
                    best_result = c_final.copy()
        except Exception:
            pass
    
    # Try SLSQP on the best result if we have time
    if best_result is not None and time.time() < _DEADLINE - 30:
        try:
            c_slsqp = _optimize_with_slsqp(best_result, max_iters=200)
            if _check_validity(c_slsqp):
                val = np.sum(c_slsqp[:, 2])
                if val > best_value:
                    best_value = val
                    best_result = c_slsqp.copy()
        except Exception:
            pass
    
    # Final validation
    if best_result is not None and _check_validity(best_result):
        return best_result
    
    # Fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
# EVOLVE-BLOCK-END
