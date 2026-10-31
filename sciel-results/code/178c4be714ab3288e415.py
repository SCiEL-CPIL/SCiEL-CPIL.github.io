"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _pairwise_indices(n):
    """Return list of (i, j) for i < j."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

PAIRS = _pairwise_indices(N)
NUM_PAIRS = len(PAIRS)

def _make_obj_and_constraints(positions, radii, mu=100.0):
    """
    Build objective and constraints for a given initial configuration.
    We optimize over variables [x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}].
    Objective: maximize sum(r_i) - mu * sum(overlap_penalty)
    Constraints:
      - x_i - r_i >= 0
      - 1 - x_i - r_i >= 0
      - y_i - r_i >= 0
      - 1 - y_i - r_i >= 0
      - r_i > 0 (handled by bounds)
      - distance between i and j >= r_i + r_j (as inequality constraint)
    """
    n = len(positions)
    x = positions[:, 0]
    y = positions[:, 1]
    r = radii
    
    def objective(v):
        """Minimize negative sum of radii plus penalty for violations."""
        xv = v[:n]
        yv = v[n:2*n]
        rv = v[2*n:3*n]
        
        # Penalty for overlaps
        penalty = 0.0
        for (i, j) in PAIRS:
            dx = xv[i] - xv[j]
            dy = yv[i] - yv[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rv[i] + rv[j]
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
        
        # Penalty for boundary violations
        for i in range(n):
            if xv[i] < rv[i]:
                penalty += (rv[i] - xv[i]) ** 2
            if xv[i] > 1 - rv[i]:
                penalty += (xv[i] - (1 - rv[i])) ** 2
            if yv[i] < rv[i]:
                penalty += (rv[i] - yv[i]) ** 2
            if yv[i] > 1 - rv[i]:
                penalty += (yv[i] - (1 - rv[i])) ** 2
        
        return -np.sum(rv) + mu * penalty
    
    def jac(v):
        """Jacobian of objective."""
        xv = v[:n]
        yv = v[n:2*n]
        rv = v[2*n:3*n]
        
        g = np.zeros(3*n)
        
        # Gradient of -sum(rv)
        g[2*n:3*n] = -1.0
        
        # Gradient of penalty terms
        for (i, j) in PAIRS:
            dx = xv[i] - xv[j]
            dy = yv[i] - yv[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = rv[i] + rv[j]
            if dist < min_dist and dist > 1e-12:
                diff = min_dist - dist
                # d/dx_i of (min_dist - dist)^2
                g[i] += 2 * mu * diff * (-dx / dist)
                g[j] += 2 * mu * diff * (dx / dist)
                g[n+i] += 2 * mu * diff * (-dy / dist)
                g[n+j] += 2 * mu * diff * (dy / dist)
                g[2*n+i] += 2 * mu * diff
                g[2*n+j] += 2 * mu * diff
        
        for i in range(n):
            if xv[i] < rv[i]:
                diff = rv[i] - xv[i]
                g[i] += 2 * mu * diff * (-1)
                g[2*n+i] += 2 * mu * diff
            if xv[i] > 1 - rv[i]:
                diff = xv[i] - (1 - rv[i])
                g[i] += 2 * mu * diff
                g[2*n+i] += 2 * mu * diff * (-1)
            if yv[i] < rv[i]:
                diff = rv[i] - yv[i]
                g[n+i] += 2 * mu * diff * (-1)
                g[2*n+i] += 2 * mu * diff
            if yv[i] > 1 - rv[i]:
                diff = yv[i] - (1 - rv[i])
                g[n+i] += 2 * mu * diff
                g[2*n+i] += 2 * mu * diff * (-1)
        
        return g
    
    # Constraints for hard constraints (inequality: c(v) >= 0)
    cons = []
    
    # Boundary constraints: x_i - r_i >= 0
    for i in range(n):
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[i] - v[2*n+i],
            'jac': lambda v, i=i: np.concatenate([
                np.array([-1.0] if k == 2*n+i else ([1.0] if k == i else [0.0]) for k in range(3*n))
            ])
        })
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1 - v[i] - v[2*n+i],
            'jac': lambda v, i=i: np.array([
                -1.0 if k == i else (-1.0 if k == 2*n+i else 0.0) for k in range(3*n)
            ])
        })
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[n+i] - v[2*n+i],
            'jac': lambda v, i=i: np.array([
                -1.0 if k == 2*n+i else (1.0 if k == n+i else 0.0) for k in range(3*n)
            ])
        })
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1 - v[n+i] - v[2*n+i],
            'jac': lambda v, i=i: np.array([
                -1.0 if k == n+i else (-1.0 if k == 2*n+i else 0.0) for k in range(3*n)
            ])
        })
    
    # Non-overlap constraints: dist(i,j) - r_i - r_j >= 0
    # These are non-smooth at dist=0, so we use a smoothed version
    for (i, j) in PAIRS:
        cons.append({
            'type': 'ineq',
            'fun': lambda v, i=i, j=j: np.sqrt((v[i]-v[j])**2 + (v[n+i]-v[n+j])**2) - v[2*n+i] - v[2*n+j],
            'jac': lambda v, i=i, j=j: _dist_jac(v, i, j)
        })
    
    return objective, jac, cons

def _dist_jac(v, i, j):
    """Jacobian for distance constraint between circles i and j."""
    n = N
    g = np.zeros(3*n)
    dx = v[i] - v[j]
    dy = v[n+i] - v[n+j]
    dist = np.sqrt(dx*dx + dy*dy)
    if dist < 1e-12:
        dist = 1e-12
    g[i] = dx / dist
    g[j] = -dx / dist
    g[n+i] = dy / dist
    g[n+j] = -dy / dist
    g[2*n+i] = -1.0
    g[2*n+j] = -1.0
    return g

def _initial_grid(n):
    """Create initial grid of circles."""
    side = int(np.ceil(np.sqrt(n)))
    positions = np.zeros((n, 2))
    radii = np.zeros(n)
    for idx in range(n):
        i = idx % side
        j = idx // side
        positions[idx] = [(i + 0.5) / side, (j + 0.5) / side]
        radii[idx] = 0.5 / side - 1e-4
    return positions, radii

def _optimize_from_start(positions, radii, mu=100.0, max_time=100.0):
    """Run optimization from a given starting configuration."""
    n = len(positions)
    x0 = np.concatenate([positions[:, 0], positions[:, 1], radii])
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x_i
    for i in range(n):
        bounds.append((0.0, 1.0))  # y_i
    for i in range(n):
        bounds.append((1e-6, 0.5))  # r_i
    
    obj, jac, cons = _make_obj_and_constraints(positions, radii, mu)
    
    start_time = time.time()
    result = None
    try:
        result = minimize(
            obj, x0, jac=jac, bounds=bounds, constraints=cons,
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
        )
    except Exception:
        pass
    
    if result is not None and result.success:
        xv = result.x[:n]
        yv = result.x[n:2*n]
        rv = result.x[2*n:3*n]
        new_positions = np.column_stack([xv, yv])
        new_radii = rv
        return new_positions, new_radii, -result.fun
    
    # If optimization failed, return original
    return positions, radii, np.sum(radii)

def _check_validity(positions, radii, tol=1e-8):
    """Check if configuration is valid (no overlaps, inside square)."""
    n = len(positions)
    x = positions[:, 0]
    y = positions[:, 1]
    r = radii
    
    # Boundary check
    if np.any(x < -tol) or np.any(x > 1 + tol):
        return False
    if np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(r < -tol):
        return False
    if np.any(x + r > 1 + tol) or np.any(x - r < -tol):
        return False
    if np.any(y + r > 1 + tol) or np.any(y - r < -tol):
        return False
    
    # Overlap check
    for (i, j) in PAIRS:
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist = np.sqrt(dx*dx + dy*dy)
        if dist < r[i] + r[j] - tol:
            return False
    
    return True

def _fix_configuration(positions, radii):
    """Fix any small violations by reducing radii slightly."""
    positions = positions.copy()
    radii = radii.copy()
    n = len(positions)
    
    # Reduce radii to ensure boundary compliance
    for i in range(n):
        max_r = min(positions[i, 0], 1 - positions[i, 0], positions[i, 1], 1 - positions[i, 1])
        if radii[i] > max_r:
            radii[i] = max(max_r * 0.999, 1e-6)
    
    # Reduce radii to ensure non-overlap
    for _ in range(10):
        changed = False
        for (i, j) in PAIRS:
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                # Reduce the larger radius
                if radii[i] >= radii[j]:
                    excess = min_dist - dist
                    reduction = excess * 0.5 + 1e-10
                    radii[i] = max(radii[i] - reduction, 1e-6)
                else:
                    excess = min_dist - dist
                    reduction = excess * 0.5 + 1e-10
                    radii[j] = max(radii[j] - reduction, 1e-6)
                changed = True
        if not changed:
            break
    
    return positions, radii

def construct():
    """Construct 26 circles with disjoint interiors in unit square, maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial configuration
    positions, radii = _initial_grid(N)
    best_positions = positions.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    # Try multiple optimization runs with different initializations
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        # For subsequent attempts, perturb the best configuration
        if attempt == 0:
            pos, rad = positions, radii
        else:
            pos = best_positions.copy() + np.random.randn(N, 2) * 0.01
            rad = best_radii.copy()
            # Ensure positions are in [0,1]
            pos = np.clip(pos, 0.01, 0.99)
        
        # Run optimization
        new_pos, new_rad, new_sum = _optimize_from_start(pos, rad, mu=100.0)
        
        # Check validity and fix if needed
        if not _check_validity(new_pos, new_rad):
            new_pos, new_rad = _fix_configuration(new_pos, new_rad)
            new_sum = np.sum(new_rad)
        
        if new_sum > best_sum:
            best_positions = new_pos
            best_radii = new_rad
            best_sum = new_sum
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        pos, rad = best_positions.copy(), best_radii.copy()
        new_pos, new_rad, new_sum = _optimize_from_start(pos, rad, mu=500.0)
        if not _check_validity(new_pos, new_rad):
            new_pos, new_rad = _fix_configuration(new_pos, new_rad)
            new_sum = np.sum(new_rad)
        if new_sum > best_sum:
            best_positions = new_pos
            best_radii = new_rad
            best_sum = new_sum
    
    # Build result array
    result = np.column_stack([best_positions[:, 0], best_positions[:, 1], best_radii])
    
    # Final safety check
    if not _check_validity(result[:, :2], result[:, 2]):
        result, _ = _fix_configuration(result[:, :2], result[:, 2])
    
    return result

# EVOLVE-BLOCK-END
