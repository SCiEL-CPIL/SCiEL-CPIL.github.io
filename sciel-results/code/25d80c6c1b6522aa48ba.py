"""Circle packing in unit square - force-based relaxation + SLSQP polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 145.0  # safety margin before 150s kill

def _force_relax(positions, radii, n_iter=2000, lr_init=0.02, lr_decay=0.999):
    """
    Force-based relaxation: circles repel when overlapping.
    positions: (N, 2) array
    radii: (N,) array
    Returns improved positions.
    """
    pos = positions.copy().astype(np.float64)
    r = radii.copy()
    n = len(pos)
    lr = lr_init
    
    for it in range(n_iter):
        # Compute forces
        forces = np.zeros_like(pos)
        # Pairwise repulsion for overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = np.sqrt(d[0]**2 + d[1]**2)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-12:
                    # Overlap: push apart
                    overlap = min_dist - dist
                    force_mag = overlap
                    d_norm = d / dist
                    forces[i] += force_mag * d_norm
                    forces[j] -= force_mag * d_norm
                elif dist < 1e-12:
                    # Coincident: random push
                    rand_dir = np.random.randn(2)
                    rand_dir /= np.linalg.norm(rand_dir)
                    forces[i] += 0.01 * rand_dir
                    forces[j] -= 0.01 * rand_dir
        
        # Wall repulsion: push circles away from boundaries
        for i in range(n):
            # Left wall
            if pos[i, 0] < r[i]:
                forces[i, 0] += (r[i] - pos[i, 0]) * 2.0
            # Right wall
            if pos[i, 0] > 1.0 - r[i]:
                forces[i, 0] -= (pos[i, 0] - (1.0 - r[i])) * 2.0
            # Bottom wall
            if pos[i, 1] < r[i]:
                forces[i, 1] += (r[i] - pos[i, 1]) * 2.0
            # Top wall
            if pos[i, 1] > 1.0 - r[i]:
                forces[i, 1] -= (pos[i, 1] - (1.0 - r[i])) * 2.0
        
        # Update positions
        pos += lr * forces
        # Clamp to valid range
        for i in range(n):
            pos[i, 0] = np.clip(pos[i, 0], r[i], 1.0 - r[i])
            pos[i, 1] = np.clip(pos[i, 1], r[i], 1.0 - r[i])
        
        lr *= lr_decay
    
    return pos


def _is_valid(positions, radii, tol=1e-6):
    """Check if configuration is valid (no overlaps, inside square)."""
    pos = positions
    r = radii
    n = len(pos)
    # Check boundaries
    for i in range(n):
        if pos[i, 0] < -tol or pos[i, 0] > 1.0 + tol:
            return False
        if pos[i, 1] < -tol or pos[i, 1] > 1.0 + tol:
            return False
        if pos[i, 0] < r[i] - tol or pos[i, 0] > 1.0 - r[i] + tol:
            return False
        if pos[i, 1] < r[i] - tol or pos[i, 1] > 1.0 - r[i] + tol:
            return False
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[i] - pos[j]
            dist = np.sqrt(d[0]**2 + d[1]**2)
            if dist < r[i] + r[j] - tol:
                return False
    return True


def _slsqp_polish(positions, radii, maxiter=300):
    """
    Polish using SLSQP to maximize sum of radii.
    Variables: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    """
    n = len(positions)
    r = radii.copy()
    pos = positions.copy()
    
    # Build initial vector
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = pos[i, 0]
        x0[3*i+1] = pos[i, 1]
        x0[3*i+2] = r[i]
    
    def objective(v):
        """Negative sum of radii (we minimize negative)."""
        total_r = 0.0
        for i in range(n):
            total_r += v[3*i+2]
        return -total_r
    
    def jac(v):
        """Gradient of objective."""
        g = np.zeros(3 * n)
        for i in range(n):
            g[3*i+2] = -1.0
        return g
    
    # Constraints: no overlap, inside square
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        # x >= r
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        # x <= 1 - r
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y >= r
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # y <= 1 - r
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2]})
    
    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2 - (v[3*i+2]+v[3*j+2])**2
            })
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
        bounds.append((0.001, 0.5))  # r
    
    result = minimize(objective, x0, method='SLSQP', jac=jac,
                      bounds=bounds, constraints=constraints,
                      options={'maxiter': maxiter, 'ftol': 1e-8})
    
    if result.success or result.fun < 0:
        v = result.x
        new_pos = np.array([[v[3*i], v[3*i+1]] for i in range(n)])
        new_r = np.array([v[3*i+2] for i in range(n)])
        # Verify validity
        if _is_valid(new_pos, new_r, tol=1e-5):
            return new_pos, new_r
    
    return positions, radii


def _random_init(seed):
    """Random initial configuration with small circles."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.05, 0.95, size=(N, 2))
    r = np.full(N, 0.04)
    return pos, r


def _hex_init():
    """Hexagonal packing initial configuration."""
    # Hexagonal arrangement
    pos = np.zeros((N, 2))
    r = np.full(N, 0.06)
    
    # Hex grid spacing
    s = 0.16  # spacing
    count = 0
    row = 0
    while count < N:
        for col in range(7):
            if count >= N:
                break
            x = 0.1 + col * s + (0.5 * s if row % 2 else 0)
            y = 0.1 + row * s * 0.866
            if x < 0.9 and y < 0.9:
                pos[count] = [x, y]
                count += 1
        row += 1
    return pos, r


def _grid_init():
    """Grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    pos = np.zeros((N, 2))
    r = np.full(N, 0.5 / side - 0.01)
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= N:
                break
            pos[count] = [(i + 0.5) / side, (j + 0.5) / side]
            count += 1
    return pos, r


def construct():
    start_time = time.time()
    
    best_score = -np.inf
    best_pos = None
    best_r = None
    
    # Strategy 1: Force relaxation with multiple random restarts
    for seed in range(12):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        pos, r = _random_init(seed)
        # Force relaxation to separate circles
        pos = _force_relax(pos, r, n_iter=1500, lr_init=0.015, lr_decay=0.9995)
        
        # Try to grow radii using force relaxation with increasing radius
        # Start with current positions, try to increase radii
        for grow_iter in range(50):
            if time.time() - start_time > DEADLINE - 30:
                break
            # Try to increase all radii slightly
            r_new = r * 1.01
            # Check if still valid
            if _is_valid(pos, r_new, tol=1e-5):
                r = r_new
            else:
                # Binary search for max scale
                lo, hi = 1.0, 1.01
                for _ in range(10):
                    mid = (lo + hi) / 2
                    r_test = r * mid
                    if _is_valid(pos, r_test, tol=1e-5):
                        lo = mid
                    else:
                        hi = mid
                r = r * lo
                break
        
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_r = r.copy()
    
    # Strategy 2: SLSQP polish on best configuration
    if best_pos is not None and time.time() - start_time < DEADLINE - 15:
        pos_p, r_p = _slsqp_polish(best_pos, best_r, maxiter=200)
        score_p = np.sum(r_p)
        if score_p > best_score:
            best_score = score_p
            best_pos = pos_p
            best_r = r_p
    
    # Strategy 3: Try hex and grid initializations with force relaxation
    for init_func in [_hex_init, _grid_init]:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        pos, r = init_func()
        pos = _force_relax(pos, r, n_iter=2000, lr_init=0.01, lr_decay=0.9995)
        
        # Grow radii
        for grow_iter in range(50):
            if time.time() - start_time > DEADLINE - 20:
                break
            r_new = r * 1.01
            if _is_valid(pos, r_new, tol=1e-5):
                r = r_new
            else:
                lo, hi = 1.0, 1.01
                for _ in range(10):
                    mid = (lo + hi) / 2
                    r_test = r * mid
                    if _is_valid(pos, r_test, tol=1e-5):
                        lo = mid
                    else:
                        hi = mid
                r = r * lo
                break
        
        score = np.sum(r)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            best_r = r.copy()
        
        # SLSQP polish
        if time.time() - start_time < DEADLINE - 10:
            pos_p, r_p = _slsqp_polish(pos, r, maxiter=150)
            score_p = np.sum(r_p)
            if score_p > best_score:
                best_score = score_p
                best_pos = pos_p
                best_r = r_p
    
    # Fallback: if something went wrong, return a valid grid
    if best_pos is None or not _is_valid(best_pos, best_r, tol=1e-4):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Final validation and fix
    if not _is_valid(best_pos, best_r, tol=1e-5):
        # Slightly shrink radii to ensure validity
        scale = 0.999
        best_r = best_r * scale
        if not _is_valid(best_pos, best_r, tol=1e-5):
            scale = 0.99
            best_r = best_r * scale
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_pos[i, 0], best_pos[i, 1], best_r[i]]
    
    return result
