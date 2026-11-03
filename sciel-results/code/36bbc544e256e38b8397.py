"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

N = 26

# EVOLVE-BLOCK-START

def _radii_from_positions(pos, n):
    """
    Compute radii from center positions using Delaunay triangulation.
    Each circle's radius is half the minimum of:
    - Distance to each Delaunay neighbor
    - Distance to each boundary (x=0, x=1, y=0, y=1)
    This ensures no overlaps and boundary containment.
    """
    radii = np.zeros(n)
    
    # Boundary constraints
    r_boundary = np.minimum(
        np.minimum(pos[:, 0], pos[:, 1]),
        np.minimum(1.0 - pos[:, 0], 1.0 - pos[:, 1])
    )
    
    # Delaunay triangulation for neighbor distances
    try:
        tri = Delaunay(pos)
        # For each point, find minimum distance to its Delaunay neighbors
        for i in range(n):
            simplices = tri.simplices[tri.simplices == i]
            if len(simplices) > 0:
                neighbors = set(simplices.flatten()) - {i}
                min_dist = np.inf
                for j in neighbors:
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < min_dist:
                        min_dist = d
                if min_dist < np.inf:
                    r_boundary[i] = min(r_boundary[i], min_dist / 2.0)
    except Exception:
        pass
    
    # Safety: ensure non-negative
    radii = np.maximum(r_boundary, 0.0)
    
    # Additional safety: verify no overlaps, reduce if needed
    for _ in range(20):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < radii[i] + radii[j] - 1e-12:
                    excess = radii[i] + radii[j] - d
                    # Reduce both proportionally
                    if radii[i] + radii[j] > 0:
                        frac_i = radii[i] / (radii[i] + radii[j])
                        radii[i] -= excess * frac_i
                        radii[j] -= excess * (1 - frac_i)
                        radii[i] = max(0.0, radii[i])
                        radii[j] = max(0.0, radii[j])
                        changed = True
        if not changed:
            break
    
    # Final boundary clamp
    for i in range(n):
        r_bound = min(pos[i, 0], pos[i, 1], 1.0 - pos[i, 0], 1.0 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, r_bound))
    
    return radii

def _objective(pos, n):
    """
    Objective function: minimize negative sum of radii.
    Radii are derived from positions via Delaunay geometry.
    """
    radii = _radii_from_positions(pos, n)
    return -np.sum(radii)

def _gradient(pos, n):
    """
    Numerical gradient for the objective.
    """
    grad = np.zeros_like(pos)
    h = 1e-5
    f0 = _objective(pos, n)
    
    for i in range(n):
        for d in range(2):
            pos_p = pos.copy()
            pos_p[i, d] += h
            f_p = _objective(pos_p, n)
            grad[i, d] = (f_p - f0) / h
    
    return grad

def _initial_hexagonal(n):
    """Create initial hexagonal packing configuration."""
    pos = np.zeros((n, 2))
    
    # Hexagonal arrangement
    spacing = 0.19
    row_spacing = spacing * np.sqrt(3) / 2
    
    idx = 0
    for row in range(8):
        for col in range(8):
            if idx >= n:
                break
            x = (col + 0.5) * spacing
            y = (row + 0.5) * row_spacing
            if row % 2 == 1:
                x += spacing / 2
            x = np.clip(x, 0.02, 0.98)
            y = np.clip(y, 0.02, 0.98)
            pos[idx] = [x, y]
            idx += 1
    
    # Fill remaining with random positions
    while idx < n:
        pos[idx] = [0.1 + 0.8 * np.random.rand(), 0.1 + 0.8 * np.random.rand()]
        idx += 1
    
    return pos

def _initial_grid(n):
    """Create initial grid configuration."""
    pos = np.zeros((n, 2))
    rows = 5
    cols = 6
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            x = (j + 0.5) / cols
            y = (i + 0.5) / rows
            pos[idx] = [x, y]
            idx += 1
    while idx < n:
        pos[idx] = [0.5, 0.5]
        idx += 1
    return pos

def _initial_random(n, seed=42):
    """Create random initial configuration with minimum separation."""
    rng = np.random.RandomState(seed)
    pos = np.zeros((n, 2))
    min_sep = 0.08
    
    for i in range(n):
        attempts = 0
        while attempts < 100:
            p = rng.uniform(0.05, 0.95, size=2)
            ok = True
            for j in range(i):
                if np.linalg.norm(p - pos[j]) < min_sep:
                    ok = False
                    break
            if ok:
                pos[i] = p
                break
            attempts += 1
        if attempts >= 100:
            pos[i] = rng.uniform(0.05, 0.95, size=2)
    
    return pos

def _optimize_positions(pos_init, n, deadline, method='L-BFGS-B'):
    """
    Optimize center positions to maximize sum of radii.
    Radii are derived from positions.
    """
    if time.time() > deadline:
        return pos_init
    
    # Bounds: positions must be in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * n)
    
    # Try L-BFGS-B first
    try:
        result = minimize(
            _objective,
            pos_init,
            args=(n,),
            method=method,
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 5000, 'gtol': 1e-6}
        )
        
        if time.time() > deadline - 5:
            return pos_init
        
        if result.success and result.fun < _objective(pos_init, n):
            return result.x
    except Exception:
        pass
    
    # Fallback: Nelder-Mead
    try:
        result = minimize(
            _objective,
            pos_init,
            args=(n,),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6}
        )
        
        if time.time() > deadline - 5:
            return pos_init
        
        if result.fun < _objective(pos_init, n):
            return result.x
    except Exception:
        pass
    
    return pos_init

def _local_search(pos, n, deadline):
    """
    Local search: try small perturbations to improve the packing.
    """
    best_pos = pos.copy()
    best_radii = _radii_from_positions(pos, n)
    best_score = np.sum(best_radii)
    
    rng = np.random.RandomState(123)
    step_size = 0.02
    no_improve = 0
    
    for it in range(300):
        if time.time() > deadline:
            break
        
        # Pick random circle and direction
        i = rng.randint(n)
        dx = rng.uniform(-step_size, step_size)
        dy = rng.uniform(-step_size, step_size)
        
        new_pos = pos.copy()
        new_pos[i, 0] = np.clip(pos[i, 0] + dx, 0.0, 1.0)
        new_pos[i, 1] = np.clip(pos[i, 1] + dy, 0.0, 1.0)
        
        new_radii = _radii_from_positions(new_pos, n)
        new_score = np.sum(new_radii)
        
        if new_score > best_score + 1e-8:
            pos = new_pos
            best_pos = new_pos.copy()
            best_radii = new_radii.copy()
            best_score = new_score
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve > 50:
            step_size *= 0.9
            no_improve = 0
            if step_size < 0.001:
                break
    
    return best_pos

def _validate(pos, radii, n, tol=1e-9):
    """Check if packing is valid."""
    for i in range(n):
        x, y = pos[i]
        r = radii[i]
        if x - r < -tol or y - r < -tol or x + r > 1 + tol or y + r > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _fix_and_validate(pos, radii, n):
    """
    Ensure validity by clamping radii.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    # Clamp to boundary
    for i in range(n):
        mr = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, mr))
    
    # Fix overlaps by reducing radii
    for _ in range(50):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                md = radii[i] + radii[j]
                if d < md - 1e-9:
                    excess = md - d
                    reduction = excess / 2.0 + 1e-6
                    radii[i] = max(0.0, radii[i] - reduction)
                    radii[j] = max(0.0, radii[j] - reduction)
                    changed = True
        if not changed:
            break
    
    # Final boundary clamp
    for i in range(n):
        mr = min(pos[i, 0], pos[i, 1], 1 - pos[i, 0], 1 - pos[i, 1])
        radii[i] = min(radii[i], max(0.0, mr))
    
    if _validate(pos, radii, n):
        return pos, radii, np.sum(radii)
    else:
        return None, None, -1.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = None
    best_score = -np.inf
    
    # Generate multiple initial configurations
    configs = []
    
    # Hexagonal
    pos_hex = _initial_hexagonal(N)
    configs.append(pos_hex)
    
    # Grid
    pos_grid = _initial_grid(N)
    configs.append(pos_grid)
    
    # Random starts
    for seed in [0, 1, 2, 3, 4]:
        pos_rand = _initial_random(N, seed=seed)
        configs.append(pos_rand)
    
    # Process each config
    for idx, pos_init in enumerate(configs):
        if time.time() > deadline - 15:
            break
        
        try:
            # Optimize positions
            opt_pos = _optimize_positions(pos_init, N, deadline - 10)
            
            if time.time() > deadline - 5:
                # Not enough time for local search, just use current
                radii = _radii_from_positions(opt_pos, N)
                valid_pos, valid_radii, score = _fix_and_validate(opt_pos, radii, N)
                if valid_pos is not None and score > best_score:
                    best_score = score
                    result = np.zeros((N, 3))
                    result[:, :2] = valid_pos
                    result[:, 2] = valid_radii
                    best_result = result
                continue
            
            # Local search
            ls_pos = _local_search(opt_pos, N, deadline - 5)
            
            # Compute final radii
            radii = _radii_from_positions(ls_pos, N)
            valid_pos, valid_radii, score = _fix_and_validate(ls_pos, radii, N)
            
            if valid_pos is not None and score > best_score:
                best_score = score
                result = np.zeros((N, 3))
                result[:, :2] = valid_pos
                result[:, 2] = valid_radii
                best_result = result
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: small circles in a grid
        result = np.zeros((N, 3))
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx >= N:
                    break
                x = (i + 0.5) / 5.0
                y = (j + 0.5) / 5.0
                r = 0.08
                result[idx, :] = [x, y, r]
                idx += 1
        if idx < N:
            result[idx, :] = [0.5, 0.5, 0.05]
        return result
    
    return best_result

# EVOLVE-BLOCK-END
