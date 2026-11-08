"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 150.0  # seconds
_START_TIME = None

def _init_grid(n=21):
    """Initialize circles in a grid layout inside a 1x1 square."""
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                positions.append(((i + 0.5) / side, (j + 0.5) / side))
    # Add remaining circles with small radius in corners if needed
    while len(positions) < n:
        positions.append((0.5, 0.5))
    
    # Initial radii: start small, we'll grow them
    base_r = 0.5 / side
    radii = np.full(n, base_r * 0.8)
    
    return np.array(positions), radii

def _check_validity(positions, radii, W, H):
    """Check if circles are valid: inside rectangle, non-overlapping."""
    # Check bounds
    for i in range(len(positions)):
        x, y = positions[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > W + 1e-9 or y - r < -1e-9 or y + r > H + 1e-9:
            return False
    
    # Check overlaps
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-9:
                return False
    return True

def _objective_fixed_box(params, W, H, n=21):
    """Objective: minimize negative sum of radii (i.e., maximize sum)."""
    positions = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Penalty for going outside bounds
    penalty = 0.0
    for i in range(n):
        x, y = positions[i]
        r = radii[i]
        if x - r < 0:
            penalty += (r - x) ** 2
        if x + r > W:
            penalty += (x + r - W) ** 2
        if y - r < 0:
            penalty += (r - y) ** 2
        if y + r > H:
            penalty += (y + r - H) ** 2
    
    # Penalty for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist = np.sqrt(dx * dx + dy * dy + 1e-12)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2
    
    return -np.sum(radii) + 100.0 * penalty

def _optimize_from_init(positions, radii, W, H, n=21, max_iters=200):
    """Run optimization from given initial state."""
    params = np.concatenate([positions.flatten(), radii])
    
    # Bounds for positions and radii
    bounds = []
    for i in range(n):
        bounds.append((0.0, W))  # x
        bounds.append((0.0, H))  # y
    for i in range(n):
        bounds.append((1e-4, max(W, H) / 2))  # radii
    
    result = minimize(
        _objective_fixed_box,
        params,
        args=(W, H, n),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iters, 'ftol': 1e-8, 'gtol': 1e-6}
    )
    
    new_positions = result.x[:2*n].reshape(n, 2)
    new_radii = result.x[2*n:]
    
    # Clean up: ensure circles are strictly inside
    for i in range(n):
        new_radii[i] = min(new_radii[i], 
                          new_positions[i, 0] - 1e-6, 
                          W - new_positions[i, 0] - 1e-6,
                          new_positions[i, 1] - 1e-6,
                          H - new_positions[i, 1] - 1e-6)
        new_radii[i] = max(new_radii[i], 1e-4)
    
    return new_positions, new_radii, result.fun

def _try_aspect_ratio(W, H, n=21, max_iters=150):
    """Try to pack circles in a W x H rectangle."""
    global _START_TIME
    if time.time() - _START_TIME > _DEADLINE - 5:
        return None, None, None
    
    positions, radii = _init_grid(n)
    # Scale initial positions to fit in W x H
    positions[:, 0] *= W
    positions[:, 1] *= H
    radii *= min(W, H)
    
    positions, radii, fun = _optimize_from_init(positions, radii, W, H, n, max_iters)
    
    if not _check_validity(positions, radii, W, H):
        return None, None, None
    
    return positions, radii, -fun

def construct():
    global _START_TIME
    _START_TIME = time.time()
    n = N
    
    best_positions = None
    best_radii = None
    best_value = -np.inf
    
    # Try several aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.67, 0.5]
    
    for ar in aspect_ratios:
        if time.time() - _START_TIME > _DEADLINE - 10:
            break
        
        # Perimeter = 2(W + H) = 4, so W + H = 2
        # If W/H = ar, then W = ar*H, so H(1 + ar) = 2, H = 2/(1+ar), W = 2*ar/(1+ar)
        H = 2.0 / (1.0 + ar)
        W = 2.0 * ar / (1.0 + ar)
        
        positions, radii, value = _try_aspect_ratio(W, H, n)
        
        if positions is not None:
            val = np.sum(radii)
            if val > best_value:
                best_value = val
                best_positions = positions.copy()
                best_radii = radii.copy()
    
    if best_positions is None:
        # Fallback: return grid
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:n])
    
    # Build result array
    result = np.column_stack([best_positions, best_radii])
    return result
# EVOLVE-BLOCK-END
