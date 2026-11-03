"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

def _softplus(x, beta=10.0):
    """Smooth approximation of max(0, x)."""
    # np.log1p(np.exp(x)) is unstable for large x, so we use a stable version
    # For x > 20, softplus ~ x. For x < -20, softplus ~ 0.
    # Using np.where for stability
    return np.where(x > 20, x, np.log1p(np.exp(x)))

def _validate(xyz, tol=1e-7):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _physics_relax(xyz, time_limit=5.0, grow_rate=0.0005):
    """
    Force-directed physics relaxation that also grows radii over time.
    This is the primary optimization engine - it naturally handles
    circle packing constraints through repulsion forces.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    # Track best valid configuration
    best_score = np.sum(r)
    best_xyz = xyz.copy()
    
    dt = 0.005
    velocity_damp = 0.95
    vx = np.zeros(N)
    vy = np.zeros(N)

    for iteration in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        # Grow radii slowly to push packing tighter
        r *= (1.0 + grow_rate)
        
        # Clip radii to fit within walls
        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_max)
        r = np.maximum(r, 1e-6)
        
        # Compute forces
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Pairwise repulsion for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) + 1e-10
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    # Strong repulsion proportional to overlap
                    force = (min_dist - dist) * 50.0
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Boundary forces - push circles away from walls
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 100.0
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * 100.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 100.0
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * 100.0
        
        # Update velocities and positions with damping
        vx = vx * velocity_damp + dt * fx
        vy = vy * velocity_damp + dt * fy
        
        x += vx
        y += vy
        
        # Clip to valid positions (hard constraints for stability)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        
        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r
        
        # Check validity and track best
        if _validate(xyz):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_xyz = xyz.copy()
    
    return best_xyz

def _objective_smooth(params, penalty_weight=100.0):
    """
    Objective function: -sum(radii) + penalty.
    Uses softplus for smooth constraints to aid gradient-based optimization.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties (smooth)
    # We want x >= r, x <= 1-r, etc.
    # Violation 1: x - r < 0  => softplus(-(x-r))
    # Violation 2: x + r - 1 > 0 => softplus(x+r-1)
    pen_x_lo = _softplus(-(x - r))
    pen_x_hi = _softplus(x + r - 1)
    pen_y_lo = _softplus(-(y - r))
    pen_y_hi = _softplus(y + r - 1)
    pen_r = _softplus(-r)
    
    penalty += np.sum(pen_x_lo) + np.sum(pen_x_hi)
    penalty += np.sum(pen_y_lo) + np.sum(pen_y_hi)
    penalty += np.sum(pen_r)
    
    # Overlap penalties (smooth)
    # dist < r_i + r_j => softplus((r_i + r_j) - dist)
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > -0.1: # Only compute softplus if close to violation
                penalty += _softplus(overlap)
    
    return -np.sum(r) + penalty_weight * penalty

def _polish_gradient(xyz, time_budget=30.0):
    """
    Refine the configuration using L-BFGS-B with a smooth objective.
    """
    start_time = time.time()
    params = xyz.flatten()
    best_params = params.copy()
    best_obj = _objective_smooth(best_params, 100.0)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
    
    # We can increase penalty weight to enforce constraints more strictly
    # but too high might make it hard to optimize radius.
    # Start with moderate weight.
    for penalty_weight in [100.0, 500.0, 2000.0]:
        if time.time() - start_time > time_budget:
            break
        
        def obj(p):
            return _objective_smooth(p, penalty_weight)
        
        try:
            result = minimize(obj, best_params, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return best_params.reshape(N, 3)

def _enforce_validity(params):
    """
    Final repair step to ensure strict validity.
    Shrinks radii or moves positions to resolve any remaining minor overlaps.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    # Clip radii to fit within walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    # Iteratively resolve overlaps by shrinking
    for _ in range(100):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 1e-9
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            # Shrink the smaller circle preferentially
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        # Re-clip to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    return p

def construct():
    deadline = time.time() + 115  # Safety margin
    
    # Phase 1: Initialization
    # Use a jittered grid to break symmetry and provide a good starting point
    # 5x5 grid covers 25, we need 26. We can use a 6x5 grid and drop one, or 5x5 + 1.
    # Let's use a 6x5 grid and randomly select 26, or just use all 30 and optimize? 
    # No, N is fixed at 26.
    # A 5x5 grid is 25. Adding one more is tricky.
    # Let's use a 6x5 grid (30 points) and randomly pick 26? 
    # Or better: Start with a hexagonal packing approximation.
    
    # Simpler robust init: Random positions, small radii.
    rng = np.random.default_rng(42)
    x = rng.uniform(0.1, 0.9, N)
    y = rng.uniform(0.1, 0.9, N)
    r = np.full(N, 0.02)
    
    xyz = np.column_stack([x, y, r])
    
    # Phase 2: Physics Relaxation
    # This will grow the radii and separate the circles.
    # Allocate ~40% of time to physics.
    physics_time = (deadline - time.time()) * 0.4
    xyz = _physics_relax(xyz, time_limit=physics_time, grow_rate=0.0005)
    
    # Phase 3: Gradient Polish
    # Use L-BFGS-B to fine-tune positions and radii.
    # Allocate ~50% of remaining time to polish.
    remaining_time = deadline - time.time()
    polish_time = remaining_time * 0.6
    xyz = _polish_gradient(xyz, time_budget=polish_time)
    
    # Phase 4: Final Repair
    # Ensure strict validity.
    xyz = _enforce_validity(xyz)
    
    # Final check
    if not _validate(xyz):
        # If still invalid, do a more aggressive shrink
        for _ in range(10):
            xyz[:, 2] *= 0.95
            if _validate(xyz):
                break
    
    return xyz
