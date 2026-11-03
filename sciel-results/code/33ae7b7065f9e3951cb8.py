"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26
_GLOBAL_START = None
_GLOBAL_DEADLINE = None

# EVOLVE-BLOCK-START

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0) using log-sum-exp."""
    return np.logaddexp(0, z / eps) * eps

def _objective(params, penalty_weight, smooth_eps=1e-4):
    """
    Vectorized objective for N=26 circles.
    params: flat array [x0,y0,r0, x1,y1,r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = np.maximum(params[2::3], 1e-6)
    
    penalty = 0.0
    
    # Boundary violations (smooth)
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    penalty += np.sum(_softplus(lx, smooth_eps)**2)
    penalty += np.sum(_softplus(ux, smooth_eps)**2)
    penalty += np.sum(_softplus(ly, smooth_eps)**2)
    penalty += np.sum(_softplus(uy, smooth_eps)**2)
    
    # Negative radius
    penalty += np.sum(_softplus(-r, smooth_eps)**2)
    
    # Full pairwise overlaps (N=26 is small enough for full pairwise)
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _gradient(params, penalty_weight, smooth_eps=1e-4):
    """Analytic gradient of the smooth penalty objective."""
    x = params[0::3]
    y = params[1::3]
    r = np.maximum(params[2::3], 1e-6)
    
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    grad[2::3] -= 1.0
    
    # Helper: d/dz [softplus(z, eps)^2] = 2 * softplus(z, eps) * sigmoid(z/eps)
    def d_softplus_sq(z, eps):
        sp = _softplus(z, eps)
        sig = 1.0 / (1.0 + np.exp(-z / eps))
        return 2.0 * sp * sig
    
    # Boundary gradients
    for i in range(N):
        # Left wall: z = r - x
        z = r[i] - x[i]
        dz = d_softplus_sq(z, smooth_eps)
        grad[i*3] += -penalty_weight * dz
        grad[i*3+2] += penalty_weight * dz
        
        # Right wall: z = x + r - 1
        z = x[i] + r[i] - 1.0
        dz = d_softplus_sq(z, smooth_eps)
        grad[i*3] += penalty_weight * dz
        grad[i*3+2] += penalty_weight * dz
        
        # Bottom wall: z = r - y
        z = r[i] - y[i]
        dz = d_softplus_sq(z, smooth_eps)
        grad[i*3+1] += -penalty_weight * dz
        grad[i*3+2] += penalty_weight * dz
        
        # Top wall: z = y + r - 1
        z = y[i] + r[i] - 1.0
        dz = d_softplus_sq(z, smooth_eps)
        grad[i*3+1] += penalty_weight * dz
        grad[i*3+2] += penalty_weight * dz
    
    # Pairwise gradients
    idx_i, idx_j = np.triu_indices(N, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    
    dz = d_softplus_sq(overlap, smooth_eps)
    ux = dx / dist
    uy = dy / dist
    
    # Accumulate into grad
    np.add.at(grad[0::3], idx_i, -penalty_weight * dz * ux)
    np.add.at(grad[0::3], idx_j, penalty_weight * dz * ux)
    np.add.at(grad[1::3], idx_i, -penalty_weight * dz * uy)
    np.add.at(grad[1::3], idx_j, penalty_weight * dz * uy)
    np.add.at(grad[2::3], idx_i, penalty_weight * dz)
    np.add.at(grad[2::3], idx_j, penalty_weight * dz)
    
    return grad

def _validate(xyz, tol=1e-7):
    """Check validity."""
    n = xyz.shape[0]
    x, y, r = xyz[:,0], xyz[:,1], xyz[:,2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i]-x[j]
            dy = y[i]-y[j]
            dist = np.sqrt(dx*dx+dy*dy)
            if dist < r[i]+r[j]-tol:
                return False
    return True

def _remaining_time():
    """Get remaining time before deadline."""
    if _GLOBAL_DEADLINE is None:
        return 140.0
    return max(0.0, _GLOBAL_DEADLINE - time.time())

def _fix_boundaries(xyz):
    """Project circles inside unit square."""
    xyz = xyz.copy()
    for i in range(N):
        r = max(0, xyz[i,2])
        xyz[i,0] = np.clip(xyz[i,0], r, 1-r) if r < 0.5 else 0.5
        xyz[i,1] = np.clip(xyz[i,1], r, 1-r) if r < 0.5 else 0.5
        xyz[i,2] = r
    return xyz

def _initial_hex_scaled():
    """Hexagonal packing with scaling to fit unit square."""
    r0 = 0.12
    centers = []
    for row in range(12):
        for col in range(12):
            x = col * 2 * r0
            y = row * np.sqrt(3) * r0
            if row % 2 == 1:
                x += r0
            centers.append((x, y))
    
    # Sort by distance from center to get compact cluster
    centers.sort(key=lambda p: (p[0]-0.5)**2 + (p[1]-0.5)**2)
    centers = centers[:N]
    
    # Calculate bounding box
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    
    if width < 1e-10 or height < 1e-10:
        width = 1.0
        height = 1.0
    
    # Scale to fit in [0.05, 0.95]
    scale = min(0.90 / width, 0.90 / height)
    
    scaled_centers = []
    for x, y in centers:
        sx = (x - min_x) * scale + 0.05
        sy = (y - min_y) * scale + 0.05
        scaled_centers.append((sx, sy))
    
    # Calculate initial radius
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((scaled_centers[i][0]-scaled_centers[j][0])**2 + 
                       (scaled_centers[i][1]-scaled_centers[j][1])**2)
            if d < min_dist:
                min_dist = d
    
    min_boundary = np.inf
    for x, y in scaled_centers:
        min_boundary = min(min_boundary, x, 1-x, y, 1-y)
    
    r_init = min(min_dist / 2, min_boundary) * 0.95
    r_init = max(r_init, 0.03)
    
    positions = []
    for x, y in scaled_centers:
        positions.append([x, y, r_init])
    
    return np.array(positions[:N])

def _initial_grid():
    """Grid initial config."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5/side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i+0.5)/side
            y = (j+0.5)/side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r*0.5])
    return np.array(positions[:N])

def _initial_corner_weighted(seed=0):
    """Random config with corner preference."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9],
               [0.5, 0.1], [0.5, 0.9], [0.1, 0.5], [0.9, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.10])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])

def _initial_ring(seed=0):
    """Ring-shaped initial config."""
    rng = np.random.default_rng(seed)
    positions = [[0.5, 0.5, 0.10]]
    for ring, n_circles, radius in [(1, 6, 0.11), (2, 10, 0.09), (3, 9, 0.07)]:
        for k in range(n_circles):
            angle = 2*np.pi*k/n_circles + rng.uniform(-0.1, 0.1)
            dist = ring * 0.15
            x = 0.5 + dist*np.cos(angle)
            y = 0.5 + dist*np.sin(angle)
            positions.append([x, y, radius])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([rng.uniform(0.1,0.9), rng.uniform(0.1,0.9), 0.06])
    return np.array(positions[:N])

def _initial_corner_cluster():
    """
    Specific initialization placing larger circles in corners and edges.
    """
    positions = []
    # 4 corners
    positions.append([0.12, 0.12, 0.14])
    positions.append([0.88, 0.12, 0.14])
    positions.append([0.12, 0.88, 0.14])
    positions.append([0.88, 0.88, 0.14])
    
    # 4 edge midpoints
    positions.append([0.5, 0.10, 0.11])
    positions.append([0.5, 0.90, 0.11])
    positions.append([0.10, 0.5, 0.11])
    positions.append([0.90, 0.5, 0.11])
    
    # Center
    positions.append([0.5, 0.5, 0.13])
    
    # Fill remaining with smaller circles in gaps
    rng = np.random.default_rng(123)
    while len(positions) < N:
        x = rng.uniform(0.2, 0.8)
        y = rng.uniform(0.2, 0.8)
        r = rng.uniform(0.05, 0.09)
        positions.append([x, y, r])
        
    return np.array(positions[:N])

def _optimize_single(xyz0):
    """Optimize from a single initial config with adaptive penalty and analytic gradient."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 1: Low penalty
    penalty = 5.0
    remaining = _remaining_time()
    if remaining > 8:
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-3),
                          jac=lambda p: _gradient(p, penalty, 1e-3),
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective(params, penalty, 1e-3):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: Medium penalty
    remaining = _remaining_time()
    if remaining > 6:
        penalty = 50.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4),
                          jac=lambda p: _gradient(p, penalty, 1e-4),
                          options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective(params, penalty, 1e-4):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 3: High penalty
    remaining = _remaining_time()
    if remaining > 4:
        penalty = 500.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5),
                          jac=lambda p: _gradient(p, penalty, 1e-5),
                          options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective(params, penalty, 1e-5):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 4: Very high penalty
    remaining = _remaining_time()
    if remaining > 2:
        penalty = 5000.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-6),
                          jac=lambda p: _gradient(p, penalty, 1e-6),
                          options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-14})
            if res.fun < _objective(params, penalty, 1e-6):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Ultra-high penalty
    remaining = _remaining_time()
    if remaining > 1:
        penalty = 50000.0
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-7),
                          jac=lambda p: _gradient(p, penalty, 1e-7),
                          options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-15})
            if res.fun < _objective(params, penalty, 1e-7):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_score

def _sa_search(xyz0, time_budget=6.0, seed=0):
    """Simulated annealing with radius growth."""
    rng = np.random.default_rng(seed)
    xyz = xyz0.copy()
    best_xyz = xyz.copy()
    best_score = np.sum(xyz[:, 2])
    
    t_start = time.time()
    T = 1.0
    T_min = 1e-4
    cooling = 0.995
    
    while time.time() - t_start < time_budget and T > T_min:
        T *= cooling
        remaining = time_budget - (time.time() - t_start)
        if remaining < 0.5:
            break
        
        move_type = rng.random()
        
        if move_type < 0.35:
            # Move a circle
            i = rng.integers(N)
            dx = rng.normal(0, 0.02)
            dy = rng.normal(0, 0.02)
            new_xyz = xyz.copy()
            new_xyz[i, 0] = np.clip(xyz[i, 0] + dx, xyz[i, 2], 1 - xyz[i, 2])
            new_xyz[i, 1] = np.clip(xyz[i, 1] + dy, xyz[i, 2], 1 - xyz[i, 2])
            
        elif move_type < 0.7:
            # Change radius
            i = rng.integers(N)
            factor = rng.uniform(0.85, 1.15)
            new_xyz = xyz.copy()
            new_xyz[i, 2] = np.clip(xyz[i, 2] * factor, 0.01, 0.45)
            r = new_xyz[i, 2]
            new_xyz[i, 0] = np.clip(new_xyz[i, 0], r, 1 - r)
            new_xyz[i, 1] = np.clip(new_xyz[i, 1], r, 1 - r)
            
        else:
            # Swap two circles' positions
            i, j = rng.choice(N, size=2, replace=False)
            new_xyz = xyz.copy()
            new_xyz[i, 0], new_xyz[j, 0] = new_xyz[j, 0], new_xyz[i, 0]
            new_xyz[i, 1], new_xyz[j, 1] = new_xyz[j, 1], new_xyz[i, 1]
        
        if _validate(new_xyz, tol=1e-6):
            new_score = np.sum(new_xyz[:, 2])
            delta = new_score - np.sum(xyz[:, 2])
            if delta > 0 or rng.random() < np.exp(delta / T):
                xyz = new_xyz
                if new_score > best_score:
                    best_score = new_score
                    best_xyz = new_xyz.copy()
        else:
            fixed = _fix_boundaries(new_xyz)
            if _validate(fixed, tol=1e-6):
                new_score = np.sum(fixed[:, 2])
                delta = new_score - np.sum(xyz[:, 2])
                if delta > 0 or rng.random() < np.exp(delta / T):
                    xyz = fixed
                    if new_score > best_score:
                        best_score = new_score
                        best_xyz = fixed.copy()
    
    return best_xyz, best_score

def _final_polish(xyz, time_budget=8.0):
    """High-precision final polish using tight tolerances and analytic gradient."""
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    t_start = time.time()
    phases = [
        (100000.0, 1e-8, 800),
        (1000000.0, 1e-9, 600),
        (10000000.0, 1e-10, 400),
    ]
    
    for penalty, eps, maxiter in phases:
        elapsed = time.time() - t_start
        if elapsed > time_budget - 2.0:
            break
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, eps),
                          jac=lambda p: _gradient(p, penalty, eps),
                          options={'maxiter': maxiter, 'ftol': 1e-18, 'gtol': 1e-16})
            if _validate(res.x.reshape(N, 3), tol=1e-7):
                new_score = np.sum(res.x[2::3])
                if new_score > best_score:
                    best_params = res.x
                    best_score = new_score
                    params = res.x
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_score

def construct():
    global _GLOBAL_START, _GLOBAL_DEADLINE
    _GLOBAL_START = time.time()
    _GLOBAL_DEADLINE = _GLOBAL_START + 145.0
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate diverse initial configurations
    inits = [
        _initial_hex_scaled(),
        _initial_grid(),
        _initial_corner_weighted(0),
        _initial_corner_weighted(1),
        _initial_ring(0),
        _initial_ring(1),
        _initial_corner_cluster(),
        _initial_hex_scaled(),  # Second hex with different scaling
    ]
    
    # Optimize from each
    for xyz0 in inits:
        remaining = _remaining_time()
        if remaining < 12.0:
            break
        xyz_opt, score = _optimize_single(xyz0)
        
        if not _validate(xyz_opt):
            xyz_fix = xyz_opt.copy()
            for _ in range(50):
                if _validate(xyz_fix):
                    break
                xyz_fix[:, 2] *= 0.98
                xyz_fix = _fix_boundaries(xyz_fix)
            xyz_opt = xyz_fix
        
        if not _validate(xyz_opt):
            continue
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Simulated annealing
    remaining = _remaining_time()
    if remaining > 10.0:
        sa_budget = min(6.0, remaining * 0.35)
        xyz_sa, score_sa = _sa_search(best_xyz, time_budget=sa_budget, seed=42)
        if _validate(xyz_sa, tol=1e-6):
            score_sa = np.sum(xyz_sa[:, 2])
            if score_sa > best_score:
                best_score = score_sa
                best_xyz = xyz_sa
    
    # Local refinement from the best solution
    remaining = _remaining_time()
    if remaining > 10.0:
        rng = np.random.default_rng(42)
        for scale in [0.005, 0.01, 0.02, 0.03]:
            remaining = _remaining_time()
            if remaining < 6.0:
                break
            perturbed = best_xyz.copy()
            noise = rng.normal(0, scale, perturbed.shape)
            noise[:, 2] *= 0.1
            perturbed += noise
            perturbed = _fix_boundaries(perturbed)
            
            xyz_opt, score = _optimize_single(perturbed)
            
            if not _validate(xyz_opt):
                xyz_fix = xyz_opt.copy()
                for _ in range(50):
                    if _validate(xyz_fix):
                        break
                    xyz_fix[:, 2] *= 0.98
                    xyz_fix = _fix_boundaries(xyz_fix)
                xyz_opt = xyz_fix
            
            if not _validate(xyz_opt):
                continue
            
            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt
    
    # Final high-precision polish
    remaining = _remaining_time()
    if remaining > 8.0:
        polish_budget = min(8.0, remaining - 4.0)
        xyz_polish, score_polish = _final_polish(best_xyz, time_budget=polish_budget)
        if _validate(xyz_polish, tol=1e-7):
            score_polish = np.sum(xyz_polish[:, 2])
            if score_polish > best_score:
                best_score = score_polish
                best_xyz = xyz_polish
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
