"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0)."""
    return np.logaddexp(0, z / eps) * eps

def _softplus_grad(z, eps=1e-4):
    """Gradient of softplus(z, eps)."""
    return 1.0 / (1.0 + np.exp(-z / eps))

def _objective_and_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Vectorized objective and analytic gradient using full pairwise checks.
    params: flat array [x0,y0,r0, x1,y1,r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = np.maximum(params[2::3], 1e-9) # Ensure positive radius for gradient stability
    
    # Objective
    obj = -np.sum(r)
    penalty = 0.0
    
    # Boundary violations (smooth)
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    # Sum of squared softplus penalties
    # f(z) = softplus(z)^2
    # f'(z) = 2 * softplus(z) * sigmoid(z/eps)
    
    # Helper to compute penalty and gradient contribution for a vector of violations
    def _pen_grad(z):
        sp = _softplus(z, smooth_eps)
        sig = _softplus_grad(z, smooth_eps)
        p = np.sum(sp * sp)
        g = 2.0 * sp * sig
        return p, g

    p_lx, g_lx = _pen_grad(lx)
    p_ux, g_ux = _pen_grad(ux)
    p_ly, g_ly = _pen_grad(ly)
    p_uy, g_uy = _pen_grad(uy)
    
    penalty += p_lx + p_ux + p_ly + p_uy
    
    # Negative radius penalty (though r is clipped, we penalize small r to encourage growth if possible? 
    # Actually, we want to maximize r, so -sum(r) handles that. 
    # We just need to ensure r > 0. The boundary constraints r-x >= 0 and x+r <= 1 imply r <= min(x, 1-x).
    # If x is in [0,1], max r is 0.5.
    # We don't strictly need a penalty for r<0 if we clip r in the gradient calculation, 
    # but let's add a small penalty for r < 1e-6 to avoid division by zero in overlaps if dist=0.
    # Actually, dist=0 is the only issue.
    
    # Pairwise overlaps
    n = N
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    
    overlap = r[idx_i] + r[idx_j] - dist
    
    p_ov, g_ov = _pen_grad(overlap)
    penalty += p_ov
    
    total_obj = obj + penalty_weight * penalty
    
    # Gradient
    grad = np.zeros_like(params)
    
    # Gradient of -sum(r)
    grad[2::3] -= 1.0
    
    # Boundary gradients
    # d/dx_i (softplus(r_i - x_i)^2) = - g_lx[i]
    # d/dr_i (softplus(r_i - x_i)^2) = + g_lx[i]
    grad[0::3] -= penalty_weight * g_lx
    grad[2::3] += penalty_weight * g_lx
    
    # d/dx_i (softplus(x_i + r_i - 1)^2) = + g_ux[i]
    # d/dr_i (softplus(x_i + r_i - 1)^2) = + g_ux[i]
    grad[0::3] += penalty_weight * g_ux
    grad[2::3] += penalty_weight * g_ux
    
    # d/dy_i (softplus(r_i - y_i)^2) = - g_ly[i]
    # d/dr_i (softplus(r_i - y_i)^2) = + g_ly[i]
    grad[1::3] -= penalty_weight * g_ly
    grad[2::3] += penalty_weight * g_ly
    
    # d/dy_i (softplus(y_i + r_i - 1)^2) = + g_uy[i]
    # d/dr_i (softplus(y_i + r_i - 1)^2) = + g_uy[i]
    grad[1::3] += penalty_weight * g_uy
    grad[2::3] += penalty_weight * g_uy
    
    # Pairwise gradients
    # d/dx_i overlap = - dx/dist
    # d/dx_j overlap = + dx/dist
    # d/dr_i overlap = 1
    # d/dr_j overlap = 1
    
    ux = dx / dist
    uy = dy / dist
    
    # Accumulate gradients using np.add.at
    # For each pair (i, j):
    # grad_x[i] += -pen * g_ov * ux
    # grad_x[j] += +pen * g_ov * ux
    # grad_y[i] += -pen * g_ov * uy
    # grad_y[j] += +pen * g_ov * uy
    # grad_r[i] += +pen * g_ov
    # grad_r[j] += +pen * g_ov
    
    np.add.at(grad[0::3], idx_i, -penalty_weight * g_ov * ux)
    np.add.at(grad[0::3], idx_j, penalty_weight * g_ov * ux)
    np.add.at(grad[1::3], idx_i, -penalty_weight * g_ov * uy)
    np.add.at(grad[1::3], idx_j, penalty_weight * g_ov * uy)
    np.add.at(grad[2::3], idx_i, penalty_weight * g_ov)
    np.add.at(grad[2::3], idx_j, penalty_weight * g_ov)
    
    return total_obj, grad

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

def _fix_boundaries(xyz):
    """Project circles inside unit square and fix overlaps by shrinking."""
    xyz = xyz.copy()
    # First, clip positions and radii to basic bounds
    for i in range(N):
        r = max(1e-6, xyz[i,2])
        xyz[i,0] = np.clip(xyz[i,0], 0, 1)
        xyz[i,1] = np.clip(xyz[i,1], 0, 1)
        # Ensure r is small enough to fit in box given position
        max_r = min(xyz[i,0], 1-xyz[i,0], xyz[i,1], 1-xyz[i,1])
        xyz[i,2] = min(r, max_r)
        
    # Iteratively shrink radii to resolve overlaps
    for _ in range(50):
        overlaps = []
        for i in range(N):
            for j in range(i+1, N):
                dx = xyz[i,0] - xyz[j,0]
                dy = xyz[i,1] - xyz[j,1]
                dist = np.sqrt(dx*dx+dy*dy)
                ov = xyz[i,2] + xyz[j,2] - dist
                if ov > 1e-9:
                    overlaps.append((i, j, ov))
        if not overlaps:
            break
        for i, j, ov in overlaps:
            # Shrink the larger circle or split
            r_i = xyz[i,2]
            r_j = xyz[j,2]
            if r_i >= r_j:
                shrink = min(ov, r_i - 1e-6)
                xyz[i,2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, r_j - 1e-6)
                    xyz[j,2] -= shrink2
            else:
                shrink = min(ov, r_j - 1e-6)
                xyz[j,2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, r_i - 1e-6)
                    xyz[i,2] -= shrink2
    return xyz

def _initial_hex_scaled():
    """
    Generate a hexagonal grid, select 26 points, scale to fit in [0.1, 0.9], 
    and set initial radii based on local clearance.
    """
    # Generate a larger hex grid
    r_grid = 0.1
    centers = []
    for row in range(15):
        for col in range(15):
            x = col * 2 * r_grid
            y = row * np.sqrt(3) * r_grid
            if row % 2 == 1:
                x += r_grid
            centers.append((x, y))
    
    # Sort by distance from center (0.5, 0.5) to pick a compact cluster
    centers.sort(key=lambda p: (p[0]-0.5)**2 + (p[1]-0.5)**2)
    centers = centers[:N]
    
    # Calculate bounding box
    xs = [c[0] for c in centers]
    ys = [c[1] for c in centers]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    width = max_x - min_x
    height = max_y - min_y
    
    # Scale to fit in [0.1, 0.9]
    scale = min(0.8 / width, 0.8 / height)
    
    scaled_centers = []
    for x, y in centers:
        sx = (x - min_x) * scale + 0.1
        sy = (y - min_y) * scale + 0.1
        scaled_centers.append((sx, sy))
        
    # Calculate initial radii
    # Min distance between any two centers
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt((scaled_centers[i][0]-scaled_centers[j][0])**2 + (scaled_centers[i][1]-scaled_centers[j][1])**2)
            if d < min_dist:
                min_dist = d
                
    # Min distance to boundary
    min_boundary = np.inf
    for x, y in scaled_centers:
        min_boundary = min(min_boundary, x, 1-x, y, 1-y)
        
    r_init = min(min_dist / 2, min_boundary) * 0.95
    
    v = []
    for x, y in scaled_centers:
        v.extend([x, y, r_init])
    return np.array(v)

def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config using analytic gradients."""
    best_params = xyz0.copy()
    best_score = np.sum(xyz0[:, 2])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))
        
    # Phase 1: Coarse optimization with moderate penalty
    remaining = deadline - time.time()
    if remaining > 10:
        try:
            res = minimize(_objective_and_grad, best_params, 
                          method='L-BFGS-B', jac=True,
                          bounds=bounds, args=(10.0, 1e-3),
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_and_grad(best_params, 10.0, 1e-3)[0]:
                best_params = res.x
                if np.sum(best_params[2::3]) > best_score:
                    best_score = np.sum(best_params[2::3])
        except Exception:
            pass
            
    # Phase 2: Medium optimization with higher penalty
    remaining = deadline - time.time()
    if remaining > 10:
        try:
            res = minimize(_objective_and_grad, best_params, 
                          method='L-BFGS-B', jac=True,
                          bounds=bounds, args=(100.0, 1e-4),
                          options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_and_grad(best_params, 100.0, 1e-4)[0]:
                best_params = res.x
                if np.sum(best_params[2::3]) > best_score:
                    best_score = np.sum(best_params[2::3])
        except Exception:
            pass
            
    # Phase 3: Fine optimization with high penalty
    remaining = deadline - time.time()
    if remaining > 5:
        try:
            res = minimize(_objective_and_grad, best_params, 
                          method='L-BFGS-B', jac=True,
                          bounds=bounds, args=(1000.0, 1e-5),
                          options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective_and_grad(best_params, 1000.0, 1e-5)[0]:
                best_params = res.x
                if np.sum(best_params[2::3]) > best_score:
                    best_score = np.sum(best_params[2::3])
        except Exception:
            pass
            
    # Phase 4: Very fine polish
    remaining = deadline - time.time()
    if remaining > 3:
        try:
            res = minimize(_objective_and_grad, best_params, 
                          method='L-BFGS-B', jac=True,
                          bounds=bounds, args=(5000.0, 1e-6),
                          options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < _objective_and_grad(best_params, 5000.0, 1e-6)[0]:
                best_params = res.x
                if np.sum(best_params[2::3]) > best_score:
                    best_score = np.sum(best_params[2::3])
        except Exception:
            pass
            
    return best_params, best_score

def construct():
    global_start = time.time()
    deadline = global_start + 115.0  # Safety margin
    
    # Generate diverse initial configurations
    inits = []
    inits.append(_initial_hex_scaled())
    
    # Add a few random perturbations of the hex grid to escape potential local minima
    rng = np.random.default_rng(42)
    for _ in range(3):
        xyz_pert = inits[0].copy()
        xyz_pert[:, 0] += rng.normal(0, 0.02, N)
        xyz_pert[:, 1] += rng.normal(0, 0.02, N)
        xyz_pert[:, 2] *= rng.uniform(0.9, 1.0, N)
        xyz_pert = _fix_boundaries(xyz_pert)
        inits.append(xyz_pert)
        
    best_xyz = inits[0].copy()
    best_score = np.sum(best_xyz[:, 2])
    
    # Optimize from each initial config
    for idx, xyz0 in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 20.0:
            break
            
        # Allocate time budget
        # First init gets most time
        if idx == 0:
            time_budget = deadline - 5.0
        else:
            time_budget = deadline - 3.0
            
        try:
            xyz_opt, score = _optimize_single(xyz0, time_budget)
            
            # Fix if invalid
            if not _validate(xyz_opt):
                xyz_fix = _fix_boundaries(xyz_opt)
                xyz_opt = xyz_fix
                
            if not _validate(xyz_opt):
                continue
                
            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt.copy()
                
        except Exception:
            continue
            
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _fix_boundaries(best_xyz)
        
    return best_xyz

# EVOLVE-BLOCK-END
