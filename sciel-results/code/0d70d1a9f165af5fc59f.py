"""Circle packing in unit square: 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

def _objective_and_grad(pos, radii, lam=50.0):
    """
    Objective: -sum(radii) + penalty for overlaps and boundary violations.
    Variables: pos (n*2,), radii (n,)
    """
    n = len(radii)
    x = pos[0::2]
    y = pos[1::2]
    
    # Boundary violations: circle must be inside [0,1]^2
    # x_i - r_i >= 0, y_i - r_i >= 0, 1 - x_i - r_i >= 0, 1 - y_i - r_i >= 0
    bnd_pen = 0.0
    bnd_grad_pos = np.zeros_like(pos)
    bnd_grad_r = np.zeros_like(radii)
    
    for i in range(n):
        # Left
        v = x[i] - radii[i]
        if v < 0:
            bnd_pen += lam * v * v
            bnd_grad_pos[2*i] += -2 * lam * v
            bnd_grad_r[i] += -2 * lam * v
        # Bottom
        v = y[i] - radii[i]
        if v < 0:
            bnd_pen += lam * v * v
            bnd_grad_pos[2*i+1] += -2 * lam * v
            bnd_grad_r[i] += -2 * lam * v
        # Right
        v = 1.0 - x[i] - radii[i]
        if v < 0:
            bnd_pen += lam * v * v
            bnd_grad_pos[2*i] += 2 * lam * v
            bnd_grad_r[i] += -2 * lam * v
        # Top
        v = 1.0 - y[i] - radii[i]
        if v < 0:
            bnd_pen += lam * v * v
            bnd_grad_pos[2*i+1] += 2 * lam * v
            bnd_grad_r[i] += -2 * lam * v
    
    # Overlap violations: dist(i,j) >= r_i + r_j
    # Penalty: lam * max(0, r_i + r_j - dist)^2
    ovl_pen = 0.0
    ovl_grad_pos = np.zeros_like(pos)
    ovl_grad_r = np.zeros_like(radii)
    
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < 1e-12:
                d = 1e-12
                dx = 1e-12
                dy = 0.0
            r_sum = radii[i] + radii[j]
            gap = r_sum - d
            if gap > 0:
                ovl_pen += lam * gap * gap
                # d/dx_i gap = -dx/d, d/dx_j gap = +dx/d
                # d/d(r_i) gap = 1, d/d(r_j) gap = 1
                coef = -2 * lam * gap / d
                ovl_grad_pos[2*i] += coef * dx
                ovl_grad_pos[2*i+1] += coef * dy
                ovl_grad_pos[2*j] += -coef * dx
                ovl_grad_pos[2*j+1] += -coef * dy
                ovl_grad_r[i] += -2 * lam * gap
                ovl_grad_r[j] += -2 * lam * gap
    
    total_pen = bnd_pen + ovl_pen
    obj = -np.sum(radii) + total_pen
    
    grad_pos = bnd_grad_pos + ovl_grad_pos
    grad_r = bnd_grad_r + ovl_grad_r - 1.0  # d(-sum r)/dr_i = -1
    
    return obj, np.concatenate([grad_pos, grad_r])

def _is_valid(pos, radii, tol=1e-6):
    n = len(radii)
    x = pos[0::2]
    y = pos[1::2]
    for i in range(n):
        if radii[i] < -tol:
            return False
        if x[i] - radii[i] < -tol:
            return False
        if y[i] - radii[i] < -tol:
            return False
        if 1.0 - x[i] - radii[i] < -tol:
            return False
        if 1.0 - y[i] - radii[i] < -tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < radii[i] + radii[j] - tol:
                return False
    return True

def _clamp(pos, radii):
    """Clamp positions and radii to be valid."""
    n = len(radii)
    pos = pos.copy()
    radii = radii.copy()
    radii = np.maximum(radii, 0.0)
    
    # Multiple iterations of clamping
    for _ in range(100):
        x = pos[0::2]
        y = pos[1::2]
        
        # Boundary clamp
        for i in range(n):
            r = radii[i]
            pos[2*i] = np.clip(pos[2*i], r, 1.0 - r)
            pos[2*i+1] = np.clip(pos[2*i+1], r, 1.0 - r)
        
        # Overlap resolution: push apart
        x = pos[0::2]
        y = pos[1::2]
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy)
                r_sum = radii[i] + radii[j]
                if d < r_sum - 1e-12:
                    if d < 1e-12:
                        dx = 1e-12
                        dy = 0.0
                        d = 1e-12
                    push = (r_sum - d) / 2.0 + 1e-9
                    ux = dx / d
                    uy = dy / d
                    x[i] += push * ux
                    y[i] += push * uy
                    x[j] -= push * ux
                    y[j] -= push * uy
                    moved = True
        pos[0::2] = x
        pos[1::2] = y
        
        if not moved:
            break
        
        # Re-clamp boundaries after pushing
        for i in range(n):
            r = radii[i]
            pos[2*i] = np.clip(pos[2*i], r, 1.0 - r)
            pos[2*i+1] = np.clip(pos[2*i+1], r, 1.0 - r)
    
    # Final check and shrink if still invalid
    if not _is_valid(pos, radii):
        # Shrink all radii until valid
        scale = 1.0
        for _ in range(100):
            radii_try = radii * scale
            if _is_valid(pos, radii_try):
                radii = radii_try
                break
            scale *= 0.95
    return pos, radii

def _grid_init():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    pos = np.array([c for cell in cells[:N] for c in cell])
    r0 = 0.5 / side - 1e-3
    radii = np.full(N, r0)
    return pos, radii

def _random_init(rng):
    pos = rng.uniform(0.1, 0.9, size=(N, 2)).flatten()
    radii = rng.uniform(0.02, 0.08, size=N)
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    best_score = -np.inf
    best_pos = None
    best_radii = None
    
    # Start with grid
    pos0, r0 = _grid_init()
    
    # Multi-start optimization
    starts = [(pos0, r0)]
    for _ in range(15):
        p, r = _random_init(rng)
        starts.append((p, r))
    
    for s_idx, (pos_init, rad_init) in enumerate(starts):
        if time.time() > deadline - 10.0:
            break
        
        # Ensure initial validity by clamping
        pos_init, rad_init = _clamp(pos_init.copy(), rad_init.copy())
        
        # If still invalid after clamping, shrink more
        if not _is_valid(pos_init, rad_init):
            scale = 1.0
            while not _is_valid(pos_init, rad_init * scale) and scale > 0.01:
                scale *= 0.9
            rad_init = rad_init * scale
        
        x0 = np.concatenate([pos_init, rad_init])
        
        try:
            res = minimize(
                _objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            x_opt = res.x
            pos_opt = x_opt[:2*N]
            rad_opt = x_opt[2*N:]
            
            # Clamp to valid
            pos_opt, rad_opt = _clamp(pos_opt, rad_opt)
            
            if _is_valid(pos_opt, rad_opt):
                score = np.sum(rad_opt)
                if score > best_score:
                    best_score = score
                    best_pos = pos_opt.copy()
                    best_radii = rad_opt.copy()
        except Exception:
            continue
    
    # If we have a valid solution, try a final polish with higher penalty
    if best_pos is not None and time.time() < deadline - 5.0:
        # Increase lambda for final polish
        x0 = np.concatenate([best_pos, best_radii])
        
        def _obj_polish(x, lam=200.0):
            pos = x[:2*N]
            radii = x[2*N:]
            obj, _ = _objective_and_grad(pos, radii, lam=lam)
            return obj
        
        def _grad_polish(x, lam=200.0):
            pos = x[:2*N]
            radii = x[2*N:]
            _, grad = _objective_and_grad(pos, radii, lam=lam)
            return grad
        
        try:
            res = minimize(
                _obj_polish,
                x0,
                jac=_grad_polish,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            
            x_opt = res.x
            pos_opt = x_opt[:2*N]
            rad_opt = x_opt[2*N:]
            
            pos_opt, rad_opt = _clamp(pos_opt, rad_opt)
            
            if _is_valid(pos_opt, rad_opt):
                score = np.sum(rad_opt)
                if score > best_score:
                    best_score = score
                    best_pos = pos_opt.copy()
                    best_radii = rad_opt.copy()
        except Exception:
            pass
    
    # Final fallback
    if best_pos is None:
        pos, radii = _grid_init()
        pos, radii = _clamp(pos, radii)
        if not _is_valid(pos, radii):
            scale = 1.0
            while not _is_valid(pos, radii * scale) and scale > 0.01:
                scale *= 0.9
            radii = radii * scale
    
    result = np.zeros((N, 3))
    result[:, 0] = best_pos[0::2]
    result[:, 1] = best_pos[1::2]
    result[:, 2] = best_radii
    
    return result
# EVOLVE-BLOCK-END
