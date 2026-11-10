# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _check_valid(centers, sides, tol=1e-7):
    """Check if configuration is valid (vectorized)."""
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    
    # Boundary check
    left = centers[:, 0] - sides / 2
    right = centers[:, 0] + sides / 2
    bottom = centers[:, 1] - sides / 2
    top = centers[:, 1] + sides / 2
    
    if np.any(left < -tol) or np.any(right > 1 + tol):
        return False
    if np.any(bottom < -tol) or np.any(top > 1 + tol):
        return False
        
    # Overlap check
    cx = centers[:, 0]
    cy = centers[:, 1]
    
    dx = np.abs(cx[:, None] - cx[None, :])
    dy = np.abs(cy[:, None] - cy[None, :])
    
    t = (sides[:, None] + sides[None, :]) / 2 - tol
    
    iu = np.triu_indices(n, k=1)
    
    dx_ij = dx[iu]
    dy_ij = dy[iu]
    t_ij = t[iu]
    
    violations = (dx_ij < t_ij) & (dy_ij < t_ij)
    
    return not np.any(violations)

def _objective_neg(x, n=10):
    """Negative of sum of sides (to minimize)."""
    s = x[2::3]
    return -np.sum(s)

def _slsqp_with_binaries(x0, n=10, tol=1e-7, maxiter=200, time_limit=30, rng=None, modes=None):
    """
    Run SLSQP optimization with a fixed assignment of separation directions.
    x: [cx_0, cy_0, s_0, ...]
    Modes: 0: i left of j, 1: i right of j, 2: i below j, 3: i above j.
    """
    if rng is None:
        rng = np.random.default_rng(42)
        
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    if modes is None:
        modes = rng.integers(0, 4, size=len(pairs))
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 1))
    
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2*x[3*i] - x[3*i+2] - tol})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2*(1 - x[3*i]) - x[3*i+2] - tol})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2*x[3*i+1] - x[3*i+2] - tol})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2*(1 - x[3*i+1]) - x[3*i+2] - tol})
        
    # Non-overlap constraints based on modes
    for k, (i, j) in enumerate(pairs):
        mode = modes[k]
        
        if mode == 0: # i left of j
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: x[3*j] - x[3*i] - 0.5*(x[3*i+2] + x[3*j+2]) + tol})
        elif mode == 1: # i right of j
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: x[3*i] - x[3*j] - 0.5*(x[3*i+2] + x[3*j+2]) + tol})
        elif mode == 2: # i below j
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: x[3*j+1] - x[3*i+1] - 0.5*(x[3*i+2] + x[3*j+2]) + tol})
        elif mode == 3: # i above j
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: x[3*i+1] - x[3*j+1] - 0.5*(x[3*i+2] + x[3*j+2]) + tol})

    start_time = time.time()
    
    try:
        result = minimize(
            _objective_neg, x0, method='SLSQP', bounds=bounds,
            constraints=constraints, options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        
        elapsed = time.time() - start_time
        
        if result.success or (result.fun is not None and np.isfinite(result.fun)):
            x_opt = np.clip(result.x, 0, 1)
            x_opt[2::3] = np.maximum(x_opt[2::3], 1e-6)
            for i in range(n):
                s = x_opt[3*i+2]
                x_opt[3*i] = np.clip(x_opt[3*i], s/2, 1-s/2)
                x_opt[3*i+1] = np.clip(x_opt[3*i+1], s/2, 1-s/2)
            return x_opt, -result.fun, elapsed
        
    except Exception:
        pass
    
    return None, None, time.time() - start_time

def _heuristic_init_2_large_8_small():
    """2 large squares (0.5) + 8 small squares (0.25) layout."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.75, 0.25]
    sides[1] = 0.5
    
    for k in range(8):
        col = k % 4
        row = k // 4
        cx = (col + 0.5) * 0.25
        cy = 0.5 + (row + 0.5) * 0.25
        centers[2+k, 0] = cx
        centers[2+k, 1] = cy
        sides[2+k] = 0.25
    
    return centers, sides

def _heuristic_init_grid():
    """4x3 grid, 10 squares of 0.25."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.full(n, 0.25)
    
    for i in range(n):
        col = i % 4
        row = i // 4
        centers[i, 0] = (col + 0.5) * 0.25
        centers[i, 1] = (row + 0.5) * 0.25
    
    return centers, sides

def _heuristic_init_mixed():
    """Try a mixed-size layout: 1 big, rest smaller."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    
    for k in range(6):
        col = k % 2
        row = k // 2
        centers[1+k, 0] = 0.5 + (col + 0.5) * 0.25
        centers[1+k, 1] = (row + 0.5) * (1/3)
        sides[1+k] = 1/3
        
    for k in range(3):
        col = k
        centers[7+k, 0] = (col + 0.5) * (0.5/3)
        centers[7+k, 1] = 0.5 + 0.5/6
        sides[7+k] = 1/3
    
    return centers, sides

def _heuristic_init_corner():
    """Layout with squares in corners and edges."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # 4 corner squares
    corners = [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8)]
    for i, (cx, cy) in enumerate(corners):
        centers[i] = [cx, cy]
        sides[i] = 0.4
    
    # 4 edge squares
    edges = [(0.5, 0.15), (0.15, 0.5), (0.85, 0.5), (0.5, 0.85)]
    for i, (cx, cy) in enumerate(edges):
        centers[4+i] = [cx, cy]
        sides[4+i] = 0.3
    
    # 2 center squares
    centers[8] = [0.35, 0.35]
    sides[8] = 0.25
    centers[9] = [0.65, 0.65]
    sides[9] = 0.25
    
    return centers, sides

def _heuristic_init_staggered():
    """Staggered layout with varying sizes."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # Row 1: 3 squares
    for i in range(3):
        centers[i] = [(i + 0.5) / 3, 0.2]
        sides[i] = 0.35
    
    # Row 2: 3 squares (staggered)
    for i in range(3):
        centers[3+i] = [(i + 0.5) / 3 + 0.05, 0.55]
        sides[3+i] = 0.3
    
    # Row 3: 3 squares
    for i in range(3):
        centers[6+i] = [(i + 0.5) / 3, 0.85]
        sides[6+i] = 0.3
    
    return centers, sides

def _local_improve_sa(centers, sides, tol=1e-7, max_iters=500000, time_limit=60, rng=None):
    """
    Simulated annealing local improvement.
    Faster validity check using vectorized numpy operations.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = np.sum(sides)
    
    if rng is None:
        rng = np.random.default_rng(42)
    
    if not _check_valid(best_centers, best_sides, tol):
        return best_centers, best_sides, best_val
    
    start_time = time.time()
    T = 0.01
    cooling = 0.9999
    min_T = 1e-8
    
    cx = best_centers[:, 0].copy()
    cy = best_centers[:, 1].copy()
    s = best_sides.copy()
    
    # Precompute pair indices
    iu = np.triu_indices(n, k=1)
    pairs = list(zip(*iu))
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if T < min_T:
            break
            
        idx = rng.integers(0, n)
        
        # Perturb
        noise = 0.01 * (0.5 + 0.5 * T / 0.01)
        cx_new = cx.copy()
        cy_new = cy.copy()
        s_new = s.copy()
        
        cx_new[idx] += rng.uniform(-noise, noise)
        cy_new[idx] += rng.uniform(-noise, noise)
        s_new[idx] += rng.uniform(-noise, noise)
        s_new[idx] = max(1e-6, min(1.0, s_new[idx]))
        
        # Clip to bounds
        s_val = s_new[idx]
        cx_new[idx] = np.clip(cx_new[idx], s_val/2, 1-s_val/2)
        cy_new[idx] = np.clip(cy_new[idx], s_val/2, 1-s_val/2)
        
        # Fast validity check
        valid = True
        left = cx_new - s_new / 2
        right = cx_new + s_new / 2
        bottom = cy_new - s_new / 2
        top = cy_new + s_new / 2
        
        if np.any(left < -tol) or np.any(right > 1 + tol):
            valid = False
        if np.any(bottom < -tol) or np.any(top > 1 + tol):
            valid = False
            
        if valid:
            dx = np.abs(cx_new[:, None] - cx_new[None, :])
            dy = np.abs(cy_new[:, None] - cy_new[None, :])
            t = (s_new[:, None] + s_new[None, :]) / 2 - tol
            dx_ij = dx[iu]
            dy_ij = dy[iu]
            t_ij = t[iu]
            violations = (dx_ij < t_ij) & (dy_ij < t_ij)
            if np.any(violations):
                valid = False
        
        if valid:
            new_val = np.sum(s_new)
            delta = new_val - best_val
            if delta > 0 or rng.random() < np.exp(delta / T):
                cx = cx_new
                cy = cy_new
                s = s_new
                best_val = new_val
                best_centers[:, 0] = cx
                best_centers[:, 1] = cy
                best_sides = s.copy()
        
        T *= cooling
    
    return best_centers, best_sides, best_val

def squares_in_square10() -> tuple:
    """
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = 615  # seconds, keep safety margin before 630s kill
    
    best_centers = None
    best_sides = None
    best_val = -1.0
    
    def _try_config(centers, sides, label=""):
        nonlocal best_centers, best_sides, best_val
        if not _check_valid(centers, sides, tol):
            return
        val = np.sum(sides)
        if val > best_val:
            best_val = val
            best_centers = centers.copy()
            best_sides = sides.copy()
    
    # 1. Start with multiple heuristic configurations
    inits = [
        ("2_large_8_small", _heuristic_init_2_large_8_small),
        ("grid", _heuristic_init_grid),
        ("mixed", _heuristic_init_mixed),
        ("corner", _heuristic_init_corner),
        ("staggered", _heuristic_init_staggered),
    ]
    
    for label, init_fn in inits:
        if time.time() - start_time > deadline - 180:
            break
        c, s = init_fn()
        _try_config(c, s, label)
    
    # 2. SA local improvement from each heuristic
    rng = np.random.default_rng(123)
    for label, init_fn in inits:
        if time.time() - start_time > deadline - 180:
            break
        c, s = init_fn()
        if _check_valid(c, s, tol):
            c_improved, s_improved, val_improved = _local_improve_sa(c, s, tol, max_iters=200000, time_limit=30, rng=rng)
            _try_config(c_improved, s_improved, label + "_sa")
            
    # 3. SLSQP optimization with binary separation modes
    if best_centers is not None:
        for attempt in range(40):
            if time.time() - start_time > deadline - 120:
                break
            
            x0 = np.zeros(3*n)
            x0[0::3] = best_centers[:, 0]
            x0[1::3] = best_centers[:, 1]
            x0[2::3] = best_sides
            
            if attempt > 0:
                noise = 0.1 * rng.random((3*n,))
                x0 = x0 + noise
                x0 = np.clip(x0, 0, 1)
                x0[2::3] = np.clip(x0[2::3], 1e-6, 1)
                for i in range(n):
                    s = x0[3*i+2]
                    x0[3*i] = np.clip(x0[3*i], s/2, 1-s/2)
                    x0[3*i+1] = np.clip(x0[3*i+1], s/2, 1-s/2)
            
            x_opt, val, elapsed = _slsqp_with_binaries(x0, n, tol, maxiter=200, time_limit=10, rng=rng)
            if x_opt is not None:
                cx = x_opt[0::3]
                cy = x_opt[1::3]
                s = x_opt[2::3]
                centers = np.column_stack([cx, cy])
                _try_config(centers, s, f"slsqp_{attempt}")
                
                # SA polish on SLSQP result
                if _check_valid(centers, s, tol):
                    c_pol, s_pol, val_pol = _local_improve_sa(centers, s, tol, max_iters=50000, time_limit=10, rng=rng)
                    _try_config(c_pol, s_pol, f"slsqp_{attempt}_polish")
                
    # 4. More SA restarts
    if best_centers is not None:
        for r in range(20):
            if time.time() - start_time > deadline - 60:
                break
            
            c_pert = best_centers + rng.uniform(-0.1, 0.1, (n, 2))
            s_pert = best_sides + rng.uniform(-0.1, 0.1, n)
            s_pert = np.clip(s_pert, 0.01, 1.0)
            
            for i in range(n):
                s = s_pert[i]
                c_pert[i, 0] = np.clip(c_pert[i, 0], s/2, 1-s/2)
                c_pert[i, 1] = np.clip(c_pert[i, 1], s/2, 1-s/2)
            
            if _check_valid(c_pert, s_pert, tol):
                c_imp, s_imp, val_imp = _local_improve_sa(c_pert, s_pert, tol, max_iters=100000, time_limit=15, rng=rng)
                _try_config(c_imp, s_imp, f"restart_{r}")
    
    # 5. Final SA polish on the best
    if best_centers is not None and time.time() - start_time < deadline - 10:
        remaining = deadline - 10 - (time.time() - start_time)
        c_final, s_final, val_final = _local_improve_sa(best_centers, best_sides, tol, max_iters=300000, time_limit=max(10, remaining), rng=rng)
        _try_config(c_final, s_final, "final_polish")
    
    # Fallback
    if best_centers is None:
        c, s = _heuristic_init_2_large_8_small()
        best_centers = c
        best_sides = s
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
