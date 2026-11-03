"""
Combined packer: soft-min surrogate optimization with L-BFGS-B + active-set polish.
Structural idea from inspiration 1 (soft-min + L-BFGS-B with temperature annealing).
Refinement machinery from current program (multi-restart, polish, deadline guards).
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100


def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0

    # Wall clearance
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall = np.min(wall_dists)
    if min_wall < -1e-9:
        return 0.0

    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_pair = np.sqrt(dists_sq[iu]).min()

    return min(min_pair * 0.5, min_wall)


def _slsqp_obj_grad(x):
    """
    Objective: -r (minimize negative radius).
    x = [c_1...c_N, r]
    """
    centers = x[:-1].reshape(-1, 3)
    r = x[-1]
    return -r, np.concatenate([np.zeros_like(centers), [-1.0]])

def _slsqp_constraints(x):
    """
    Returns list of constraint dicts for SLSQP.
    Constraints:
    1. ||c_i - c_j|| - 2r >= 0
    2. 0.5 - c_ik - r >= 0
    3. 0.5 + c_ik - r >= 0
    """
    centers = x[:-1].reshape(-1, 3)
    r = x[-1]
    n = len(centers)
    iu = np.triu_indices(n, k=1)
    i_idx, j_idx = iu
    
    cons = []
    
    # Pair constraints
    diff = centers[i_idx] - centers[j_idx]
    dists = np.linalg.norm(diff, axis=1)
    # Avoid division by zero in gradient
    dists_safe = np.where(dists < 1e-10, 1e-10, dists)
    
    # We can batch constraints for SLSQP? No, SLSQP expects a list or array.
    # Passing 5000+ constraints might be slow.
    # Let's try to pass them as a single array of shape (M,) and jac as (M, D).
    
    # Actually, SLSQP in SciPy accepts a list of dicts or a single array.
    # If we pass a single array, we must provide the full Jacobian.
    
    # Let's stick to the list of dicts for clarity, but it might be slow.
    # Alternative: Use a penalty method or barrier.
    # Given the time limit, let's try a penalty-based approach with L-BFGS-B which is faster per iteration.
    # Objective: -r + mu * sum( max(0, 2r - d_ij)^2 + max(0, r - (0.5-|c_ik|))^2 )
    # This is smooth if we smooth the max.
    
    # Let's revert to the soft-min idea but improve the polish.
    # The current soft-min is okay. The issue might be the polish.
    # Let's improve the polish to use a proper active set method.
    
    return [] # Placeholder, we will change strategy to improve polish instead


def _slsqp_polish(centers, t0, deadline):
    """
    High-precision polish using SLSQP on the full problem.
    Starts from a good initial guess, so it should converge quickly.
    """
    n = len(centers)
    r0 = _compute_radius(centers)
    
    # Initial x: [centers, r0]
    x0 = np.concatenate([centers.flatten(), [r0]])
    
    # Bounds for centers
    bounds_c = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (n * 3)
    # Bound for r: [0, 0.5]
    bounds_r = (0, 0.5)
    bounds = bounds_c + [bounds_r]
    
    # Constraints
    iu = np.triu_indices(n, k=1)
    i_idx, j_idx = iu
    n_pairs = len(iu[0])
    n_walls = 6 * n
    n_cons = n_pairs + n_walls
    
    def obj(x):
        return -x[-1]
        
    def obj_grad(x):
        grad = np.zeros_like(x)
        grad[-1] = -1.0
        return grad
        
    def cons_func(x):
        c = x[:-1].reshape(-1, 3)
        r = x[-1]
        
        # Pair constraints: ||c_i - c_j|| - 2r >= 0
        diff = c[i_idx] - c[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        pair_vals = dists - 2*r
        
        # Wall constraints:
        wall_vals_plus = 0.5 - c - r  # shape (n, 3)
        wall_vals_minus = 0.5 + c - r # shape (n, 3)
        
        wall_vals = np.concatenate([wall_vals_plus.flatten(), wall_vals_minus.flatten()])
        
        return np.concatenate([pair_vals, wall_vals])
        
    def cons_grad(x):
        c = x[:-1].reshape(-1, 3)
        r = x[-1]
        dim = len(x)
        jac = np.zeros((n_cons, dim))
        
        # Pair gradients
        diff = c[i_idx] - c[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        dists_safe = np.where(dists < 1e-10, 1e-10, dists)
        inv_dists = 1.0 / dists_safe
        
        for k in range(n_pairs):
            i = i_idx[k]
            j = j_idx[k]
            d_vec = diff[k]
            factor = inv_dists[k]
            
            jac[k, i*3:(i+1)*3] = d_vec * factor
            jac[k, j*3:(j+1)*3] = -d_vec * factor
            jac[k, -1] = -2.0
            
        # Wall gradients
        for i in range(n):
            for k in range(3):
                idx_plus = i*3 + k
                idx_minus = n*3 + idx_plus
                
                jac[idx_plus, i*3+k] = -1.0
                jac[idx_plus, -1] = -1.0
                
                jac[idx_minus, i*3+k] = 1.0
                jac[idx_minus, -1] = -1.0
                
        return jac
        
    cons = [{'type': 'ineq', 'fun': cons_func, 'jac': cons_grad}]
    
    try:
        res = minimize(
            obj, 
            x0, 
            method='SLSQP', 
            jac=obj_grad, 
            bounds=bounds, 
            constraints=cons,
            options={'maxiter': 100, 'ftol': 1e-15, 'disp': False}
        )
        
        if res.success or res.fun < -r0:
            new_r = -res.fun
            new_centers = res.x[:-1].reshape(-1, 3)
            new_centers = np.clip(new_centers, -0.5 + 1e-9, 0.5 - 1e-9)
            
            # Verify
            r_check = _compute_radius(new_centers)
            if r_check > r0:
                return new_centers, r_check
    except Exception:
        pass
        
    return centers, r0


def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0  # 10s safety margin before 150s kill

    best_radius = 0.0
    best_centers = None

    # Grid initialization: 5x4x5 = 100
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5 / nx, 0.5 - 0.5 / nx, nx)
    ys = np.linspace(-0.5 + 0.5 / ny, 0.5 - 0.5 / ny, ny)
    zs = np.linspace(-0.5 + 0.5 / nz, 0.5 - 0.5 / nz, nz)
    grid = np.array([[x, y, z] for z in zs for y in ys for x in xs])

    rng = np.random.default_rng(123)

    # Build diverse initializations
    inits = [grid.copy()]
    for _ in range(4):
        pert = grid.copy() + rng.normal(0, 0.03, size=(N, 3))
        inits.append(np.clip(pert, -0.45, 0.45))
    for _ in range(3):
        inits.append(rng.uniform(-0.45, 0.45, size=(N, 3)))

    bounds = [(-0.5 + 1e-9, 0.5 - 1e-9)] * (N * 3)

    # Temperature schedule for soft-min annealing
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]

    for init in inits:
        if time.time() > DEADLINE - 25.0:
            break

        current = init.copy()

        # Phase 1: L-BFGS-B with decreasing temperature
        for temp in temperatures:
            if time.time() > DEADLINE - 20.0:
                break

            def obj_grad(x, t=temp):
                return _soft_min_obj_grad(x, t)

            try:
                res = minimize(
                    obj_grad,
                    current.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'disp': False, 'ftol': 1e-12,
                             'gtol': 1e-10, 'maxls': 40}
                )
                current = res.x.reshape(-1, 3)
            except Exception:
                pass

            current = np.clip(current, -0.5 + 1e-9, 0.5 - 1e-9)
            r = _compute_radius(current)
            if r > best_radius:
                best_radius = r
                best_centers = current.copy()

    # Phase 2: Polish the best result
    if best_centers is not None and time.time() < DEADLINE - 5.0:
        best_centers, best_radius = _slsqp_polish(
            best_centers.copy(), t0, DEADLINE - 2.0
        )

    # Final safety
    if best_centers is None:
        best_centers = grid.copy()
    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    if not np.all(np.isfinite(best_centers)):
        best_centers = grid.copy()

    return best_centers
