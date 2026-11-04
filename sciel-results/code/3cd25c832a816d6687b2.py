"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
def _smooth_max(x, beta=100.0):
    """Soft-max approximation of max(x) using log-sum-exp."""
    m = np.max(x)
    if m < -50:
        return 0.0
    return m + np.log(np.sum(np.exp(x - m))) / beta


def _objective_and_grad(z, beta=100.0):
    """
    Objective: maximize sum of radii.
    z = [x_0..x_25, y_0..y_25, r_0..r_25]
    Constraints handled via penalty in objective.
    Returns (neg_objective, gradient).
    """
    n = N
    xs = z[:n]
    ys = z[n:2*n]
    rs = z[2*n:3*n]

    # Penalty for boundary violations
    boundary_viol = np.maximum(0, xs + rs - 1.0) + np.maximum(0, -xs + rs) + \
                    np.maximum(0, ys + rs - 1.0) + np.maximum(0, -ys + rs)
    # Penalty for overlap: for each pair, need dist >= r_i + r_j
    # We penalize when dist < r_i + r_j
    overlap_pen = 0.0
    overlap_grad = np.zeros(3*n)
    
    # Use a smooth penalty
    pen_coeff = 1000.0
    
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq + 1e-12)
            min_dist = rs[i] + rs[j]
            if dist < min_dist:
                # Penalty: (min_dist - dist)^2
                diff = min_dist - dist
                overlap_pen += pen_coeff * diff * diff
                # Gradient of penalty w.r.t. positions
                # d/dx_i of diff^2 = 2*diff * d/dx_i(min_dist - dist)
                # d/dx_i dist = dx / dist
                if dist > 1e-10:
                    factor = 2 * pen_coeff * diff / dist
                    overlap_grad[i] += factor * dx
                    overlap_grad[j] -= factor * dx
                    overlap_grad[n+i] += factor * dy
                    overlap_grad[n+j] -= factor * dy
                # Gradient w.r.t. radii
                # d/dr_i diff = -1
                overlap_grad[2*n+i] += -2 * pen_coeff * diff
                overlap_grad[2*n+j] += -2 * pen_coeff * diff
    
    # Boundary gradients
    bound_grad = np.zeros(3*n)
    # For x_i + r_i > 1: gradient w.r.t x_i is 1, w.r.t r_i is 1
    # For x_i - r_i < 0: gradient w.r.t x_i is 1, w.r.t r_i is -1
    for i in range(n):
        if xs[i] + rs[i] > 1.0:
            bound_grad[i] += pen_coeff
            bound_grad[2*n+i] += pen_coeff
        if xs[i] - rs[i] < 0.0:
            bound_grad[i] += pen_coeff
            bound_grad[2*n+i] -= pen_coeff
        if ys[i] + rs[i] > 1.0:
            bound_grad[n+i] += pen_coeff
            bound_grad[2*n+i] += pen_coeff
        if ys[i] - rs[i] < 0.0:
            bound_grad[n+i] += pen_coeff
            bound_grad[2*n+i] -= pen_coeff
    
    total_pen = boundary_viol.sum() * pen_coeff + overlap_pen
    # Objective: maximize sum(r) -> minimize -sum(r) + penalty
    obj_val = -np.sum(rs) + total_pen
    grad = -np.ones(3*n)
    grad[:n] += bound_grad[:n] + overlap_grad[:n]
    grad[n:2*n] += bound_grad[n:2*n] + overlap_grad[n:2*n]
    grad[2*n:3*n] += bound_grad[2*n:3*n] + overlap_grad[2*n:3*n]
    
    return obj_val, grad


def _check_validity(z):
    """Check if the packing is valid (all circles inside unit square, no overlaps)."""
    n = N
    xs = z[:n]
    ys = z[n:2*n]
    rs = z[2*n:3*n]
    if np.any(rs <= 0):
        return False
    if np.any(xs + rs > 1.0 + 1e-9) or np.any(xs - rs < -1e-9):
        return False
    if np.any(ys + rs > 1.0 + 1e-9) or np.any(ys - rs < -1e-9):
        return False
    for i in range(n):
        for j in range(i+1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rs[i] + rs[j] - 1e-9:
                return False
    return True


def _grid_init():
    """Create a grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.001
                cells.append([x, y, r])
    return np.array(cells).flatten()


def _random_init():
    """Create a random initialization."""
    z = np.zeros(3*N)
    # Random positions and small radii
    z[:N] = np.random.rand(N) * 0.8 + 0.1
    z[N:2*N] = np.random.rand(N) * 0.8 + 0.1
    z[2*N:3*N] = np.random.rand(N) * 0.05 + 0.02
    return z


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_z = None
    best_score = -np.inf
    
    # Try multiple initializations
    inits = [_grid_init()]
    for _ in range(5):
        inits.append(_random_init())
    
    for init in inits:
        if time.time() > deadline:
            break
        
        # Multi-stage optimization with increasing penalty
        z = init.copy()
        for stage, pen_coeff in enumerate([100.0, 1000.0, 5000.0, 20000.0]):
            if time.time() > deadline:
                break
            
            # Redefine objective with current penalty
            def obj_pen(z, pc=pen_coeff):
                n = N
                xs = z[:n]
                ys = z[n:2*n]
                rs = z[2*n:3*n]
                
                # Boundary penalty
                bv = np.maximum(0, xs + rs - 1.0) + np.maximum(0, -xs + rs) + \
                     np.maximum(0, ys + rs - 1.0) + np.maximum(0, -ys + rs)
                
                # Overlap penalty
                op = 0.0
                for i in range(n):
                    for j in range(i+1, n):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                        min_dist = rs[i] + rs[j]
                        if dist < min_dist:
                            diff = min_dist - dist
                            op += pc * diff * diff
                
                return -np.sum(rs) + bv.sum() * pc + op
            
            def grad_pen(z, pc=pen_coeff):
                n = N
                xs = z[:n]
                ys = z[n:2*n]
                rs = z[2*n:3*n]
                
                grad = -np.ones(3*n)
                
                # Boundary gradients
                for i in range(n):
                    if xs[i] + rs[i] > 1.0:
                        grad[i] += pc
                        grad[2*n+i] += pc
                    if xs[i] - rs[i] < 0.0:
                        grad[i] += pc
                        grad[2*n+i] -= pc
                    if ys[i] + rs[i] > 1.0:
                        grad[n+i] += pc
                        grad[2*n+i] += pc
                    if ys[i] - rs[i] < 0.0:
                        grad[n+i] += pc
                        grad[2*n+i] -= pc
                
                # Overlap gradients
                for i in range(n):
                    for j in range(i+1, n):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                        min_dist = rs[i] + rs[j]
                        if dist < min_dist and dist > 1e-10:
                            diff = min_dist - dist
                            factor = 2 * pc * diff / dist
                            grad[i] += factor * dx
                            grad[j] -= factor * dx
                            grad[n+i] += factor * dy
                            grad[n+j] -= factor * dy
                            grad[2*n+i] += -2 * pc * diff
                            grad[2*n+j] += -2 * pc * diff
                
                return grad
            
            try:
                result = minimize(
                    obj_pen, z, jac=grad_pen, method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
                )
                z = result.x
            except Exception:
                pass
        
        # Check validity and score
        if _check_validity(z):
            score = np.sum(z[2*N:3*N])
            if score > best_score:
                best_score = score
                best_z = z.copy()
        
        # Also check if we can improve by allowing small tolerance
        xs = z[:N]
        ys = z[N:2*N]
        rs = z[2*N:3*N]
        # Try to see if a slightly perturbed version is valid
        for trial in range(20):
            if time.time() > deadline:
                break
            z_trial = z.copy()
            z_trial[:N] += np.random.randn(N) * 0.001
            z_trial[N:2*N] += np.random.randn(N) * 0.001
            if _check_validity(z_trial):
                score = np.sum(z_trial[2*N:3*N])
                if score > best_score:
                    best_score = score
                    best_z = z_trial.copy()
    
    if best_z is None:
        # Fallback to grid
        z = _grid_init()
        best_z = z
    
    # Final polish: try to increase radii slightly
    if time.time() < deadline - 10:
        for _ in range(50):
            if time.time() > deadline:
                break
            z_trial = best_z.copy()
            # Try increasing all radii by a small amount
            z_trial[2*N:3*N] *= (1.0 + 0.001)
            # Then optimize to fix violations
            try:
                def obj_final(z):
                    n = N
                    xs = z[:n]
                    ys = z[n:2*n]
                    rs = z[2*n:3*n]
                    bv = np.maximum(0, xs + rs - 1.0) + np.maximum(0, -xs + rs) + \
                         np.maximum(0, ys + rs - 1.0) + np.maximum(0, -ys + rs)
                    op = 0.0
                    for i in range(n):
                        for j in range(i+1, n):
                            dx = xs[i] - xs[j]
                            dy = ys[i] - ys[j]
                            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                            min_dist = rs[i] + rs[j]
                            if dist < min_dist:
                                diff = min_dist - dist
                                op += 50000.0 * diff * diff
                    return -np.sum(rs) + bv.sum() * 50000.0 + op
                
                def grad_final(z):
                    n = N
                    xs = z[:n]
                    ys = z[n:2*n]
                    rs = z[2*n:3*n]
                    grad = -np.ones(3*n)
                    pc = 50000.0
                    for i in range(n):
                        if xs[i] + rs[i] > 1.0:
                            grad[i] += pc
                            grad[2*n+i] += pc
                        if xs[i] - rs[i] < 0.0:
                            grad[i] += pc
                            grad[2*n+i] -= pc
                        if ys[i] + rs[i] > 1.0:
                            grad[n+i] += pc
                            grad[2*n+i] += pc
                        if ys[i] - rs[i] < 0.0:
                            grad[n+i] += pc
                            grad[2*n+i] -= pc
                    for i in range(n):
                        for j in range(i+1, n):
                            dx = xs[i] - xs[j]
                            dy = ys[i] - ys[j]
                            dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                            min_dist = rs[i] + rs[j]
                            if dist < min_dist and dist > 1e-10:
                                diff = min_dist - dist
                                factor = 2 * pc * diff / dist
                                grad[i] += factor * dx
                                grad[j] -= factor * dx
                                grad[n+i] += factor * dy
                                grad[n+j] -= factor * dy
                                grad[2*n+i] += -2 * pc * diff
                                grad[2*n+j] += -2 * pc * diff
                    return grad
                
                result = minimize(obj_final, z_trial, jac=grad_final, method='L-BFGS-B',
                                  options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                if _check_validity(result.x):
                    score = np.sum(result.x[2*N:3*N])
                    if score > best_score:
                        best_score = score
                        best_z = result.x.copy()
            except Exception:
                pass
    
    # Format output
    result = np.zeros((N, 3))
    result[:, 0] = best_z[:N]
    result[:, 1] = best_z[N:2*N]
    result[:, 2] = best_z[2*N:3*N]
    return result
# EVOLVE-BLOCK-END
