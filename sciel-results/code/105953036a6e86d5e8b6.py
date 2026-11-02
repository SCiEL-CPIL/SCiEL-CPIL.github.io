"""
Circle packing in perimeter-4 rectangle: Simulated Annealing + SLSQP Refinement.

Strategy:
1. Initialize with a grid layout.
2. Use Simulated Annealing to explore the space of (x, y, r, w).
   - Objective: Maximize sum(r) subject to constraints (no overlap, inside box).
   - Energy function: -sum(r) + lambda * (violation_penalty).
   - Moves: Random perturbations of positions, radii, and aspect ratio.
3. Refine the best found configuration using SLSQP (constrained optimization).
4. Polish with individual radius expansion.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
_TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _get_xyrw(z):
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w
    return x, y, r, w, h

def _compute_energy(z, lambda_overlap=10.0, lambda_bound=10.0):
    """
    Compute energy: -sum(r) + penalties.
    Lower energy is better.
    """
    x, y, r, w, h = _get_xyrw(z)
    
    # Objective: maximize sum(r) -> minimize -sum(r)
    obj = -np.sum(r)
    
    # Penalty for overlaps
    # d_ij < r_i + r_j -> violation
    # Use a smooth penalty: max(0, r_i+r_j-d_ij)^2
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx*dx + dy*dy) + 1e-10
    min_d = r[:, None] + r[None, :]
    
    # Only upper triangle to avoid double counting
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    overlaps = np.maximum(min_d - d, 0.0)
    overlap_penalty = np.sum(overlaps[mask] ** 2)
    
    # Penalty for boundary violations
    # x < r or x > w-r
    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)
    
    bound_penalty = (np.sum(left_viol**2) + np.sum(right_viol**2) + 
                     np.sum(bottom_viol**2) + np.sum(top_viol**2))
    
    # If box is too small, huge penalty
    if w < 0.2 or h < 0.2:
        return 1e10
        
    return obj + lambda_overlap * overlap_penalty + lambda_bound * bound_penalty

def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check validity of circles in a w x h rectangle."""
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(len(r)):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True

def _slsqp_refine(z, max_time=5.0):
    """
    Refine solution using SLSQP.
    """
    start = time.time()
    
    def objective(z):
        return _compute_energy(z, lambda_overlap=100.0, lambda_bound=100.0)
    
    def gradient(z):
        # Numerical gradient
        eps = 1e-5
        grad = np.zeros_like(z)
        for i in range(len(z)):
            z_plus = z.copy()
            z_minus = z.copy()
            z_plus[i] += eps
            z_minus[i] -= eps
            grad[i] = (objective(z_plus) - objective(z_minus)) / (2 * eps)
        return grad

    # Constraints:
    # 1. w + h = 2 is handled by variable definition (h = 2-w)
    # 2. w > 0.1, h > 0.1
    # 3. x_i >= r_i
    # 4. x_i <= w - r_i
    # 5. y_i >= r_i
    # 6. y_i <= h - r_i
    # 7. r_i > 0
    
    constraints = []
    # Box aspect ratio bounds
    constraints.append({'type': 'ineq', 'fun': lambda z: z[-1] - 0.2}) # w >= 0.2
    constraints.append({'type': 'ineq', 'fun': lambda z: (2.0 - z[-1]) - 0.2}) # h >= 0.2
    
    # Boundary constraints for each circle
    for i in range(N):
        idx_x = i
        idx_y = N + i
        idx_r = 2*N + i
        
        # x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[i] - z[2*N+i]})
        # w - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[-1] - z[i] - z[2*N+i]})
        # y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[N+i] - z[2*N+i]})
        # h - y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: (2.0 - z[-1]) - z[N+i] - z[2*N+i]})
        # r_i >= 0.01
        constraints.append({'type': 'ineq', 'fun': lambda z, i=i: z[2*N+i] - 0.01})

    # Overlap constraints: d_ij >= r_i + r_j
    # This is non-linear and SLSQP can handle it, but it's expensive.
    # Instead, we rely on the penalty in the objective for overlaps during SLSQP,
    # or we can add them as inequality constraints.
    # Let's add them as inequality constraints for strictness.
    for i in range(N):
        for j in range(i + 1, N):
            def overlap_con(z, i=i, j=j):
                dx = z[i] - z[j]
                dy = z[N+i] - z[N+j]
                d = np.sqrt(dx*dx + dy*dy)
                r_sum = z[2*N+i] + z[2*N+j]
                return d - r_sum
            constraints.append({'type': 'ineq', 'fun': overlap_con})

    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.01, None)) # r
    bounds.append((0.2, 1.8)) # w
    # x, y bounds are handled by constraints, but we need some bounds for SLSQP
    # x in [0, 2], y in [0, 2] is a safe loose bound
    for i in range(N):
        bounds.append((0.0, 2.0)) # x
    for i in range(N):
        bounds.append((0.0, 2.0)) # y
        
    # Reorder bounds to match z: [x1..xN, y1..yN, r1..rN, w]
    bounds_final = []
    for i in range(N):
        bounds_final.append((0.0, 2.0)) # x
    for i in range(N):
        bounds_final.append((0.0, 2.0)) # y
    for i in range(N):
        bounds_final.append((0.01, None)) # r
    bounds_final.append((0.2, 1.8)) # w

    try:
        res = minimize(objective, z, method='SLSQP', jac=gradient, 
                       bounds=bounds_final, constraints=constraints,
                       options={'maxiter': 100, 'ftol': 1e-8, 'disp': False})
        
        if res.success and res.fun < objective(z):
            return res.x
    except Exception:
        pass
        
    return z

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    best_value = 0.0
    best_z = None
    
    # Initial guess: Grid
    side = int(np.ceil(np.sqrt(N)))
    z_init = np.zeros(3 * N + 1)
    w0 = 1.0
    h0 = 1.0
    for i in range(N):
        row, col = divmod(i, side)
        cx = (col + 0.5) / side * w0
        cy = (row + 0.5) / side * h0
        r0 = 0.1
        z_init[3 * i] = cx
        z_init[3 * i + 1] = cy
        z_init[3 * i + 2] = r0
    z_init[-1] = w0
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.995
    z_current = z_init.copy()
    energy_current = _compute_energy(z_current)
    
    # Track best
    z_best_sa = z_current.copy()
    energy_best_sa = energy_current
    
    # Precompute indices for moves
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 20.0 and T > T_min:
        
        # Choose a move type
        move_type = rng.random()
        
        z_trial = z_current.copy()
        
        if move_type < 0.5:
            # Move a single circle position
            i = rng.integers(0, N)
            z_trial[3*i] += rng.normal(0, 0.05)
            z_trial[3*i+1] += rng.normal(0, 0.05)
        elif move_type < 0.7:
            # Move a single circle radius
            i = rng.integers(0, N)
            z_trial[3*i+2] += rng.normal(0, 0.02)
            if z_trial[3*i+2] < 0.01:
                z_trial[3*i+2] = 0.01
        elif move_type < 0.8:
            # Move aspect ratio
            z_trial[-1] += rng.normal(0, 0.05)
            if z_trial[-1] < 0.2 or z_trial[-1] > 1.8:
                z_trial[-1] = z_current[-1]
        else:
            # Global perturbation (restart-like)
            z_trial = z_current + rng.normal(0, 0.1, size=z_current.shape)
            z_trial[-1] = np.clip(z_trial[-1], 0.2, 1.8)
            
        # Clamp to box roughly to avoid huge penalties
        x = z_trial[0::3][:N]
        y = z_trial[1::3][:N]
        r = z_trial[2::3][:N]
        w = z_trial[-1]
        h = 2.0 - w
        
        # Soft clamp
        z_trial[0::3][:N] = np.clip(x, 0, w)
        z_trial[1::3][:N] = np.clip(y, 0, h)
        z_trial[2::3][:N] = np.clip(r, 0.01, min(w, h)/2)
        
        energy_trial = _compute_energy(z_trial)
        
        # Acceptance
        delta_E = energy_trial - energy_current
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            z_current = z_trial
            energy_current = energy_trial
            
            if energy_trial < energy_best_sa:
                energy_best_sa = energy_trial
                z_best_sa = z_trial.copy()
                
                # Check validity for best tracking
                x, y, r, w, h = _get_xyrw(z_best_sa)
                if _is_valid(x, y, r, w, h):
                    val = np.sum(r)
                    if val > best_value:
                        best_value = val
                        best_z = z_best_sa.copy()
                        
        T *= cooling
        
    # Refine best SA result with SLSQP
    if best_z is not None and time.time() < deadline - 10.0:
        z_refined = _slsqp_refine(best_z, max_time=10.0)
        x, y, r, w, h = _get_xyrw(z_refined)
        if _is_valid(x, y, r, w, h):
            val = np.sum(r)
            if val > best_value:
                best_value = val
                best_z = z_refined

    # Final polish: individual expansion
    if best_z is not None and time.time() < deadline - 2.0:
        z = best_z.copy()
        n = N
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            improved = False
            order = rng.permutation(n)
            for i in order:
                r_i = z[2*i+2]
                x_i = z[2*i]
                y_i = z[2*i+1]
                w = z[-1]
                h = 2.0 - w
                
                new_r_i = r_i + 0.001
                if new_r_i > w/2 or new_r_i > h/2:
                    continue
                
                if new_r_i > x_i or x_i + new_r_i > w or new_r_i > y_i or y_i + new_r_i > h:
                    continue
                
                dx = x_i - z[0::3][:n]
                dy = y_i - z[1::3][:n]
                d = np.sqrt(dx*dx + dy*dy)
                min_d = new_r_i + z[2::3][:n]
                min_d[i] = np.inf
                
                if np.any(d < min_d - 1e-9):
                    continue
                
                z[2*i+2] = new_r_i
                improved = True
                break
                
            if not improved:
                break
                
        x, y, r, w, h = _get_xyrw(z)
        if _is_valid(x, y, r, w, h):
            val = np.sum(r)
            if val > best_value:
                best_value = val
                best_z = z

    if best_z is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_z[0::3][:N]
    y = best_z[1::3][:N]
    r = best_z[2::3][:N]
    
    return np.column_stack((x, y, r))

if __name__ == "__main__":
    res = construct()
    print(res.shape)
    print(np.sum(res[:, 2]))
