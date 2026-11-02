"""Circle packing in unit square: 26 circles, maximize sum of radii.
Strategy: Physics-based soft-body relaxation with multi-scale annealing.
Uses elastic potentials for overlaps and wall constraints to guide a global
search, followed by a precise SLSQP polish on the contact graph.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
EPS = 1e-9
DEADLINE = 145.0  # Safety margin from 150s limit

def _check_validity(circles, tol=1e-7):
    """Check if circles are valid (inside square, non-overlapping)."""
    n = len(circles)
    s = 0.0
    for i in range(n):
        x, y, r = circles[i]
        if r < 0:
            return False, 0.0
        if x < r + tol or x > 1 - r - tol or y < r + tol or y > 1 - r - tol:
            return False, 0.0
        s += r
        for j in range(i + 1, n):
            x2, y2, r2 = circles[j]
            d2 = (x - x2) ** 2 + (y - y2) ** 2
            if d2 < (r + r2) ** 2 - tol:
                return False, 0.0
    return True, s

def _soft_energy(pos, r, k_overlap, k_wall):
    """Calculate elastic potential energy and gradients for soft-body simulation.
    Returns energy, grad_pos, grad_r.
    """
    n = len(pos)
    energy = 0.0
    grad_pos = np.zeros_like(pos)
    grad_r = np.zeros_like(r)
    
    # Wall constraints
    for i in range(n):
        # Left wall: pos[i,0] - r[i] >= 0
        viol = r[i] - pos[i, 0]
        if viol > 0:
            energy += 0.5 * k_wall * viol * viol
            grad_pos[i, 0] -= k_wall * viol
            grad_r[i] += k_wall * viol
            
        # Right wall: (1 - pos[i,0]) - r[i] >= 0
        viol = r[i] - (1 - pos[i, 0])
        if viol > 0:
            energy += 0.5 * k_wall * viol * viol
            grad_pos[i, 0] += k_wall * viol
            grad_r[i] += k_wall * viol
            
        # Bottom wall: pos[i,1] - r[i] >= 0
        viol = r[i] - pos[i, 1]
        if viol > 0:
            energy += 0.5 * k_wall * viol * viol
            grad_pos[i, 1] -= k_wall * viol
            grad_r[i] += k_wall * viol
            
        # Top wall: (1 - pos[i,1]) - r[i] >= 0
        viol = r[i] - (1 - pos[i, 1])
        if viol > 0:
            energy += 0.5 * k_wall * viol * viol
            grad_pos[i, 1] += k_wall * viol
            grad_r[i] += k_wall * viol
            
    # Pair overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.hypot(dx, dy)
            if d < 1e-12:
                d = 1e-12
                dx, dy = 1e-6, 0.0
            viol = r[i] + r[j] - d
            if viol > 0:
                energy += 0.5 * k_overlap * viol * viol
                f = k_overlap * viol / d
                grad_pos[i, 0] += f * dx
                grad_pos[i, 1] += f * dy
                grad_pos[j, 0] -= f * dx
                grad_pos[j, 1] -= f * dy
                grad_r[i] += k_overlap * viol
                grad_r[j] += k_overlap * viol
                
    return energy, grad_pos, grad_r

def _soft_relax(pos, r, time_budget=5.0):
    """Run soft-body relaxation with simulated annealing."""
    t_start = time.time()
    pos = pos.copy()
    r = r.copy()
    
    # Parameters
    k_overlap = 100.0
    k_wall = 500.0
    step_size = 0.05
    temp = 0.05
    cooling = 0.995
    
    # Pre-calculate initial energy
    e0, gp0, gr0 = _soft_energy(pos, r, k_overlap, k_wall)
    
    max_iters = 5000
    for it in range(max_iters):
        if time.time() - t_start > time_budget:
            break
            
        e, gp, gr = _soft_energy(pos, r, k_overlap, k_wall)
        
        # Gradient step
        pos_new = pos + step_size * gp
        r_new = r + step_size * gr
        
        # Add noise for annealing
        if temp > 1e-4:
            noise_pos = np.random.normal(0, temp * 0.1, pos.shape)
            noise_r = np.random.normal(0, temp * 0.05, r.shape)
            pos_new += noise_pos
            r_new += noise_r
            
        # Clip positions and radii to reasonable bounds
        pos_new = np.clip(pos_new, 0.0, 1.0)
        r_new = np.clip(r_new, 0.0, 0.5)
        
        # Calculate new energy
        e_new, _, _ = _soft_energy(pos_new, r_new, k_overlap, k_wall)
        
        # Metropolis criterion
        delta_e = e_new - e
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / (temp + 1e-9)):
            pos = pos_new
            r = r_new
            e = e_new
        else:
            # Keep old state but maybe try a smaller step? For simplicity, just reject.
            pass
            
        temp *= cooling
        
    return pos, r

def _polish_slsqp(pos, r, time_budget=5.0):
    """Final polish using SLSQP on the contact graph."""
    t_start = time.time()
    
    # Identify active constraints (contact graph)
    active_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            if d <= r[i] + r[j] + 1e-4:
                active_pairs.append((i, j))
                
    active_walls = []
    for i in range(N):
        if pos[i, 0] <= r[i] + 1e-4 or pos[i, 0] >= 1 - r[i] - 1e-4 or \
           pos[i, 1] <= r[i] + 1e-4 or pos[i, 1] >= 1 - r[i] - 1e-4:
            active_walls.append(i)

    # Variables: [x0, y0, ..., xN, yN, r0, ..., rN]
    vars0 = np.zeros(4 * N)
    vars0[:2 * N] = pos.flatten()
    vars0[2 * N:] = r
    
    def objective(v):
        x = v[:2 * N].reshape(N, 2)
        rad = v[2 * N:]
        if np.any(rad < 0): return 1e6
        # Check walls
        for i in range(N):
            if x[i, 0] < rad[i] or x[i, 0] > 1 - rad[i] or x[i, 1] < rad[i] or x[i, 1] > 1 - rad[i]:
                return 1e6
        # Check active pairs
        for (i, j) in active_pairs:
            d = np.hypot(x[i, 0] - x[j, 0], x[i, 1] - x[j, 1])
            if d < rad[i] + rad[j] - 1e-9:
                return 1e6
        return -np.sum(rad)
        
    # Constraints
    constraints = []
    for i in range(N):
        # x[i,0] - r[i] >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i] - v[2*N+i]})
        # x[i,1] - r[i] >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i+1] - v[2*N+i]})
        # (1-x[i,0]) - r[i] >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: (1 - v[2*i]) - v[2*N+i]})
        # (1-x[i,1]) - r[i] >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: (1 - v[2*i+1]) - v[2*N+i]})
        
    for (i, j) in active_pairs:
        constraints.append({
            'type': 'ineq', 
            'fun': lambda v, i=i, j=j: np.hypot(v[2*i]-v[2*j], v[2*i+1]-v[2*j+1]) - v[2*N+i] - v[2*N+j]
        })
        
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))
        
    try:
        res = minimize(objective, vars0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': 100, 'ftol': 1e-10})
        if res.success:
            new_pos = res.x[:2*N].reshape(N, 2)
            new_r = res.x[2*N:]
            return new_pos, new_r
    except Exception:
        pass
        
    return pos, r

def construct():
    t_start = time.time()
    best_score = -1.0
    best_circles = None
    
    def update_best(pos, r):
        nonlocal best_score, best_circles
        circles = list(zip(pos[:, 0], pos[:, 1], r))
        valid, s = _check_validity(circles)
        if valid and s > best_score:
            best_score = s
            best_circles = [tuple(c) for c in circles]
            
    # Initializations
    inits = []
    # 1. Grid
    side = 5
    cells = []
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N: break
            cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side * 0.8))
            idx += 1
    if len(cells) >= N: inits.append(cells[:N])
    
    # 2. Hexagonal
    pts = []
    y = 0.1
    row = 0
    while len(pts) < N and y < 0.9:
        x = 0.1 + (row % 2) * 0.08
        while x < 0.9 and len(pts) < N:
            pts.append((x, y, 0.08))
            x += 0.16
        y += 0.14
        row += 1
    if len(pts) >= N: inits.append(pts[:N])
    
    # 3. Random
    rng = np.random.default_rng(123)
    for _ in range(3):
        pts = []
        for _ in range(N):
            x = 0.1 + 0.8 * rng.random()
            y = 0.1 + 0.8 * rng.random()
            r = 0.05 + 0.05 * rng.random()
            pts.append((x, y, r))
        inits.append(pts)
        
    for init in inits:
        if time.time() > DEADLINE - 10: break
        
        pos = np.array([(p[0], p[1]) for p in init], dtype=float)
        r = np.array([p[2] for p in init], dtype=float)
        
        # Phase 1: Soft relaxation (Global search)
        # Use multiple "restarts" of the soft relaxation with different noise scales
        for restart in range(3):
            if time.time() > DEADLINE - 5: break
            # Jitter to escape local minima in the soft landscape
            if restart > 0:
                pos = pos + np.random.normal(0, 0.05, pos.shape)
                pos = np.clip(pos, 0, 1)
                
            pos, r = _soft_relax(pos, r, time_budget=3.0)
            circles = list(zip(pos[:, 0], pos[:, 1], r))
            update_best(pos, r)
            
            # Phase 2: SLSQP Polish (Local refinement)
            if time.time() < DEADLINE - 2:
                pos_polished, r_polished = _polish_slsqp(pos, r, time_budget=2.0)
                update_best(pos_polished, r_polished)
                # Update state for next restart if polished is better
                if _check_validity(list(zip(pos_polished[:, 0], pos_polished[:, 1], r_polished)))[1] > _check_validity(list(zip(pos[:, 0], pos[:, 1], r)))[1]:
                    pos, r = pos_polished, r_polished

    if best_circles is None:
        # Fallback
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return np.array(best_circles)

if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
