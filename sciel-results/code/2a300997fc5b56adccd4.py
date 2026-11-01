"""Circle packing in unit square n=32 using gradient-based optimization with constraints."""
N = 32
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

# EVOLVE-BLOCK-START

def _hexagonal_init(n):
    """Generate a hexagonal packing initial configuration."""
    # For n=32, use a hexagonal grid arrangement
    # Hexagonal packing: rows offset by half a spacing
    # Try to fit as many as possible in a hexagonal pattern
    
    # Approximate hexagonal packing for 32 circles
    # Use a 6x6 grid with some removed, or 5 rows
    # Let's use a structured approach:
    # Row 0: 6 circles, Row 1: 5 circles, Row 2: 6 circles, Row 3: 5 circles, Row 4: 6 circles, Row 5: 4 circles
    # Total: 6+5+6+5+6+4 = 32
    
    rows = [6, 5, 6, 5, 6, 4]
    circles = []
    y_spacing = 1.0 / (len(rows) + 0.5)
    x_spacing_base = 1.0 / 7.0  # approximate
    
    for i, count in enumerate(rows):
        y = (i + 0.5) * y_spacing
        if i % 2 == 0:
            # Even row: centered
            xs = np.linspace(0.5, 0.5, count) if count == 1 else np.linspace(0.5, 0.5, count)
            # Better: distribute evenly
            if count > 1:
                xs = np.linspace(0.5, 0.5, count)  # placeholder
                xs = np.array([(j + 0.5) / count for j in range(count)])
            else:
                xs = np.array([0.5])
        else:
            # Odd row: offset
            if count > 1:
                xs = np.array([(j + 0.5) / count for j in range(count)])
            else:
                xs = np.array([0.5])
        
        for x in xs:
            circles.append([x, y, 0.05])  # initial radius
    
    return np.array(circles[:n])


def _objective_and_constraints(x, n):
    """
    x: array of shape (3n,) = [x_1, y_1, r_1, x_2, y_2, r_2, ..., x_n, y_n, r_n]
    Maximize sum of radii = minimize -sum(r_i)
    
    Constraints:
    1. x_i >= r_i, x_i <= 1 - r_i, y_i >= r_i, y_i <= 1 - r_i
    2. (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 for all i < j
    3. r_i > 0
    """
    pos = x.reshape(n, 3)
    xs = pos[:, 0]
    ys = pos[:, 1]
    rs = pos[:, 2]
    
    # Objective: minimize -sum(r)
    obj = -np.sum(rs)
    
    # Inequality constraints (g(x) >= 0):
    # Boundary: x_i - r_i >= 0, 1 - x_i - r_i >= 0, y_i - r_i >= 0, 1 - y_i - r_i >= 0
    # Pairwise: (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
    # Radius: r_i >= epsilon
    
    return obj, pos


def _build_constraints(n, eps=1e-6):
    """Build constraint functions for SLSQP."""
    
    def boundary_constraints(x):
        pos = x.reshape(n, 3)
        xs = pos[:, 0]
        ys = pos[:, 1]
        rs = pos[:, 2]
        
        c = np.concatenate([
            xs - rs,       # x_i >= r_i
            1 - xs - rs,   # x_i <= 1 - r_i
            ys - rs,       # y_i >= r_i
            1 - ys - rs,   # y_i <= 1 - r_i
            rs - eps       # r_i >= eps
        ])
        return c
    
    def pairwise_constraints(x):
        pos = x.reshape(n, 3)
        xs = pos[:, 0]
        ys = pos[:, 1]
        rs = pos[:, 2]
        
        c = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dr = rs[i] + rs[j]
                c.append(dx*dx + dy*dy - dr*dr)
        return np.array(c)
    
    return boundary_constraints, pairwise_constraints


def _gradient_objective(x, n):
    """Gradient of -sum(r_i)."""
    g = np.zeros(3 * n)
    for i in range(n):
        g[3*i + 2] = -1.0  # d(-r_i)/dr_i = -1
    return g


def _gradient_boundary(x, n, eps=1e-6):
    """Jacobian of boundary constraints."""
    pos = x.reshape(n, 3)
    xs = pos[:, 0]
    ys = pos[:, 1]
    rs = pos[:, 2]
    
    # 5n constraints: [x_i - r_i, 1 - x_i - r_i, y_i - r_i, 1 - y_i - r_i, r_i - eps]
    J = np.zeros((5 * n, 3 * n))
    for i in range(n):
        base = i * 3
        # x_i - r_i: dx=1, dy=0, dr=-1
        J[5*i, base] = 1.0
        J[5*i, base+2] = -1.0
        # 1 - x_i - r_i: dx=-1, dy=0, dr=-1
        J[5*i+1, base] = -1.0
        J[5*i+1, base+2] = -1.0
        # y_i - r_i: dx=0, dy=1, dr=-1
        J[5*i+2, base+1] = 1.0
        J[5*i+2, base+2] = -1.0
        # 1 - y_i - r_i: dx=0, dy=-1, dr=-1
        J[5*i+3, base+1] = -1.0
        J[5*i+3, base+2] = -1.0
        # r_i - eps: dx=0, dy=0, dr=1
        J[5*i+4, base+2] = 1.0
    
    return J


def _gradient_pairwise(x, n):
    """Jacobian of pairwise distance constraints."""
    pos = x.reshape(n, 3)
    xs = pos[:, 0]
    ys = pos[:, 1]
    rs = pos[:, 2]
    
    num_pairs = n * (n - 1) // 2
    J = np.zeros((num_pairs, 3 * n))
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dr = rs[i] + rs[j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(max(dist_sq, 1e-12))
            
            # Constraint: dist_sq - dr^2 >= 0
            # d/dx_i = 2*dx, d/dx_j = -2*dx
            # d/dy_i = 2*dy, d/dy_j = -2*dy
            # d/dr_i = -2*dr, d/dr_j = -2*dr
            
            J[idx, 3*i] = 2 * dx
            J[idx, 3*i+1] = 2 * dy
            J[idx, 3*i+2] = -2 * dr
            J[idx, 3*j] = -2 * dx
            J[idx, 3*j+1] = -2 * dy
            J[idx, 3*j+2] = -2 * dr
            idx += 1
    
    return J


def _solve_single(start_x, n, maxiter=2000, tol=1e-8):
    """Solve the optimization problem from a single starting point."""
    boundary_c, pairwise_c = _build_constraints(n)
    
    cons = [
        {'type': 'ineq', 'fun': boundary_c, 'jac': _gradient_boundary},
        {'type': 'ineq', 'fun': pairwise_c, 'jac': _gradient_pairwise}
    ]
    
    bounds = [(0, 1) for _ in range(n)] + [(0, 1) for _ in range(n)] + [(1e-6, 0.5) for _ in range(n)]
    
    result = minimize(
        fun=_objective_and_constraints,
        x0=start_x,
        args=(n,),
        method='SLSQP',
        jac=_gradient_objective,
        bounds=bounds,
        constraints=cons,
        options={'maxiter': maxiter, 'ftol': tol, 'disp': False}
    )
    
    return result


def _check_validity(pos, n, tol=1e-6):
    """Check if the solution is valid (all constraints satisfied)."""
    xs = pos[:, 0]
    ys = pos[:, 1]
    rs = pos[:, 2]
    
    # Boundary check
    if np.any(xs < rs - tol) or np.any(xs > 1 - rs + tol):
        return False
    if np.any(ys < rs - tol) or np.any(ys > 1 - rs + tol):
        return False
    if np.any(rs < tol):
        return False
    
    # Pairwise check
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < rs[i] + rs[j] - tol:
                return False
    
    return True


def _fix_violations(pos, n, tol=1e-4):
    """Attempt to fix small constraint violations by adjusting radii/positions."""
    pos = pos.copy()
    xs = pos[:, 0]
    ys = pos[:, 1]
    rs = pos[:, 2]
    
    # Clip to boundaries
    for i in range(n):
        rs[i] = max(rs[i], 1e-6)
        xs[i] = np.clip(xs[i], rs[i], 1 - rs[i])
        ys[i] = np.clip(ys[i], rs[i], 1 - rs[i])
    
    # Iteratively reduce radii of overlapping pairs
    for _ in range(100):
        violated = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist - 1e-8:
                    # Reduce radii proportionally
                    overlap = min_dist - dist
                    total_r = rs[i] + rs[j]
                    if total_r > 1e-8:
                        scale = (dist - 1e-8) / total_r if dist > 1e-8 else 0.5
                        rs[i] *= scale
                        rs[j] *= scale
                    violated = True
        if not violated:
            break
    
    pos[:, 0] = xs
    pos[:, 1] = ys
    pos[:, 2] = rs
    return pos


def construct():
    """Construct 32 disjoint circles in the unit square maximizing sum of radii."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initial configuration
    init = _hexagonal_init(N)
    x0 = init.flatten()
    
    best_pos = init.copy()
    best_score = np.sum(init[:, 2])
    
    # Try multiple random perturbations of the initial config
    np.random.seed(42)
    n_tries = 0
    max_tries = 10
    
    for try_idx in range(max_tries):
        if time.time() > deadline:
            break
        
        if try_idx == 0:
            x_start = x0.copy()
        else:
            # Add random perturbation
            perturbation_scale = 0.02 * (1 - try_idx / max_tries)
            x_start = x0 + np.random.randn(3 * N) * perturbation_scale
            # Keep radii positive
            for i in range(N):
                x_start[3*i + 2] = max(x_start[3*i + 2], 0.01)
        
        try:
            result = _solve_single(x_start, N, maxiter=1500, tol=1e-7)
            
            if result.success or result.status in [0, 1]:
                pos = result.x.reshape(N, 3)
                pos = _fix_violations(pos, N)
                
                if _check_validity(pos, N, tol=1e-5):
                    score = np.sum(pos[:, 2])
                    if score > best_score:
                        best_score = score
                        best_pos = pos.copy()
        except Exception as e:
            pass
        
        n_tries += 1
    
    # Final polish: run a tight optimization from the best solution found
    if time.time() < deadline - 30 and n_tries > 0:
        try:
            x_best = best_pos.flatten()
            result = _solve_single(x_best, N, maxiter=3000, tol=1e-10)
            if result.success or result.status in [0, 1]:
                pos = result.x.reshape(N, 3)
                pos = _fix_violations(pos, N)
                if _check_validity(pos, N, tol=1e-5):
                    score = np.sum(pos[:, 2])
                    if score > best_score:
                        best_score = score
                        best_pos = pos.copy()
        except Exception:
            pass
    
    # Ensure final validity
    if not _check_validity(best_pos, N, tol=1e-5):
        # Fallback to a valid grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_pos = np.array(cells[:N])
    
    return best_pos

# EVOLVE-BLOCK-END
