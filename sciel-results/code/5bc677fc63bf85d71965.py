"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, SLSQP
from scipy.spatial import Delaunay

N = 26

# EVOLVE-BLOCK-START

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

def _get_delaunay_edges(centers):
    """Get Delaunay triangulation edges for the given centers."""
    try:
        # Delaunay requires at least 3 points and non-collinear
        if len(centers) < 3:
            return [(i, j) for i in range(len(centers)) for j in range(i+1, len(centers))]
        
        # Check for collinearity or degenerate cases
        # Simple heuristic: if all points are close to a line, fallback
        # Or just catch exceptions from Delaunay
        
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                a, b = simplex[i], simplex[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback to all pairs if Delaunay fails (e.g., collinear points)
        return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _penalty_objective(params, penalty_weight=100.0, use_delaunay=True):
    """
    Objective function with penalty terms for constraint violations.
    Minimizes -sum(r) + penalty.
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary penalties
    # x - r >= 0  =>  r - x <= 0
    pen_x_lo = np.maximum(0, r - x)
    pen_x_hi = np.maximum(0, x + r - 1)
    pen_y_lo = np.maximum(0, r - y)
    pen_y_hi = np.maximum(0, y + r - 1)
    pen_r = np.maximum(0, -r)
    
    penalty += np.sum(pen_x_lo ** 2) + np.sum(pen_x_hi ** 2)
    penalty += np.sum(pen_y_lo ** 2) + np.sum(pen_y_hi ** 2)
    penalty += np.sum(pen_r ** 2)
    
    # Overlap penalties
    if use_delaunay:
        centers = np.column_stack([x, y])
        edges = _get_delaunay_edges(centers)
        
        # Add all pairs that are very close but not in Delaunay (safety net)
        # This is O(N^2) but we only do it if Delaunay fails or for safety?
        # Actually, Delaunay might miss some close pairs if the triangulation is weird?
        # No, Delaunay edges connect close points. But if points are collinear, it fails.
        # Let's just use all pairs for robustness in the penalty, as N=26 is small enough
        # for O(N^2) in a penalty function called by L-BFGS-B?
        # 26*25/2 = 325 pairs. Very fast.
        # The inspiration suggested Delaunay for speed, but for N=26, full pairwise is fine
        # and more robust. Let's stick to full pairwise for the penalty to ensure
        # we catch all overlaps, as Delaunay might miss some in degenerate cases.
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    penalty += overlap ** 2
    else:
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                overlap = r[i] + r[j] - dist
                if overlap > 0:
                    penalty += overlap ** 2

    return -np.sum(r) + penalty_weight * penalty

def _slsqp_objective(params):
    """Objective for SLSQP: minimize -sum(r)"""
    r = params[2::3]
    return -np.sum(r)

def _slsqp_constraints(params):
    """Generate constraint functions for SLSQP."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    cons = []
    
    # Boundary constraints
    for i in range(N):
        idx_x = 3*i
        idx_y = 3*i + 1
        idx_r = 3*i + 2
        
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i] - x[3*i+2]})
        # 1 - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i] - x[3*i+2]})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[3*i+1] - x[3*i+2]})
        # 1 - y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1 - x[3*i+1] - x[3*i+2]})
        
    # Overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            idx_i = 3*i
            idx_j = 3*j
            def overlap_fun(x, i=i, j=j):
                dx = x[idx_i] - x[idx_j]
                dy = x[idx_i+1] - x[idx_j+1]
                dr = x[idx_i+2] + x[idx_j+2]
                return dx*dx + dy*dy - dr*dr
            cons.append({'type': 'ineq', 'fun': overlap_fun})
            
    return cons

def _slsqp_bounds():
    """Bounds for SLSQP."""
    b = []
    for _ in range(N):
        b.append((0.0, 1.0)) # x
        b.append((0.0, 1.0)) # y
        b.append((1e-6, 0.5)) # r
    return b

def _initial_config_grid():
    """Simple grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
    return np.array(cells[:N])

def _initial_config_hex():
    """Hexagonal-like initial configuration."""
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.1 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    
    positions.append([0.5, 0.5, 0.5 / side * 0.5])
    return np.array(positions[:N])

def _physics_relax(xyz, iterations=50, time_limit=1.0):
    """
    Simple force-directed relaxation to push overlapping circles apart.
    """
    start_time = time.time()
    xyz = xyz.copy()
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()
    
    for _ in range(iterations):
        if time.time() - start_time > time_limit:
            break
        
        fx = np.zeros(N)
        fy = np.zeros(N)
        
        # Repulsion forces for overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy) + 1e-10
                min_dist = r[i] + r[j]
                
                if dist < min_dist:
                    force = (min_dist - dist) * 10.0
                    fx[i] += force * dx / dist
                    fy[i] += force * dy / dist
                    fx[j] -= force * dx / dist
                    fy[j] -= force * dy / dist
        
        # Boundary forces
        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 20.0
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * 20.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 20.0
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * 20.0
        
        # Update positions
        step = 0.01
        x += step * fx
        y += step * fy
        
        # Clip to boundaries
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        
        xyz[:, 0] = x
        xyz[:, 1] = y
    
    return xyz

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    best_score = -np.inf
    best_x = None
    
    # Helper to check validity and calculate score
    def evaluate(x_flat):
        x = x_flat.reshape(N, 3)
        xs, ys, rs = x[:, 0], x[:, 1], x[:, 2]
        
        # Check boundaries
        if np.any(rs <= 0): return -np.inf
        if np.any(xs - rs < -1e-9) or np.any(1 - xs - rs < -1e-9): return -np.inf
        if np.any(ys - rs < -1e-9) or np.any(1 - ys - rs < -1e-9): return -np.inf
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                rad_sum = rs[i] + rs[j]
                if dist_sq < rad_sum**2 - 1e-9:
                    return -np.inf
        return np.sum(rs)

    # Generate initial guesses
    def get_initial_guesses():
        guesses = []
        
        # 1. Grid based
        side = int(np.ceil(np.sqrt(N)))
        grid_cells = []
        for i in range(side):
            for j in range(side):
                if len(grid_cells) >= N: break
                x0 = (i + 0.5) / side
                y0 = (j + 0.5) / side
                r0 = 0.5 / side * 0.8
                grid_cells.append([x0, y0, r0])
        guesses.append(grid_cells)
        
        # 2. Random uniform
        np.random.seed(42)
        x_rand = np.random.uniform(0.1, 0.9, N)
        y_rand = np.random.uniform(0.1, 0.9, N)
        r_rand = np.full(N, 0.05)
        guesses.append(list(zip(x_rand, y_rand, r_rand)))
        
        # 3. Hexagonal
        guesses.append(_initial_config_hex().tolist())
        
        return guesses

    initial_guesses = get_initial_guesses()
    
    # Strategy 1: Penalty-based L-BFGS-B
    # This is robust against local minima traps if started from diverse points
    for start_idx, guess in enumerate(initial_guesses):
        if time.time() > deadline - 20: break
        
        x0 = np.array(guess).flatten()
        
        # Relax first to get out of obvious overlaps
        x_relaxed = _physics_relax(x0.reshape(N, 3), iterations=50, time_limit=2.0).flatten()
        
        try:
            res = minimize(
                _penalty_objective, 
                x_relaxed, 
                method='L-BFGS-B', 
                bounds=_slsqp_bounds(),
                options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
            )
            
            # Check if the result is valid
            score = evaluate(res.x)
            if score > best_score:
                best_score = score
                best_x = res.x.copy()
                
        except Exception:
            continue

    # Strategy 2: SLSQP from the best solution found so far
    # SLSQP is better at handling constraints exactly, so it can polish the solution
    if best_x is not None and time.time() < deadline - 10:
        x0 = best_x.copy()
        try:
            res = minimize(
                _slsqp_objective, 
                x0, 
                method='SLSQP', 
                bounds=_slsqp_bounds(),
                constraints=_slsqp_constraints(x0), # Note: constraints are static functions here
                options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
            )
            
            score = evaluate(res.x)
            if score > best_score:
                best_score = score
                best_x = res.x.copy()
                
        except Exception:
            pass

    # If no valid solution found, fallback to simple grid
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_x = np.array(cells[:N]).flatten()
        best_score = evaluate(best_x)

    return best_x.reshape(N, 3)

# EVOLVE-BLOCK-END
