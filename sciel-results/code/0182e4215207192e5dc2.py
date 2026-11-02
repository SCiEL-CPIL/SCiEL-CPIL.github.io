"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Strategy: Penalty-based L-BFGS-B with smooth softplus penalty,
Delaunay contact-graph constraint selection, multi-start + local search.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-9
DEADLINE = 145.0  # safety margin below 150s kill


def _softplus(x, k=50.0):
    """Smooth approximation of max(0, x)."""
    return (1.0 / k) * np.log1p(np.exp(k * x))


def _max_radius(pos, r):
    """Maximal radius for each circle given current positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                dmin = min(dmin, np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) - r[j])
        mr[i] = max(0.0, dmin)
    return mr


def _get_contact_edges(pos):
    """Get Delaunay contact edges for constraint selection."""
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = simplex[a], simplex[b]
                    edges.add((min(i, j), max(i, j)))
        return list(edges)
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _penalty_objective(x, edges, penalty_weight):
    """Objective: -sum(r) + penalty for constraint violations."""
    pos = x[:2*N].reshape(N, 2)
    r = x[2*N:]
    
    # Wall violations (softplus of violation amount)
    wall_viol = 0.0
    for i in range(N):
        wall_viol += _softplus(r[i] - pos[i, 0])
        wall_viol += _softplus(r[i] - pos[i, 1])
        wall_viol += _softplus(pos[i, 0] + r[i] - 1.0)
        wall_viol += _softplus(pos[i, 1] + r[i] - 1.0)
    
    # Pairwise separation violations for contact edges
    sep_viol = 0.0
    for (i, j) in edges:
        d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
        sep_viol += _softplus(r[i] + r[j] - d)
    
    # Also check all pairs for safety (lighter weight)
    all_pair_viol = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if (i, j) not in edges and (j, i) not in edges:
                d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                all_pair_viol += _softplus(r[i] + r[j] - d)
    
    penalty = penalty_weight * (wall_viol + 2.0 * sep_viol + 0.5 * all_pair_viol)
    return -np.sum(r) + penalty


def _penalty_gradient(x, edges, penalty_weight):
    """Gradient of penalty objective."""
    pos = x[:2*N].reshape(N, 2)
    r = x[2*N:]
    grad = np.zeros_like(x)
    
    # Base gradient: -1 for radii
    grad[2*N:] = -1.0
    
    # Wall gradient contributions
    k = 50.0
    for i in range(N):
        # r_i - x_i
        v = r[i] - pos[i, 0]
        s = 1.0 / (1.0 + np.exp(-k * v))
        grad[2*i] += penalty_weight * s  # d/dx_i
        grad[2*N+i] += penalty_weight * s  # d/dr_i
        
        # r_i - y_i
        v = r[i] - pos[i, 1]
        s = 1.0 / (1.0 + np.exp(-k * v))
        grad[2*i+1] += penalty_weight * s
        grad[2*N+i] += penalty_weight * s
        
        # x_i + r_i - 1
        v = pos[i, 0] + r[i] - 1.0
        s = 1.0 / (1.0 + np.exp(-k * v))
        grad[2*i] += penalty_weight * s
        grad[2*N+i] += penalty_weight * s
        
        # y_i + r_i - 1
        v = pos[i, 1] + r[i] - 1.0
        s = 1.0 / (1.0 + np.exp(-k * v))
        grad[2*i+1] += penalty_weight * s
        grad[2*N+i] += penalty_weight * s
    
    # Pairwise separation gradient for contact edges
    for (i, j) in edges:
        dx = pos[i, 0] - pos[j, 0]
        dy = pos[i, 1] - pos[j, 1]
        d = np.hypot(dx, dy)
        if d < 1e-12:
            d = 1e-12
        v = r[i] + r[j] - d
        s = 1.0 / (1.0 + np.exp(-k * v))
        w = penalty_weight * 2.0 * s
        grad[2*i] += w * (dx / d)
        grad[2*i+1] += w * (dy / d)
        grad[2*j] -= w * (dx / d)
        grad[2*j+1] -= w * (dy / d)
        grad[2*N+i] += w
        grad[2*N+j] += w
    
    # All other pairs (lighter)
    edge_set = set(edges)
    for i in range(N):
        for j in range(i+1, N):
            if (i, j) in edge_set or (j, i) in edge_set:
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.hypot(dx, dy)
            if d < 1e-12:
                d = 1e-12
            v = r[i] + r[j] - d
            s = 1.0 / (1.0 + np.exp(-k * v))
            w = penalty_weight * 0.5 * s
            grad[2*i] += w * (dx / d)
            grad[2*i+1] += w * (dy / d)
            grad[2*j] -= w * (dx / d)
            grad[2*j+1] -= w * (dy / d)
            grad[2*N+i] += w
            grad[2*N+j] += w
    
    return grad


def _valid_check(arr, tol=1e-7):
    """Check all circles are in-square and non-overlapping."""
    for i in range(len(arr)):
        x, y, r = arr[i]
        if r < 0 or x < r + tol or x > 1 - r - tol or y < r + tol or y > 1 - r - tol:
            return False
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            d = np.hypot(arr[i, 0] - arr[j, 0], arr[i, 1] - arr[j, 1])
            if d < arr[i, 2] + arr[j, 2] - tol:
                return False
    return True


def _fix_boundaries(pos, rad):
    """Push circles back into valid region."""
    pos = pos.copy()
    rad = rad.copy()
    for i in range(N):
        rad[i] = max(0.0, min(rad[i], 0.5))
        pos[i, 0] = max(rad[i], min(1.0 - rad[i], pos[i, 0]))
        pos[i, 1] = max(rad[i], min(1.0 - rad[i], pos[i, 1]))
    return pos, rad


def _optimize_single(pos0, rad0, time_budget=8.0, initial_penalty=10.0):
    """Run SLSQP optimization with Delaunay constraints."""
    t0 = time.time()
    pos = pos0.copy()
    rad = rad0.copy()
    
    # Ensure initial validity
    rad = np.minimum(rad, _max_radius(pos, np.full(N, 0.5)))
    
    best_pos = pos.copy()
    best_rad = rad.copy()
    best_score = float(rad.sum())
    
    # SLSQP iterations
    for _ in range(15):
        if time.time() - t0 > time_budget:
            break
            
        # Get current Delaunay edges
        edges = _get_contact_edges(pos)
        
        # Define objective: maximize sum of radii
        def obj(x):
            return -x[2*N:].sum()
            
        # Define gradient
        def jac(x):
            g = np.zeros_like(x)
            g[2*N:] = -1.0
            return g
            
        # Define constraints
        # 1. Wall constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
        #    SLSQP constraints are usually h(x) = 0 (equality) or g(x) >= 0 (inequality).
        #    We want:
        #    x_i - r_i >= 0
        #    1 - x_i - r_i >= 0
        #    y_i - r_i >= 0
        #    1 - y_i - r_i >= 0
        
        def wall_constraints(x):
            pos_x = x[:2*N].reshape(N, 2)
            r = x[2*N:]
            c = np.zeros(4*N)
            for i in range(N):
                c[4*i] = pos_x[i, 0] - r[i]
                c[4*i+1] = 1.0 - pos_x[i, 0] - r[i]
                c[4*i+2] = pos_x[i, 1] - r[i]
                c[4*i+3] = 1.0 - pos_x[i, 1] - r[i]
            return c
            
        def wall_jac(x):
            J = np.zeros((4*N, 3*N))
            for i in range(N):
                # x_i - r_i
                J[4*i, 2*i] = 1.0
                J[4*i, 2*N+i] = -1.0
                # 1 - x_i - r_i
                J[4*i+1, 2*i] = -1.0
                J[4*i+1, 2*N+i] = -1.0
                # y_i - r_i
                J[4*i+2, 2*i+1] = 1.0
                J[4*i+2, 2*N+i] = -1.0
                # 1 - y_i - r_i
                J[4*i+3, 2*i+1] = -1.0
                J[4*i+3, 2*N+i] = -1.0
            return J
            
        # 2. Pairwise distance constraints: dist(i,j) >= r_i + r_j
        #    h(x) = dist(i,j) - r_i - r_j >= 0
        
        def pair_constraints(x):
            pos_x = x[:2*N].reshape(N, 2)
            r = x[2*N:]
            c = []
            for (i, j) in edges:
                d = np.hypot(pos_x[i,0]-pos_x[j,0], pos_x[i,1]-pos_x[j,1])
                c.append(d - r[i] - r[j])
            return np.array(c)
            
        def pair_jac(x):
            pos_x = x[:2*N].reshape(N, 2)
            r = x[2*N:]
            J = np.zeros((len(edges), 3*N))
            for k, (i, j) in enumerate(edges):
                dx = pos_x[i,0] - pos_x[j,0]
                dy = pos_x[i,1] - pos_x[j,1]
                d = np.hypot(dx, dy)
                if d < 1e-12:
                    d = 1e-12
                # dist(i,j) - r_i - r_j
                # d/dx_i = dx/d, d/dy_i = dy/d
                # d/dx_j = -dx/d, d/dy_j = -dy/d
                # d/dr_i = -1, d/dr_j = -1
                J[k, 2*i] = dx/d
                J[k, 2*i+1] = dy/d
                J[k, 2*j] = -dx/d
                J[k, 2*j+1] = -dy/d
                J[k, 2*N+i] = -1.0
                J[k, 2*N+j] = -1.0
            return J
            
        constraints = []
        # Wall constraints
        constraints.append({'type': 'ineq', 'fun': wall_constraints, 'jac': wall_jac})
        # Pair constraints
        if len(edges) > 0:
             constraints.append({'type': 'ineq', 'fun': pair_constraints, 'jac': pair_jac})
        
        bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
        
        x0 = np.concatenate([pos.reshape(-1), rad])
        
        try:
            res = minimize(obj, x0, jac=jac, method='SLSQP', bounds=bounds, 
                           constraints=constraints, 
                           options={'maxiter': 100, 'ftol': 1e-9, 'disp': False})
            if res.success or res.fun < obj(x0):
                x_new = res.x
                pos_new = x_new[:2*N].reshape(N, 2)
                rad_new = np.maximum(x_new[2*N:], 0.0)
                
                # Check validity
                arr = np.column_stack([pos_new, rad_new])
                if _valid_check(arr, tol=1e-6):
                    s = float(rad_new.sum())
                    if s > best_score:
                        best_score = s
                        best_pos = pos_new.copy()
                        best_rad = rad_new.copy()
                        pos = pos_new.copy()
                        rad = rad_new.copy()
                else:
                    # If invalid, try to repair by reducing radii
                    rad_new = _max_radius(pos_new, rad_new)
                    arr = np.column_stack([pos_new, rad_new])
                    if _valid_check(arr, tol=1e-6):
                        s = float(rad_new.sum())
                        if s > best_score:
                            best_score = s
                            best_pos = pos_new.copy()
                            best_rad = rad_new.copy()
                            pos = pos_new.copy()
                            rad = rad_new.copy()
                            
        except Exception:
            pass
            
    # Final radius maximization
    rad_final = _max_radius(best_pos, best_rad)
    arr = np.column_stack([best_pos, rad_final])
    if _valid_check(arr, tol=1e-6):
        s = float(rad_final.sum())
        if s > best_score:
            best_score = s
            best_rad = rad_final.copy()
            
    return best_pos, best_rad


def _initial_hex():
    """Hexagonal lattice initialization."""
    r_hex = 0.12
    dx = 2 * r_hex
    dy = r_hex * np.sqrt(3)
    hex_pos = []
    for row in range(8):
        for col in range(8):
            x = col * dx + (row % 2) * dx / 2 + 0.15
            y = row * dy + 0.15
            if 0.05 < x < 0.95 and 0.05 < y < 0.95:
                hex_pos.append((x, y))
    hex_pos = hex_pos[:N]
    if len(hex_pos) >= N:
        hp = np.array(hex_pos, dtype=float)
        hr = _max_radius(hp, np.full(N, 0.1))
        return hp, hr
    return None, None


def _initial_grid(side):
    """Grid initialization."""
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side))
    cells = cells[:N]
    pos = np.array(cells, dtype=float)
    rad = np.full(N, 0.5 / side - 1e-4)
    for _ in range(10):
        rad = _max_radius(pos, rad)
    return pos, rad


def _initial_corner_weighted(rng):
    """Random initialization with corner bias."""
    pos = np.zeros((N, 2))
    for i in range(N):
        if i < 4:
            # Corners
            corner = rng.choice(4)
            cx = [0.15, 0.85, 0.15, 0.85][corner]
            cy = [0.15, 0.15, 0.85, 0.85][corner]
            pos[i] = [cx + 0.05*(rng.random()-0.5), cy + 0.05*(rng.random()-0.5)]
        elif i < 8:
            # Edges
            edge = rng.choice(4)
            if edge == 0:
                pos[i] = [rng.random()*0.8+0.1, 0.12]
            elif edge == 1:
                pos[i] = [rng.random()*0.8+0.1, 0.88]
            elif edge == 2:
                pos[i] = [0.12, rng.random()*0.8+0.1]
            else:
                pos[i] = [0.88, rng.random()*0.8+0.1]
        else:
            pos[i] = rng.random(2) * 0.7 + 0.15
    pos = np.clip(pos, 0.05, 0.95)
    rad = _max_radius(pos, np.full(N, 0.05))
    return pos, rad


def _shake_and_refine(pos, rad, rng, time_budget=3.0):
    """Perturb and re-optimize."""
    t0 = time.time()
    best_pos = pos.copy()
    best_rad = rad.copy()
    best_score = float(rad.sum())
    
    for _ in range(5):
        if time.time() - t0 > time_budget:
            break
        # Perturb random subset
        n_perturb = rng.integers(3, 8)
        idx_perturb = rng.choice(N, size=n_perturb, replace=False)
        pos_p = pos.copy()
        pos_p[idx_perturb] += 0.03 * (rng.random((n_perturb, 2)) * 2 - 1)
        pos_p = np.clip(pos_p, 0.05, 0.95)
        rad_p = _max_radius(pos_p, rad)
        
        pos_f, rad_f = _optimize_single(pos_p, rad_p, time_budget=min(2.0, time_budget - (time.time()-t0)))
        arr = np.column_stack([pos_f, rad_f])
        if _valid_check(arr, tol=1e-6):
            s = float(rad_f.sum())
            if s > best_score:
                best_score = s
                best_pos = pos_f.copy()
                best_rad = rad_f.copy()
    
    return best_pos, best_rad


def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    best = None
    best_score = -1.0

    def consider(pos, rad):
        nonlocal best, best_score
        arr = np.column_stack([pos, rad])
        if not _valid_check(arr, tol=1e-6):
            # Try to fix: reduce radii slightly
            rad = np.maximum(rad * 0.998, 0)
            arr = np.column_stack([pos, rad])
            if not _valid_check(arr, tol=1e-6):
                return
        s = float(arr[:, 2].sum())
        if s > best_score:
            best_score = s
            best = arr.copy()

    # ---- Phase A: Generate diverse initializations ----
    starts = []

    # Grid initializations (various densities)
    for side in [4, 5, 6]:
        if time.time() > DEADLINE - 40:
            break
        pos, rad = _initial_grid(side)
        starts.append((pos.copy(), rad.copy()))
        
        # Perturbed grid
        p0 = pos + 0.04 * (rng.random((N, 2)) * 2 - 1)
        p0 = np.clip(p0, 0.05, 0.95)
        r0 = _max_radius(p0, rad)
        starts.append((p0, r0))

    # Hexagonal packing
    if time.time() <= DEADLINE - 40:
        hp, hr = _initial_hex()
        if hp is not None:
            starts.append((hp.copy(), hr.copy()))

    # Corner-weighted random
    for k in range(4):
        if time.time() > DEADLINE - 40:
            break
        pos, rad = _initial_corner_weighted(rng)
        starts.append((pos, rad))

    # Pure random
    for k in range(4):
        if time.time() > DEADLINE - 40:
            break
        p0 = rng.random((N, 2)) * 0.7 + 0.15
        r0 = _max_radius(p0, np.full(N, 0.05))
        starts.append((p0, r0))

    # ---- Phase B: Optimize from each start ----
    n_starts = len(starts)
    for idx, (pos0, rad0) in enumerate(starts):
        if time.time() > DEADLINE - 25:
            break
        remaining = DEADLINE - 25 - time.time()
        budget = min(10.0, max(3.0, remaining / max(1, n_starts - idx)))
        pos_r, rad_r = _optimize_single(pos0, rad0, time_budget=budget, initial_penalty=5.0)
        consider(pos_r, rad_r)

    # ---- Phase C: Shake-and-refine on best ----
    if best is not None and time.time() < DEADLINE - 15:
        pos_b = best[:, :2].copy()
        rad_b = best[:, 2].copy()
        remaining = DEADLINE - 15 - time.time()
        pos_s, rad_s = _shake_and_refine(pos_b, rad_b, rng, time_budget=min(10.0, remaining))
        consider(pos_s, rad_s)

    # ---- Phase D: Final polish with higher penalty ----
    if best is not None and time.time() < DEADLINE - 8:
        pos_b = best[:, :2].copy()
        rad_b = best[:, 2].copy()
        remaining = DEADLINE - 8 - time.time()
        pos_p, rad_p = _optimize_single(pos_b, rad_b, time_budget=min(5.0, remaining), initial_penalty=50.0)
        consider(pos_p, rad_p)

    if best is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        pos, rad = _initial_grid(side)
        return np.column_stack([pos, rad])
    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
