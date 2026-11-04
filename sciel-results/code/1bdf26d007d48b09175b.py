"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
Strategy: Combinatorial topology search over contact graphs + Newton solver + gradient polish."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize


def _eval_packing(xy, r):
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _random_contact_graph(n, seed, max_degree=6, min_edges=None):
    """Generate a random planar-ish contact graph."""
    rng = random.Random(seed)
    adj = {i: set() for i in range(n)}
    
    # Random spanning tree
    nodes = list(range(n))
    rng.shuffle(nodes)
    for i in range(1, n):
        parent = rng.choice(nodes[:i])
        child = nodes[i]
        adj[parent].add(child)
        adj[child].add(parent)
    
    # Add extra edges (prefer nearby in tree for planarity)
    attempts = n * 4
    for _ in range(attempts):
        i, j = rng.sample(range(n), 2)
        if j not in adj[i] and len(adj[i]) < max_degree and len(adj[j]) < max_degree:
            # Prefer edges that would create triangles (more rigid)
            adj[i].add(j)
            adj[j].add(i)
    
    # Ensure enough edges for rigidity: need at least 2n-3 edges for a rigid framework
    target_edges = max(2 * n - 3, n - 1 + n // 2)
    current_edges = sum(len(adj[i]) for i in range(n)) // 2
    while current_edges < target_edges:
        i, j = rng.sample(range(n), 2)
        if j not in adj[i] and len(adj[i]) < max_degree and len(adj[j]) < max_degree:
            adj[i].add(j)
            adj[j].add(i)
            current_edges += 1
    
    return adj


def _count_edges(adj, n):
    return sum(len(adj[i]) for i in range(n)) // 2


def _newton_solve(adj, n, x0, y0, r0, max_iter=200, tol=1e-12):
    """
    Solve the tangency system: for each edge (i,j), dist(i,j) = r_i + r_j.
    Variables: x, y, r (3n variables).
    We fix one circle's position (x[0]=0, y[0]=0) to remove gauge freedom,
    so we have 3n - 2 unknowns and need 3n - 2 equations.
    
    Actually, let's use a different approach: solve for all 3n variables
    but add 2 additional constraints (e.g., center of mass at origin, or
    fix x[0]=0, y[0]=0).
    
    Better: we have E edges giving E equations. For rigidity we need E >= 3n - 6
    (accounting for 6 rigid body motions: 3 translations + 3 rotations... 
    actually 2 translations + 1 rotation = 3, so 3n - 3 equations needed).
    
    Let's use: fix x[0]=0, y[0]=0, and solve for the rest.
    That gives 3n - 2 unknowns. We need 3n - 2 equations from edges.
    """
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    
    E = len(edges)
    # We need E >= 3n - 2 for a well-determined system (after fixing 2 DOF)
    if E < 3 * n - 2:
        return None
    
    # Use only the first (3n-2) edges
    edges_used = edges[:3 * n - 2]
    
    # Variables: x[1:], y[1:], r[:]  -> (n-1) + (n-1) + n = 3n - 2
    # x[0] = 0, y[0] = 0
    
    def residual(v):
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)
        x[0] = 0.0
        y[0] = 0.0
        x[1:] = v[0:n-1]
        y[1:] = v[n-1:2*(n-1)]
        r[:] = v[2*(n-1):]
        
        res = np.zeros(len(edges_used))
        for k, (i, j) in enumerate(edges_used):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            res[k] = dist - (r[i] + r[j])
        return res
    
    def jacobian(v):
        x = np.zeros(n)
        y = np.zeros(n)
        r = np.zeros(n)
        x[0] = 0.0
        y[0] = 0.0
        x[1:] = v[0:n-1]
        y[1:] = v[n-1:2*(n-1)]
        r[:] = v[2*(n-1):]
        
        J = np.zeros((len(edges_used), 3*n - 2))
        for k, (i, j) in enumerate(edges_used):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-15:
                dist = 1e-15
            
            # d(dist)/d(x_i) = dx/dist, d(dist)/d(x_j) = -dx/dist
            # d(dist)/d(r_i) = -1, d(dist)/d(r_j) = -1
            
            if i > 0:
                col_xi = i - 1
                J[k, col_xi] += dx / dist
            if i > 0:
                col_yi = (n - 1) + (i - 1)
                J[k, col_yi] += dy / dist
            col_ri = 2*(n-1) + i
            J[k, col_ri] += -1.0
            
            if j > 0:
                col_xj = j - 1
                J[k, col_xj] += -dx / dist
            if j > 0:
                col_yj = (n - 1) + (j - 1)
                J[k, col_yj] += -dy / dist
            col_rj = 2*(n-1) + j
            J[k, col_rj] += -1.0
        
        return J
    
    # Initial guess
    v0 = np.zeros(3*n - 2)
    v0[0:n-1] = x0[1:]
    v0[n-1:2*(n-1)] = y0[1:]
    v0[2*(n-1):] = r0
    
    try:
        for it in range(max_iter):
            res = residual(v0)
            if np.max(np.abs(res)) < tol:
                break
            J = jacobian(v0)
            try:
                delta = np.linalg.solve(J, -res)
            except np.linalg.LinAlgError:
                # Use least squares
                delta, _, _, _ = np.linalg.lstsq(J, -res, rcond=None)
            v0 = v0 + delta
        res_final = residual(v0)
        if np.max(np.abs(res_final)) > 1e-6:
            return None
    except Exception:
        return None
    
    # Extract solution
    x_sol = np.zeros(n)
    y_sol = np.zeros(n)
    r_sol = np.zeros(n)
    x_sol[0] = 0.0
    y_sol[0] = 0.0
    x_sol[1:] = v0[0:n-1]
    y_sol[1:] = v0[n-1:2*(n-1)]
    r_sol[:] = v0[2*(n-1):]
    
    return x_sol, y_sol, r_sol


def _fit_to_square(x, y, r):
    """
    Given a configuration (possibly outside [0,1]^2), find the best
    translation + uniform scale to fit it inside [0,1]^2.
    Returns (x_new, y_new, r_new, valid).
    """
    n = len(x)
    # Current bounding box
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    w = x_max - x_min
    h = y_max - y_min
    
    if w < 1e-12 or h < 1e-12:
        return None
    
    # Scale factor to fit in unit square
    s = min(1.0 / w, 1.0 / h)
    
    # Scale
    x_s = x * s
    y_s = y * s
    r_s = r * s
    
    # Translate to [0,1]^2
    x_min_s = np.min(x_s - r_s)
    y_min_s = np.min(y_s - r_s)
    x_s = x_s - x_min_s
    y_s = y_s - y_min_s
    
    # Check if it fits
    x_max_s = np.max(x_s + r_s)
    y_max_s = np.max(y_s + r_s)
    
    if x_max_s > 1.0 + 1e-9 or y_max_s > 1.0 + 1e-9:
        return None
    
    # All radii must be positive
    if np.min(r_s) < 1e-10:
        return None
    
    return x_s, y_s, r_s


def _topology_search(deadline, num_topologies=100):
    """Search over random contact graph topologies."""
    best_xy = None
    best_r = None
    best_total = 0.0
    
    for t in range(num_topologies):
        if time.time() > deadline:
            break
        
        seed = 1000 + t
        rng = random.Random(seed)
        adj = _random_contact_graph(N, seed, max_degree=6)
        
        # Initial guess: random positions
        x0 = np.random.RandomState(seed).uniform(-1, 1, size=N)
        y0 = np.random.RandomState(seed + 1).uniform(-1, 1, size=N)
        r0 = np.random.RandomState(seed + 2).uniform(0.02, 0.06, size=N)
        
        sol = _newton_solve(adj, N, x0, y0, r0)
        if sol is None:
            continue
        
        x_sol, y_sol, r_sol = sol
        
        # Fit to unit square
        fitted = _fit_to_square(x_sol, y_sol, r_sol)
        if fitted is None:
            continue
        
        x_fit, y_fit, r_fit = fitted
        
        # Check validity
        xy = np.column_stack([x_fit, y_fit])
        total, valid = _eval_packing(xy, r_fit)
        if valid and total > best_total:
            best_total = total
            best_xy = xy.copy()
            best_r = r_fit.copy()
    
    return best_xy, best_r, best_total


def _smooth_objective(v, n):
    x = v[0:n]
    y = v[n:2*n]
    r = v[2*n:3*n]

    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min

    beta = 20.0
    wall_pen = np.sum(np.log1p(np.exp(beta * wall_viol)) / beta)

    total_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            total_viol += np.log1p(np.exp(beta * viol)) / beta

    return -np.sum(r) + 100.0 * (wall_pen + total_viol)


def _smooth_grad(v, n):
    x = v[0:n].copy()
    y = v[n:2*n].copy()
    r = v[2*n:3*n].copy()

    grad = np.zeros(3 * n)
    beta = 20.0

    wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    wall_viol = r - wall_min
    sigmoid_w = 1.0 / (1.0 + np.exp(-beta * wall_viol))

    for i in range(n):
        sw = sigmoid_w[i]
        if sw > 1e-10:
            dists = [x[i], 1.0 - x[i], y[i], 1.0 - y[i]]
            min_idx = int(np.argmin(dists))
            grad[2*n + i] += 100.0 * sw
            if min_idx == 0:
                grad[i] -= 100.0 * sw
            elif min_idx == 1:
                grad[i] += 100.0 * sw
            elif min_idx == 2:
                grad[n + i] -= 100.0 * sw
            else:
                grad[n + i] += 100.0 * sw

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            viol = r[i] + r[j] - dist
            s = 1.0 / (1.0 + np.exp(-beta * viol))
            if s > 1e-10:
                factor = 100.0 * s
                if dist > 1e-12:
                    grad[i] += factor * dx / dist
                    grad[n + i] += factor * dy / dist
                    grad[j] -= factor * dx / dist
                    grad[n + j] -= factor * dy / dist
                grad[2*n + i] += factor
                grad[2*n + j] += factor

    grad[2*n:3*n] -= 1.0

    return grad


def _gradient_polish(xy, r, deadline, tol=1e-10, max_iter=1000):
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    bounds = []
    for i in range(n):
        bounds.append((0.0001, 0.9999))
    for i in range(n):
        bounds.append((0.0001, 0.9999))
    for i in range(n):
        bounds.append((0.0001, 0.4999))

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _smooth_objective(v, n)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        return _smooth_grad(v, n)

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8,
                                   'maxfun': 20000})
    except Exception:
        return xy, r

    v_opt = result.x
    new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    new_r = v_opt[2*n:3*n]

    total, valid = _eval_packing(new_xy, new_r)
    if valid:
        return new_xy, new_r

    return xy, r


def _best_fit_local_search(xy, r, deadline, grid_size=80):
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 20
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            others_xy = xy[mask]
            others_r = r[mask]

            if len(others_xy) > 0:
                dx = candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                dy = candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                circle_clear = np.min(dists - others_r[np.newaxis, :], axis=1)
                r_candidates = np.minimum(wall_clear, circle_clear)
            else:
                r_candidates = wall_clear.copy()

            idx = int(np.argmax(r_candidates))
            new_x, new_y = candidates[idx, 0], candidates[idx, 1]
            new_r = r_candidates[idx]

            old_r = r[i]
            if new_r > old_r + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _grid_init(side):
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.002
                positions.append([x, y, r])
    while len(positions) < N:
        positions.append([0.5, 0.5, 0.01])
    return np.array(positions[:N])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Topology search (the main new strategy)
    if time.time() < deadline - 30:
        topo_xy, topo_r, topo_total = _topology_search(deadline - 25, num_topologies=150)
        if topo_xy is not None and topo_total > best_total:
            best_total = topo_total
            best_valid = True
            best_xy = topo_xy.copy()
            best_r = topo_r.copy()

    # Phase 2: Grid init + local search as fallback/diversity
    for side in [5, 6]:
        if time.time() > deadline - 30:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline - 25, grid_size=60)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 3: Gradient polish on best
    if best_valid and best_xy is not None and time.time() < deadline - 15:
        polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline - 10)
        total, valid = _eval_packing(polished_xy, polished_r)
        if valid and total > best_total:
            best_total = total
            best_xy = polished_xy
            best_r = polished_r

    # Phase 4: Final local search
    if best_valid and best_xy is not None and time.time() < deadline - 5:
        final_xy, final_r, final_total, final_valid = _best_fit_local_search(
            best_xy, best_r, deadline - 2, grid_size=100)
        if final_valid and final_total > best_total:
            best_total = final_total
            best_xy = final_xy
            best_r = final_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()
        total, valid = _eval_packing(xy, r)
        if valid:
            best_total = total
            best_valid = True
            best_xy = xy
            best_r = r
        else:
            result = np.zeros((N, 3))
            for i in range(N):
                gx = i % side
                gy = i // side
                result[i, 0] = (gx + 0.5) / side
                result[i, 1] = (gy + 0.5) / side
                result[i, 2] = 0.4 / side
            return result

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
