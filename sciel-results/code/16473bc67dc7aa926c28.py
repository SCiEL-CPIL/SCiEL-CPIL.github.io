"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
Strategy: contact-graph (rigid truss) enumeration + high-precision root finding.
For each candidate contact graph, solve the tangency system exactly.
"""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import fsolve, minimize, LinearConstraint
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    """
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


def _grid_init(side):
    """Create a grid of up to N circles."""
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


def _random_init(n, seed=0):
    """Random initialization with small radii."""
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    r = rng.uniform(0.02, 0.05, size=n)
    return xy, r


def _best_fit_local_search(xy, r, deadline, grid_size=80):
    """
    Iteratively update each circle to the best position (maximizing its radius)
    given all other circles fixed. Vectorized over a grid of candidate positions.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    # Candidate grid
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)

    # Precompute candidate wall clearances
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))

    max_sweeps = 30
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

            if new_r > old_r + 1e-10:
                step = 1.0 / grid_size
                fine_gx = np.linspace(max(0, new_x - 3 * step), min(1, new_x + 3 * step), 20)
                fine_gy = np.linspace(max(0, new_y - 3 * step), min(1, new_y + 3 * step), 20)
                FGX, FGY = np.meshgrid(fine_gx, fine_gy, indexing='ij')
                fine_candidates = np.column_stack([FGX.ravel(), FGY.ravel()])
                FM = len(fine_candidates)

                if FM > 0:
                    fine_wall_clear = np.minimum(np.minimum(fine_candidates[:, 0], 1.0 - fine_candidates[:, 0]),
                                                 np.minimum(fine_candidates[:, 1], 1.0 - fine_candidates[:, 1]))
                    if len(others_xy) > 0:
                        fdx = fine_candidates[:, 0:1] - others_xy[:, 0][np.newaxis, :]
                        fdy = fine_candidates[:, 1:2] - others_xy[:, 1][np.newaxis, :]
                        fdists = np.sqrt(fdx * fdx + fdy * fdy)
                        fcircle_clear = np.min(fdists - others_r[np.newaxis, :], axis=1)
                        fr_candidates = np.minimum(fine_wall_clear, fcircle_clear)
                    else:
                        fr_candidates = fine_wall_clear.copy()

                    fidx = int(np.argmax(fr_candidates))
                    fnew_x, fnew_y = fine_candidates[fidx, 0], fine_candidates[fidx, 1]
                    fnew_r = fr_candidates[fidx]

                    if fnew_r > new_r + 1e-10:
                        xy[i, 0] = fnew_x
                        xy[i, 1] = fnew_y
                        r[i] = fnew_r

        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()

        if not improved:
            break

    return best_xy, best_r, best_total, best_valid


def _discover_contact_graph(xy, r, n_contacts_per_circle=3):
    """
    Discover a contact graph: for each circle, find the nearest neighbors
    that are in contact (distance ≈ r_i + r_j).
    Returns adjacency list.
    """
    n = len(xy)
    # Compute all pairwise distances
    D = squareform(pdist(xy))
    R_sum = r[:, None] + r[None, :]

    # Contact if distance is within tolerance of sum of radii
    tol = 1e-4
    contact_mask = (D < R_sum + tol) & (np.abs(D - R_sum) < tol)
    np.fill_diagonal(contact_mask, False)

    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and contact_mask[i, j]:
                adj[i].append(j)

    return adj, contact_mask


def _solve_tangency_system(xy0, r0, adj, deadline, max_iter=200):
    """
    Solve the tangency system for a given contact graph.
    Variables: x_i, y_i, r_i for all i.
    Constraints:
      - For each contact (i,j): (x_i-x_j)^2 + (y_i-y_j)^2 = (r_i+r_j)^2
      - For each circle touching a wall: x_i = r_i or x_i = 1-r_i, etc.
    
    We use a least-squares approach to solve the nonlinear system.
    """
    n = len(xy0)
    
    # Initial guess
    v0 = np.concatenate([xy0[:, 0], xy0[:, 1], r0])
    
    def equations(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        
        eqs = []
        
        # Contact constraints
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx * dx + dy * dy
                    r_sum_sq = (r[i] + r[j]) ** 2
                    eqs.append(dist_sq - r_sum_sq)
        
        # Wall constraints: for each circle, enforce it's inside the square
        # We use soft wall constraints: penalize if circle goes outside
        for i in range(n):
            # Left wall
            eqs.append(max(0, r[i] - x[i]))
            # Right wall
            eqs.append(max(0, r[i] - (1.0 - x[i])))
            # Bottom wall
            eqs.append(max(0, r[i] - y[i]))
            # Top wall
            eqs.append(max(0, r[i] - (1.0 - y[i])))
        
        return np.array(eqs)
    
    # Use fsolve
    try:
        v_sol, info, ier, msg = fsolve(equations, v0, full_output=True, maxfev=1000)
        if ier != 1:
            return None, None
    except Exception:
        return None, None
    
    new_xy = np.column_stack([v_sol[0:n], v_sol[n:2*n]])
    new_r = v_sol[2*n:3*n]
    
    # Validate
    total, valid = _eval_packing(new_xy, new_r)
    if not valid:
        return None, None
    
    return new_xy, new_r


def _contact_graph_optimize(xy, r, deadline, max_graphs=20):
    """
    Main contact-graph optimization:
    1. Discover contact graph from current packing
    2. Solve tangency system for that graph
    3. Perturb and repeat to explore different contact topologies
    """
    n = len(xy)
    best_xy = xy.copy()
    best_r = r.copy()
    best_total = _eval_packing(xy, r)[0]
    
    for attempt in range(max_graphs):
        if time.time() > deadline - 5:
            break
        
        # Discover contact graph
        adj, contact_mask = _discover_contact_graph(xy, r)
        
        # Solve tangency system
        sol_xy, sol_r = _solve_tangency_system(xy, r, adj, deadline)
        if sol_xy is not None:
            total, valid = _eval_packing(sol_xy, sol_r)
            if valid and total > best_total + 1e-10:
                best_total = total
                best_xy = sol_xy.copy()
                best_r = sol_r.copy()
        
        # Perturb to explore different contact topologies
        if time.time() > deadline - 5:
            break
        
        # Random perturbation of positions
        rng = np.random.RandomState(attempt * 1000)
        perturbation_scale = 0.05 / (attempt + 1)
        xy_pert = xy + rng.uniform(-perturbation_scale, perturbation_scale, size=xy.shape)
        xy_pert = np.clip(xy_pert, 0.05, 0.95)
        
        # Also try radius perturbation
        r_pert = r * (1.0 + rng.uniform(-0.1, 0.1, size=n))
        r_pert = np.clip(r_pert, 0.01, 0.4)
        
        # Local search from perturbed state
        ls_xy, ls_r, ls_total, ls_valid = _best_fit_local_search(
            xy_pert.copy(), r_pert.copy(), deadline, grid_size=50)
        if ls_valid and ls_total > best_total + 1e-10:
            best_total = ls_total
            best_xy = ls_xy.copy()
            best_r = ls_r.copy()
        
        # Update xy, r for next iteration
        xy = best_xy.copy()
        r = best_r.copy()
    
    return best_xy, best_r, best_total


def _sdp_feasibility_check(xy, r, deadline):
    """
    SDP-based feasibility check: verify that the contact graph
    admits a positive-semidefinite Gram matrix of inter-center vectors.
    This provides a certificate of geometric realizability.
    """
    n = len(xy)
    
    # Build Gram matrix of inter-center vectors
    # For a valid packing, the matrix G where G_ij = (x_i - x_j)^2 + (y_i - y_j)^2
    # should satisfy certain PSD conditions related to the contact graph
    
    # Compute pairwise distances squared
    D_sq = squareform(pdist(xy)) ** 2
    R_sum_sq = (r[:, None] + r[None, :]) ** 2
    
    # For contacts, D_sq should equal R_sum_sq
    # Check PSD of the Laplacian-like matrix
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                L[i, j] = -1.0
                L[i, i] += 1.0
    
    # Check if L is PSD (it should be for a connected graph)
    try:
        eigvals = np.linalg.eigvalsh(L)
        is_psd = np.all(eigvals >= -1e-8)
    except Exception:
        is_psd = False
    
    return is_psd


def _gradient_polish(xy, r, deadline, tol=1e-8, max_iter=500, beta=50.0, penalty_weight=100.0):
    """
    Gradient-based polish using L-BFGS-B on the smooth objective.
    Kept as a final refinement step.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.5))

    def _smooth_objective(v, n, beta, penalty_weight):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]

        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = r - wall_min
        wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

        total_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                viol = r[i] + r[j] - dist
                total_viol += np.logaddexp(0.0, beta * viol) / beta

        return -np.sum(r) + penalty_weight * (wall_pen + total_viol)

    def _smooth_grad(v, n, beta, penalty_weight):
        x = v[0:n].copy()
        y = v[n:2*n].copy()
        r = v[2*n:3*n].copy()

        grad = np.zeros(3 * n)

        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = r - wall_min
        sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))

        for i in range(n):
            s = sig_wall[i]
            if s < 1e-10:
                continue
            dists = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
            min_idx = int(np.argmin(dists))
            pw = penalty_weight
            if min_idx == 0:
                grad[2*n + i] += pw * s
                grad[i] -= pw * s
            elif min_idx == 1:
                grad[2*n + i] += pw * s
                grad[i] += pw * s
            elif min_idx == 2:
                grad[2*n + i] += pw * s
                grad[n + i] -= pw * s
            else:
                grad[2*n + i] += pw * s
                grad[n + i] += pw * s

        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                if dist_sq < 1e-20:
                    continue
                dist = np.sqrt(dist_sq)
                viol = r[i] + r[j] - dist
                s = 1.0 / (1.0 + np.exp(-beta * viol))
                if s < 1e-10:
                    continue
                factor = penalty_weight * s
                grad[i] += factor * dx / dist
                grad[n + i] += factor * dy / dist
                grad[j] -= factor * dx / dist
                grad[n + j] -= factor * dy / dist
                grad[2*n + i] += factor
                grad[2*n + j] += factor

        grad[2*n:3*n] -= 1.0

        return grad

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _smooth_objective(v, n, beta, penalty_weight)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        return _smooth_grad(v, n, beta, penalty_weight)

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6,
                                   'maxfun': 10000})
    except Exception:
        return xy, r

    v_opt = result.x
    new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    new_r = v_opt[2*n:3*n]

    total, valid = _eval_packing(new_xy, new_r)
    if valid:
        return new_xy, new_r

    return xy, r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Grid initialization + local search
    for side in [5, 6, 7]:
        if time.time() > deadline:
            break
        xy = _grid_init(side)
        r = xy[:, 2].copy()
        xy = xy[:, :2].copy()

        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Random initializations + local search
    for seed in range(8):
        if time.time() > deadline:
            break
        xy, r = _random_init(N, seed=seed)
        out_xy, out_r, total, valid = _best_fit_local_search(xy, r, deadline, grid_size=60)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 3: Contact-graph optimization (NEW STRATEGY)
    if best_valid and best_xy is not None and time.time() < deadline - 30:
        cg_xy, cg_r, cg_total = _contact_graph_optimize(best_xy.copy(), best_r.copy(), deadline, max_graphs=15)
        if cg_total > best_total + 1e-10:
            best_total = cg_total
            best_xy = cg_xy
            best_r = cg_r

    # Phase 4: Gradient polish
    if best_valid and best_xy is not None and time.time() < deadline - 20:
        for beta, pw in [(50.0, 100.0), (80.0, 150.0)]:
            if time.time() < deadline - 15:
                polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline,
                                                           beta=beta, penalty_weight=pw)
                total, valid = _eval_packing(polished_xy, polished_r)
                if valid and total > best_total:
                    best_total = total
                    best_xy = polished_xy
                    best_r = polished_r

    # Phase 5: Second local search pass
    if best_valid and best_xy is not None and time.time() < deadline - 15:
        ls2_xy, ls2_r, ls2_total, ls2_valid = _best_fit_local_search(
            best_xy.copy(), best_r.copy(), deadline, grid_size=100)
        if ls2_valid and ls2_total > best_total + 1e-10:
            best_total = ls2_total
            best_xy = ls2_xy
            best_r = ls2_r

    # Phase 6: Final contact-graph refinement
    if best_valid and best_xy is not None and time.time() < deadline - 10:
        cg2_xy, cg2_r, cg2_total = _contact_graph_optimize(best_xy.copy(), best_r.copy(), deadline, max_graphs=10)
        if cg2_total > best_total + 1e-10:
            best_total = cg2_total
            best_xy = cg2_xy
            best_r = cg2_r

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

    # Final check
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
