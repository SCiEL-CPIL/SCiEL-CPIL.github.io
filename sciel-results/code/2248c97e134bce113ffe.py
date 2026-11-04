"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize, fsolve
from scipy.spatial.distance import cdist


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
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


def _solve_contact_system(contact_pairs, wall_contacts, n):
    """
    Given a contact graph (pairs of touching circles) and wall contacts,
    solve for positions and radii using fsolve.
    
    contact_pairs: list of (i, j) where circle i touches circle j
    wall_contacts: list of (i, wall) where wall is 0=left,1=right,2=bottom,3=top
    
    Returns (xy, r, success) or (None, None, False)
    """
    # Variables: x_i, y_i, r_i for each circle -> 3n variables
    # But we can reduce: fix one circle's position to break translational symmetry
    # Actually, let's just solve the full system with constraints
    
    # Number of equations:
    # - For each contact pair (i,j): dist_ij = r_i + r_j  (1 eq per pair)
    # - For each wall contact (i, w): the circle touches that wall (1 eq per contact)
    # - We need exactly 3n equations for 3n unknowns
    # - If we have fewer, the system is underdetermined; if more, overdetermined
    
    n_pairs = len(contact_pairs)
    n_walls = len(wall_contacts)
    n_eqs = n_pairs + n_walls
    
    if n_eqs != 3 * n:
        # Underdetermined or overdetermined - skip
        return None, None, False
    
    def equations(v):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        eqs = []
        
        # Contact constraints
        for i, j in contact_pairs:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            eqs.append(dist - r[i] - r[j])
        
        # Wall constraints
        for i, w in wall_contacts:
            if w == 0:    # left wall
                eqs.append(x[i] - r[i])
            elif w == 1:  # right wall
                eqs.append(1.0 - x[i] - r[i])
            elif w == 2:  # bottom wall
                eqs.append(y[i] - r[i])
            elif w == 3:  # top wall
                eqs.append(1.0 - y[i] - r[i])
        
        return np.array(eqs)
    
    # Initial guess: spread out in the unit square
    rng = np.random.RandomState(42)
    x0 = rng.uniform(0.1, 0.9, n)
    y0 = rng.uniform(0.1, 0.9, n)
    r0 = rng.uniform(0.03, 0.06, n)
    v0 = np.concatenate([x0, y0, r0])
    
    try:
        sol, info, ier, msg = fsolve(equations, v0, full_output=True, maxfev=10000)
        if ier != 1:
            return None, None, False
        
        x = sol[0:n]
        y = sol[n:2*n]
        r = sol[2*n:3*n]
        
        # Check validity
        if np.any(r < 0.001) or np.any(r > 0.5):
            return None, None, False
        if np.any(x < -0.01) or np.any(x > 1.01) or np.any(y < -0.01) or np.any(y > 1.01):
            return None, None, False
        
        xy = np.column_stack([x, y])
        return xy, r, True
    except Exception:
        return None, None, False


def _generate_random_contact_graph(n, num_pairs, num_walls, seed=0):
    """
    Generate a random contact graph with specified number of circle-circle contacts
    and circle-wall contacts.
    """
    rng = np.random.RandomState(seed)
    
    # Random circle-circle contacts
    all_pairs = list(itertools.combinations(range(n), 2))
    rng.shuffle(all_pairs)
    contact_pairs = all_pairs[:num_pairs]
    
    # Random wall contacts
    wall_contacts = []
    for i in range(n):
        if len(wall_contacts) >= num_walls:
            break
        w = rng.randint(0, 4)
        wall_contacts.append((i, w))
    
    return contact_pairs, wall_contacts


def _local_search_refine(xy, r, deadline, grid_size=50):
    """
    Local search: iteratively move each circle to best position/radius.
    """
    n = len(xy)
    best_total, best_valid = _eval_packing(xy, r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    candidates = np.column_stack([GX.ravel(), GY.ravel()])
    M = len(candidates)
    
    wall_clear = np.minimum(np.minimum(candidates[:, 0], 1.0 - candidates[:, 0]),
                            np.minimum(candidates[:, 1], 1.0 - candidates[:, 1]))
    
    max_sweeps = 10
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
            
            if new_r > r[i] + 1e-10:
                xy[i, 0] = new_x
                xy[i, 1] = new_y
                r[i] = new_r
                improved = True
        
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total + 1e-10:
            best_total = total
            best_xy = xy.copy()
            best_r = r.copy()
        
        if not improved:
            break
    
    return best_xy, best_r, best_total, best_valid


def _smooth_objective(v, n, beta=50.0):
    """Smooth objective for gradient-based polish."""
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
    
    return -np.sum(r) + 100.0 * (wall_pen + total_viol)


def _gradient_polish(xy, r, deadline):
    """Gradient-based polish using L-BFGS-B."""
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])
    
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.5))
    
    beta = 100.0
    
    def obj(v):
        if time.time() > deadline:
            return float('inf')
        return _smooth_objective(v, n, beta)
    
    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        # Numerical gradient (slower but simpler)
        eps = 1e-7
        g = np.zeros(3 * n)
        for k in range(3 * n):
            v_plus = v.copy()
            v_plus[k] += eps
            v_minus = v.copy()
            v_minus[k] -= eps
            g[k] = (obj(v_plus) - obj(v_minus)) / (2 * eps)
        return g
    
    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8,
                                   'maxfun': 50000})
        v_opt = result.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]
        
        total, valid = _eval_packing(new_xy, new_r)
        if valid:
            return new_xy, new_r
    except Exception:
        pass
    
    return xy, r


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


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

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

        out_xy, out_r, total, valid = _local_search_refine(xy, r, deadline, grid_size=60)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Contact graph search - try different topologies
    # For n=26, we need 3*26=78 equations
    # Typical contact graphs for circle packing have ~2n-5 contacts (for planar graphs)
    # So ~47 circle-circle contacts and ~31 wall contacts (but max 4 walls per circle, so 26*4=104 max)
    # Let's try various combinations
    
    n = N
    # Target: 3n = 78 equations
    # Try different splits between circle-circle and wall contacts
    for cc_contacts in [40, 45, 50]:
        if time.time() > deadline - 30:
            break
        wall_contacts_needed = 3 * n - cc_contacts
        if wall_contacts_needed < 0 or wall_contacts_needed > 4 * n:
            continue
        
        # Try multiple random topologies
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            
            contact_pairs, wall_contacts = _generate_random_contact_graph(
                n, cc_contacts, wall_contacts_needed, seed=seed + 100)
            
            xy, r, success = _solve_contact_system(contact_pairs, wall_contacts, n)
            
            if success:
                # Refine with local search
                out_xy, out_r, total, valid = _local_search_refine(xy, r, deadline, grid_size=40)
                if valid and total > best_total:
                    best_total = total
                    best_valid = True
                    best_xy = out_xy
                    best_r = out_r

    # Phase 3: Gradient polish on best result
    if best_valid and best_xy is not None:
        if time.time() < deadline - 25:
            polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline)
            total, valid = _eval_packing(polished_xy, polished_r)
            if valid and total > best_total:
                best_total = total
                best_xy = polished_xy
                best_r = polished_r

        # Phase 4: Second local search pass
        if best_valid and best_xy is not None and time.time() < deadline - 15:
            ls2_xy, ls2_r, ls2_total, ls2_valid = _local_search_refine(
                best_xy.copy(), best_r.copy(), deadline, grid_size=80)
            if ls2_valid and ls2_total > best_total + 1e-10:
                best_total = ls2_total
                best_xy = ls2_xy
                best_r = ls2_r

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
