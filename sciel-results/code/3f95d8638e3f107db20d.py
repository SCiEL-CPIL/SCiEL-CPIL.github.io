"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
_GLOBAL_START = None
_GLOBAL_DEADLINE = None

# EVOLVE-BLOCK-START

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0) using log-sum-exp."""
    return np.logaddexp(0, z / eps) * eps

def _objective(params, penalty_weight, smooth_eps=1e-4, contact_edges=None):
    """
    Vectorized objective for N=26 circles.
    params: flat array [x0,y0,r0, x1,y1,r1, ...]
    contact_edges: optional list of (i,j) pairs to check for overlap (None = all pairs)
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary violations (smooth)
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    penalty += np.sum(_softplus(lx, smooth_eps)**2)
    penalty += np.sum(_softplus(ux, smooth_eps)**2)
    penalty += np.sum(_softplus(ly, smooth_eps)**2)
    penalty += np.sum(_softplus(uy, smooth_eps)**2)
    
    # Negative radius
    penalty += np.sum(_softplus(-r, smooth_eps)**2)
    
    # Overlap constraints
    n = len(r)
    if contact_edges is not None and len(contact_edges) > 0:
        for (i, j) in contact_edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(max(dx*dx + dy*dy, 1e-12))
            overlap = r[i] + r[j] - dist
            penalty += _softplus(overlap, smooth_eps)**2
    else:
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _objective_grad(params, penalty_weight, smooth_eps=1e-4, contact_edges=None):
    """
    Analytic gradient of the objective.
    """
    x = params[0::3].copy()
    y = params[1::3].copy()
    r = params[2::3].copy()
    n = len(r)
    
    grad = np.zeros(3 * n)
    
    # Boundary gradients
    def _sp_grad(z, eps):
        # d/dz of softplus(z, eps) = sigmoid(z/eps)
        return 1.0 / (1.0 + np.exp(-z / eps))
    
    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0
    
    # d/dx of softplus(r-x)^2 = 2*softplus(r-x)*(-1)*sigmoid((r-x)/eps)
    sp_lx = _softplus(lx, smooth_eps)
    g_lx = 2 * sp_lx * _sp_grad(lx, smooth_eps)
    grad[0::3] -= g_lx  # d/dx
    grad[2::3] += g_lx  # d/dr
    
    sp_ux = _softplus(ux, smooth_eps)
    g_ux = 2 * sp_ux * _sp_grad(ux, smooth_eps)
    grad[0::3] += g_ux
    grad[2::3] += g_ux
    
    sp_ly = _softplus(ly, smooth_eps)
    g_ly = 2 * sp_ly * _sp_grad(ly, smooth_eps)
    grad[1::3] -= g_ly
    grad[2::3] += g_ly
    
    sp_uy = _softplus(uy, smooth_eps)
    g_uy = 2 * sp_uy * _sp_grad(uy, smooth_eps)
    grad[1::3] += g_uy
    grad[2::3] += g_uy
    
    # Negative radius
    sp_nr = _softplus(-r, smooth_eps)
    g_nr = 2 * sp_nr * _sp_grad(-r, smooth_eps)
    grad[2::3] -= g_nr
    
    # Objective: -sum(r)
    grad[2::3] -= 1.0
    
    # Overlap gradients
    if contact_edges is not None and len(contact_edges) > 0:
        for (i, j) in contact_edges:
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(max(dist_sq, 1e-12))
            overlap = r[i] + r[j] - dist
            sp_ov = _softplus(overlap, smooth_eps)
            g_ov = 2 * sp_ov * _sp_grad(overlap, smooth_eps)
            
            if dist > 1e-12:
                # d/dx_i of overlap = -dx/dist
                # d/dx_i of sp(overlap)^2 = g_ov * (-dx/dist)
                grad[3*i] += g_ov * (-dx / dist)
                grad[3*i+1] += g_ov * (-dy / dist)
                grad[3*i+2] += g_ov
                grad[3*j] += g_ov * (dx / dist)
                grad[3*j+1] += g_ov * (dy / dist)
                grad[3*j+2] += g_ov
    else:
        idx_i, idx_j = np.triu_indices(n, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        sp_ov = _softplus(overlap, smooth_eps)
        g_ov = 2 * sp_ov * _sp_grad(overlap, smooth_eps)
        
        inv_dist = np.where(dist > 1e-12, 1.0/dist, 0.0)
        gx = g_ov * dx * inv_dist
        gy = g_ov * dy * inv_dist
        
        for k in range(len(idx_i)):
            i = idx_i[k]
            j = idx_j[k]
            grad[3*i] -= gx[k]
            grad[3*i+1] -= gy[k]
            grad[3*i+2] += g_ov[k]
            grad[3*j] += gx[k]
            grad[3*j+1] += gy[k]
            grad[3*j+2] += g_ov[k]
    
    return penalty_weight * grad

def _validate(xyz, tol=1e-7):
    """Check validity."""
    n = xyz.shape[0]
    x, y, r = xyz[:,0], xyz[:,1], xyz[:,2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i]-x[j]
            dy = y[i]-y[j]
            dist = np.sqrt(dx*dx+dy*dy)
            if dist < r[i]+r[j]-tol:
                return False
    return True

def _remaining_time():
    """Get remaining time before deadline."""
    if _GLOBAL_DEADLINE is None:
        return 140.0
    return max(0.0, _GLOBAL_DEADLINE - time.time())

def _fix_boundaries(xyz):
    """Project circles inside unit square."""
    xyz = xyz.copy()
    for i in range(N):
        r = max(0, xyz[i,2])
        xyz[i,0] = np.clip(xyz[i,0], r, 1-r) if r < 0.5 else 0.5
        xyz[i,1] = np.clip(xyz[i,1], r, 1-r) if r < 0.5 else 0.5
        xyz[i,2] = r
    return xyz

def _get_contact_edges(xyz, margin=1.2):
    """Get contact edges from Delaunay triangulation with margin filter."""
    centers = xyz[:, :2]
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for k in range(3):
                a, b = simplex[k], simplex[(k + 1) % 3]
                edges.add((min(a, b), max(a, b)))
        # Filter edges by distance threshold
        filtered = []
        for (i, j) in edges:
            dx = xyz[i,0] - xyz[j,0]
            dy = xyz[i,1] - xyz[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            threshold = margin * (xyz[i,2] + xyz[j,2])
            if dist < threshold:
                filtered.append((i, j))
        return filtered if filtered else None
    except Exception:
        return None

def _initial_hex():
    """Hexagonal packing initial config."""
    cols = 5
    rows = 6
    positions = []
    r = 0.09
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5/cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r*0.5])
    return np.array(positions[:N])

def _initial_grid():
    """Grid initial config."""
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5/side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i+0.5)/side
            y = (j+0.5)/side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r*0.5])
    return np.array(positions[:N])

def _initial_corner_weighted(seed=0):
    """Random config with corner preference."""
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
               [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.08])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])

def _initial_ring(seed=0):
    """Ring-shaped initial config."""
    rng = np.random.default_rng(seed)
    positions = [[0.5, 0.5, 0.08]]
    for ring, n_circles, radius in [(1, 6, 0.10), (2, 10, 0.08), (3, 9, 0.07)]:
        for k in range(n_circles):
            angle = 2*np.pi*k/n_circles + rng.uniform(-0.1, 0.1)
            dist = ring * 0.14
            x = 0.5 + dist*np.cos(angle)
            y = 0.5 + dist*np.sin(angle)
            positions.append([x, y, radius])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([rng.uniform(0.1,0.9), rng.uniform(0.1,0.9), 0.06])
    return np.array(positions[:N])

def _optimize_single(xyz0):
    """Optimize from a single initial config with adaptive penalty, analytic gradients, and Delaunay contact graph."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    def _make_obj(pw, eps, edges):
        def obj(p):
            return _objective(p, pw, eps, edges)
        def grad(p):
            return _objective_grad(p, pw, eps, edges)
        return obj, grad
    
    # Phase 1: Low penalty, full pairwise, focus on layout
    penalty = 5.0
    remaining = _remaining_time()
    if remaining > 10:
        try:
            obj, grad = _make_obj(penalty, 1e-3, None)
            res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 1000})
            if res.fun < _objective(params, penalty, 1e-3, None):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: Medium penalty, full pairwise
    remaining = _remaining_time()
    if remaining > 8:
        penalty = 50.0
        try:
            obj, grad = _make_obj(penalty, 1e-4, None)
            res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 350, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 1200})
            if res.fun < _objective(params, penalty, 1e-4, None):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 3: High penalty, start using Delaunay contact graph
    remaining = _remaining_time()
    if remaining > 6:
        penalty = 500.0
        xyz_current = params.reshape(N, 3)
        edges = _get_contact_edges(xyz_current, margin=1.3)
        try:
            obj, grad = _make_obj(penalty, 1e-5, edges)
            res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 1500})
            if res.fun < _objective(params, penalty, 1e-5, edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 4: Very high penalty, refined Delaunay contact graph
    remaining = _remaining_time()
    if remaining > 4:
        penalty = 5000.0
        xyz_current = params.reshape(N, 3)
        edges = _get_contact_edges(xyz_current, margin=1.2)
        try:
            obj, grad = _make_obj(penalty, 1e-6, edges)
            res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 350, 'ftol': 1e-16, 'gtol': 1e-15, 'maxfun': 1200})
            if res.fun < _objective(params, penalty, 1e-6, edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Ultra-high penalty, tightest contact graph, final polish
    remaining = _remaining_time()
    if remaining > 2:
        penalty = 50000.0
        xyz_current = params.reshape(N, 3)
        edges = _get_contact_edges(xyz_current, margin=1.1)
        try:
            obj, grad = _make_obj(penalty, 1e-7, edges)
            res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-15, 'maxfun': 1000})
            if res.fun < _objective(params, penalty, 1e-7, edges):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 6: Final high-precision polish with full pairwise check
    remaining = _remaining_time()
    if remaining > 1:
        penalty = 100000.0
        try:
            obj, grad = _make_obj(penalty, 1e-8, None)
            res = minimize(obj, params, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-16, 'maxfun': 800})
            if res.fun < _objective(params, penalty, 1e-8, None):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_score

def _anneal_refine(xyz, rng, time_budget=5.0):
    """
    Simulated-annealing style local search: perturb positions and radii,
    accept if valid and better, or with probability exp(-delta/T).
    """
    best_xyz = xyz.copy()
    best_score = np.sum(best_xyz[:, 2])
    start = time.time()
    T = 0.05
    cooling = 0.97
    
    while time.time() - start < time_budget and T > 1e-6:
        perturbed = best_xyz.copy()
        # Perturb positions and radii
        scale = T * 0.5
        noise = rng.normal(0, scale, perturbed.shape)
        noise[:, 2] *= 0.3  # Less perturbation on radii
        perturbed += noise
        perturbed = _fix_boundaries(perturbed)
        
        if not _validate(perturbed):
            T *= cooling
            continue
        
        score = np.sum(perturbed[:, 2])
        delta = score - best_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            best_xyz = perturbed
            best_score = score
        
        T *= cooling
    
    return best_xyz, best_score

def construct():
    global _GLOBAL_START, _GLOBAL_DEADLINE
    _GLOBAL_START = time.time()
    _GLOBAL_DEADLINE = _GLOBAL_START + 145.0  # 5s safety margin before 150s kill
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate diverse initial configurations
    inits = [
        _initial_grid(),
        _initial_hex(),
        _initial_corner_weighted(0),
        _initial_corner_weighted(1),
        _initial_ring(0),
        _initial_ring(1),
        _initial_corner_weighted(2),
        _initial_ring(2),
    ]
    
    # Optimize from each
    for xyz0 in inits:
        remaining = _remaining_time()
        if remaining < 12.0:
            break
        xyz_opt, score = _optimize_single(xyz0)
        
        if not _validate(xyz_opt):
            xyz_fix = xyz_opt.copy()
            for _ in range(80):
                if _validate(xyz_fix):
                    break
                xyz_fix[:, 2] *= 0.97
                xyz_fix = _fix_boundaries(xyz_fix)
            xyz_opt = xyz_fix
        
        if not _validate(xyz_opt):
            continue
        
        score = np.sum(xyz_opt[:, 2])
        if score > best_score:
            best_score = score
            best_xyz = xyz_opt
    
    # Annealing refinement from the best solution
    remaining = _remaining_time()
    if remaining > 10.0:
        rng = np.random.default_rng(42)
        xyz_ann, score_ann = _anneal_refine(best_xyz, rng, time_budget=min(remaining - 5.0, 8.0))
        if _validate(xyz_ann) and score_ann > best_score:
            best_score = score_ann
            best_xyz = xyz_ann
    
    # Local refinement from the best solution with perturbations
    remaining = _remaining_time()
    if remaining > 15.0:
        rng = np.random.default_rng(123)
        for scale in [0.003, 0.008, 0.015]:
            if remaining < 6.0:
                break
            perturbed = best_xyz.copy()
            noise = rng.normal(0, scale, perturbed.shape)
            noise[:, 2] *= 0.1
            perturbed += noise
            perturbed = _fix_boundaries(perturbed)
            
            xyz_opt, score = _optimize_single(perturbed)
            
            if not _validate(xyz_opt):
                xyz_fix = xyz_opt.copy()
                for _ in range(80):
                    if _validate(xyz_fix):
                        break
                    xyz_fix[:, 2] *= 0.97
                    xyz_fix = _fix_boundaries(xyz_fix)
                xyz_opt = xyz_fix
            
            if not _validate(xyz_opt):
                continue
            
            score = np.sum(xyz_opt[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_opt
            remaining = _remaining_time()
    
    # Final anneal from best
    remaining = _remaining_time()
    if remaining > 5.0:
        rng = np.random.default_rng(777)
        xyz_ann, score_ann = _anneal_refine(best_xyz, rng, time_budget=min(remaining - 2.0, 4.0))
        if _validate(xyz_ann) and score_ann > best_score:
            best_score = score_ann
            best_xyz = xyz_ann
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
