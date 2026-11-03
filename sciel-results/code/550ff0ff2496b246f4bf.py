import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree

N = 50
DEADLINE = 145.0

def _score_vec(points):
    """Compute r = min(min_{i<j} |c_i - c_j|/2, min_i (1 - |c_i|))."""
    points = points.reshape(N, 2)
    norms = np.linalg.norm(points, axis=1)
    r = 1.0 - np.max(norms)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(N, 1)
    min_pair = np.min(dists[iu]) / 2.0
    r = min(r, min_pair)
    return r

def _get_neighbors(points, max_dist=None):
    """Get neighbor pairs using Delaunay triangulation for efficiency."""
    if max_dist is None:
        # Estimate from typical spacing
        max_dist = 0.3
    tree = cKDTree(points)
    pairs = tree.query_pairs(max_dist, output_type='ndarray')
    return pairs

def _barrier_objective(x, r_target, beta=50.0):
    """
    Smooth barrier objective: minimize sum of violations where
    d_ij < 2*r_target or |c_i| + r_target > 1.
    Uses smooth min via log-sum-exp.
    """
    pts = x.reshape(N, 2)
    
    # Boundary violations: |c_i| + r_target - 1 > 0
    norms = np.linalg.norm(pts, axis=1)
    b_viol = norms + r_target - 1.0
    
    # Pair violations: |c_i - c_j| - 2*r_target < 0
    # Use Delaunay for efficiency
    try:
        delaunay = Delaunay(pts)
        edges = delaunay.simplices
        # Get unique edges
        edge_set = set()
        for tri in edges:
            for i in range(3):
                a, b = tri[i], tri[(i+1) % 3]
                if a < b:
                    edge_set.add((a, b))
                else:
                    edge_set.add((b, a))
        edges = np.array(list(edge_set), dtype=int)
    except Exception:
        edges = np.triu_indices(N, 1)
    
    if len(edges) == 0:
        edges = np.triu_indices(N, 1)
    
    ci = pts[edges[:, 0]]
    cj = pts[edges[:, 1]]
    d = np.sqrt(np.sum((ci - cj) ** 2, axis=1))
    p_viol = 2.0 * r_target - d
    
    all_viol = np.concatenate([b_viol, p_viol])
    
    # Smooth max of violations (we want all <= 0)
    # Use log-sum-exp for smooth max
    m = np.max(all_viol)
    if m < -1e-10:
        return 0.0  # Feasible
    # Smooth approximation of max
    s = np.log(np.sum(np.exp(beta * (all_viol - m)))) / beta + m
    return max(0.0, s)

def _barrier_grad(x, r_target, beta=50.0):
    """Gradient of barrier objective."""
    pts = x.reshape(N, 2)
    grad = np.zeros_like(pts)
    
    norms = np.linalg.norm(pts, axis=1)
    b_viol = norms + r_target - 1.0
    
    try:
        delaunay = Delaunay(pts)
        edges = delaunay.simplices
        edge_set = set()
        for tri in edges:
            for i in range(3):
                a, b = tri[i], tri[(i+1) % 3]
                if a < b:
                    edge_set.add((a, b))
                else:
                    edge_set.add((b, a))
        edges = np.array(list(edge_set), dtype=int)
    except Exception:
        edges = np.triu_indices(N, 1)
    
    if len(edges) == 0:
        edges = np.triu_indices(N, 1)
    
    ci = pts[edges[:, 0]]
    cj = pts[edges[:, 1]]
    d = np.sqrt(np.sum((ci - cj) ** 2, axis=1))
    p_viol = 2.0 * r_target - d
    
    all_viol = np.concatenate([b_viol, p_viol])
    m = np.max(all_viol)
    
    if m < -1e-10:
        return grad.flatten()
    
    weights = np.exp(beta * (all_viol - m))
    weights /= np.sum(weights)
    
    # Boundary gradient
    for i in range(N):
        w = weights[i]
        if norms[i] > 1e-12:
            grad[i] += w * pts[i] / norms[i]
    
    # Pair gradients
    for idx in range(len(edges)):
        w = weights[N + idx]
        i, j = edges[idx, 0], edges[idx, 1]
        d_vec = pts[i] - pts[j]
        d_norm = np.linalg.norm(d_vec)
        if d_norm > 1e-12:
            unit = d_vec / d_norm
            grad[i] -= w * unit
            grad[j] += w * unit
    
    return grad.flatten()

def _optimize_for_radius(x0, r_target, time_limit=10.0):
    """Optimize positions to achieve radius r_target. Returns (points, success)."""
    t_start = time.time()
    
    def obj(x):
        return _barrier_objective(x, r_target)
    
    def grad(x):
        return _barrier_grad(x, r_target)
    
    try:
        res = minimize(
            obj, x0.flatten(), jac=grad,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 1000}
        )
        if res.fun < 1e-8:
            return res.x.reshape(N, 2), True
    except Exception:
        pass
    
    return x0.reshape(N, 2), False

def _binary_search_radius(x0, r_lo, r_hi, time_limit=30.0):
    """Binary search for maximum feasible radius."""
    t_start = time.time()
    best_pts = x0.reshape(N, 2)
    best_r = r_lo
    
    while time.time() - t_start < time_limit and (r_hi - r_lo) > 1e-12:
        r_mid = (r_lo + r_hi) / 2.0
        pts, success = _optimize_for_radius(best_pts, r_mid, time_limit=min(8.0, time_limit - (time.time() - t_start)))
        
        if success:
            r_lo = r_mid
            best_pts = pts
            best_r = r_mid
        else:
            r_hi = r_mid
        
        if time.time() - t_start > time_limit - 5:
            break
    
    return best_pts, best_r

def _ring_init_v2(n, structure=None):
    """
    Generate ring-based initial configuration with multiple structures.
    Known structures for 50 circles:
    - 1+6+12+18+13 (concentric rings)
    - Other arrangements
    """
    if structure is None:
        structure = [1, 6, 12, 18, 13]
    
    points = []
    # Estimate ring radii based on target radius
    # For 50 circles in unit circle, r ~ 0.126
    # Ring radii roughly: 0, 0.35, 0.65, 0.9, 1.0
    
    total = sum(structure)
    if total != n:
        # Adjust
        while sum(structure) > n:
            structure[-1] -= 1
            if structure[-1] <= 0:
                structure.pop()
        while sum(structure) < n:
            structure[-1] += 1
    
    # Use geometric spacing for ring radii
    n_rings = len(structure)
    if n_rings == 1:
        ring_radii = [0.0]
    else:
        # Spread rings from 0 to ~0.95
        ring_radii = np.linspace(0.0, 0.92, n_rings).tolist()
    
    for ri, (size, radius) in enumerate(zip(structure, ring_radii)):
        for k in range(size):
            if size == 1:
                points.append([0.0, 0.0])
            else:
                theta = 2 * np.pi * k / size + (np.pi / size if ri % 2 == 0 else 0)
                points.append([radius * np.cos(theta), radius * np.sin(theta)])
    
    return np.array(points, dtype=float)

def _hex_init(n):
    """Hexagonal lattice initialization."""
    best_points = None
    best_r = 0.0
    for spacing in np.linspace(0.1, 0.25, 30):
        s = spacing
        points = []
        max_coord = int(1.2 / s) + 2
        for j in range(-max_coord, max_coord + 1):
            for i in range(-max_coord, max_coord + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3) / 2
                if np.hypot(x, y) <= 0.95:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= n:
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)[:n]
            cand = points[idx]
            r = _score_vec(cand)
            if r > best_r:
                best_r = r
                best_points = cand.copy()
    if best_points is None:
        best_points = np.random.randn(n, 2) * 0.3
    return best_points

def _refine_positions(points, time_limit=15.0):
    """Refine positions using L-BFGS on smooth objective (maximize soft-min)."""
    t_start = time.time()
    
    def soft_min_obj(x):
        pts = x.reshape(N, 2)
        norms = np.linalg.norm(pts, axis=1)
        b = 1.0 - norms
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu] / 2.0
        all_vals = np.concatenate([d, b])
        beta = 100.0
        m = np.min(all_vals)
        return -(m + np.log(np.sum(np.exp(-beta * (all_vals - m)))) / beta)
    
    def soft_min_grad(x):
        pts = x.reshape(N, 2)
        norms = np.linalg.norm(pts, axis=1)
        b = 1.0 - norms
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu] / 2.0
        all_vals = np.concatenate([d, b])
        beta = 100.0
        m = np.min(all_vals)
        weights = np.exp(-beta * (all_vals - m))
        weights /= np.sum(weights)
        
        grad = np.zeros((N, 2))
        
        # Pair contributions
        for idx in range(len(d)):
            w = weights[idx]
            i, j = iu[0][idx], iu[1][idx]
            d_vec = pts[i] - pts[j]
            d_norm = np.linalg.norm(d_vec)
            if d_norm > 1e-12:
                unit = d_vec / d_norm
                grad[i] += w * unit / 2.0
                grad[j] -= w * unit / 2.0
        
        # Boundary contributions
        for i in range(N):
            w = weights[N + i]
            if norms[i] > 1e-12:
                grad[i] -= w * pts[i] / norms[i]
        
        return -grad.flatten()
    
    try:
        res = minimize(
            soft_min_obj, points.flatten(), jac=soft_min_grad,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13}
        )
        if res.fun < soft_min_obj(points.flatten()):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    
    return points

def _polish(points, time_limit=5.0):
    """High-precision polish using subgradient of exact min."""
    pts = points.flatten().copy()
    t_start = time.time()
    best_pts = pts.copy()
    best_score = _score_vec(best_pts)
    
    lr = 0.002
    momentum = 0.9
    v = np.zeros_like(pts)
    no_improve = 0
    
    while time.time() - t_start < time_limit:
        c = pts.reshape(N, 2)
        norms = np.linalg.norm(c, axis=1)
        b = 1.0 - norms
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        iu = np.triu_indices(N, 1)
        d = dists[iu] / 2.0
        
        r = min(np.min(b), np.min(d))
        
        if r > best_score + 1e-15:
            best_score = r
            best_pts = pts.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 50:
                lr *= 0.5
                v *= 0.5
                no_improve = 0
                if lr < 1e-9:
                    break
        
        grad = np.zeros((N, 2))
        
        # Subgradient: average of active constraints
        active_b = np.abs(b - r) < 1e-10
        active_d = np.abs(d - r) < 1e-10
        
        n_active = np.sum(active_b) + np.sum(active_d)
        if n_active == 0:
            break
        
        for i in range(N):
            if active_b[i]:
                if norms[i] > 1e-12:
                    grad[i] -= c[i] / norms[i]
        
        for idx in range(len(d)):
            if active_d[idx]:
                i, j = iu[0][idx], iu[1][idx]
                d_vec = c[i] - c[j]
                d_norm = np.linalg.norm(d_vec)
                if d_norm > 1e-12:
                    unit = d_vec / d_norm
                    grad[i] += unit / 2.0
                    grad[j] -= unit / 2.0
        
        grad /= n_active
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        grad /= grad_norm
        
        v = momentum * v + lr * grad
        pts = pts + v
        
        c_new = pts.reshape(N, 2)
        norms_new = np.linalg.norm(c_new, axis=1)
        for i in range(N):
            if norms_new[i] > 0.9999:
                c_new[i] *= 0.9999 / norms_new[i]
                pts[i*2:(i+1)*2] = c_new[i]
    
    return best_pts.reshape(N, 2), best_score

def _simulated_annealing(points, time_limit=20.0):
    """Simulated annealing to escape local minima."""
    t_start = time.time()
    current = points.copy()
    current_score = _score_vec(current)
    best = current.copy()
    best_score = current_score
    
    T = 0.01  # Initial temperature
    cooling = 0.995
    
    while time.time() - t_start < time_limit and T > 1e-5:
        # Propose perturbation
        n_pert = np.random.randint(1, 5)
        idx = np.random.choice(N, n_pert, replace=False)
        delta = np.random.randn(n_pert, 2) * T * 0.5
        proposed = current.copy()
        proposed[idx] += delta
        
        # Clamp to unit circle
        norms = np.linalg.norm(proposed, axis=1)
        for i in range(N):
            if norms[i] > 0.999:
                proposed[i] *= 0.999 / norms[i]
        
        prop_score = _score_vec(proposed)
        
        # Accept if better or with probability exp((new-old)/T)
        if prop_score > current_score:
            current = proposed
            current_score = prop_score
            if prop_score > best_score:
                best = proposed.copy()
                best_score = prop_score
        else:
            delta_e = prop_score - current_score
            if np.random.rand() < np.exp(delta_e / (T * 0.01)):
                current = proposed
                current_score = prop_score
        
        T *= cooling
    
    return best, best_score

def construct():
    t_start = time.time()
    np.random.seed(42)
    
    # Multiple initializations
    inits = []
    
    # Ring structures
    structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 16, 17],
        [1, 8, 12, 15, 14],
        [1, 6, 12, 14, 17],
    ]
    for s in structures:
        pts = _ring_init_v2(N, s)
        inits.append(pts)
    
    # Hex init
    inits.append(_hex_init(N))
    
    # Random init
    inits.append(np.random.randn(N, 2) * 0.4)
    
    # Select best initial
    best_points = None
    best_score = -1
    for cand in inits:
        r = _score_vec(cand)
        if r > best_score:
            best_score = r
            best_points = cand.copy()
    
    # Refine the best initial
    if time.time() < DEADLINE - 60:
        refined = _refine_positions(best_points, time_limit=min(20.0, DEADLINE - (time.time() - t_start) - 60))
        r_ref = _score_vec(refined)
        if r_ref > best_score:
            best_score = r_ref
            best_points = refined
    
    # Binary search on radius
    if time.time() < DEADLINE - 40:
        # Start from current best, search for improvement
        r_lo = best_score * 0.99
        r_hi = best_score * 1.05
        pts_bs, r_bs = _binary_search_radius(best_points, r_lo, r_hi, time_limit=min(30.0, DEADLINE - (time.time() - t_start) - 40))
        if r_bs > best_score:
            best_score = r_bs
            best_points = pts_bs
    
    # Simulated annealing to escape local minima
    if time.time() < DEADLINE - 20:
        sa_pts, sa_score = _simulated_annealing(best_points, time_limit=min(15.0, DEADLINE - (time.time() - t_start) - 20))
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts
    
    # Final refinement
    if time.time() < DEADLINE - 10:
        refined2 = _refine_positions(best_points, time_limit=min(8.0, DEADLINE - (time.time() - t_start) - 10))
        r_ref2 = _score_vec(refined2)
        if r_ref2 > best_score:
            best_score = r_ref2
            best_points = refined2
    
    # Polish
    if time.time() < DEADLINE - 3:
        polished, r_polish = _polish(best_points, time_limit=min(5.0, DEADLINE - (time.time() - t_start) - 3))
        if r_polish > best_score:
            best_score = r_polish
            best_points = polished
    
    # Ensure validity
    c_final = best_points.reshape(N, 2)
    r2 = np.linalg.norm(c_final, axis=1)
    for i in range(N):
        if r2[i] > 1.0 - best_score + 1e-9:
            c_final[i] *= (1.0 - best_score - 1e-10) / r2[i]
    
    r_check = _score_vec(c_final)
    if r_check < best_score - 1e-10:
        # Slightly shrink
        c_final *= (1.0 - 1e-8)
    
    return c_final
