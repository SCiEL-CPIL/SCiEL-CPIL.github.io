"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Delaunay

N = 26

# EVOLVE-BLOCK-START

def _softplus(z, eps=1e-4):
    """Smooth approximation of max(z, 0) using log-sum-exp."""
    return np.logaddexp(0, z / eps) * eps

def _delaunay_pairs(centers):
    """Get Delaunay edges as pairs of indices."""
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a,b), max(a,b)))
        return list(edges)
    except Exception:
        # Fallback: k-NN
        tree = cKDTree(centers)
        dists, indices = tree.query(centers, k=8)
        edges = set()
        for i in range(N):
            for j_idx in range(1, 8):
                j = indices[i, j_idx]
                edges.add((min(i,j), max(i,j)))
        return list(edges)

def _objective_delaunay(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective using Delaunay edges for overlap checking.
    Only checks pairs that are Delaunay neighbors (much fewer than all pairs).
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
    
    # Delaunay-based overlap check
    centers = np.column_stack([x, y])
    pairs = _delaunay_pairs(centers)
    
    if len(pairs) > 0:
        i_arr = np.array([p[0] for p in pairs])
        j_arr = np.array([p[1] for p in pairs])
        dx = x[i_arr] - x[j_arr]
        dy = y[i_arr] - y[j_arr]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[i_arr] + r[j_arr] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    # Also check k-NN to catch near-misses not in Delaunay
    try:
        tree = cKDTree(centers)
        dists, indices = tree.query(centers, k=6)
        for i in range(N):
            for j_idx in range(1, 6):
                j = indices[i, j_idx]
                if j <= i:
                    continue
                overlap = r[i] + r[j] - dists[i, j_idx]
                if overlap > -smooth_eps:  # only if close to overlapping
                    penalty += _softplus(overlap, smooth_eps)**2
    except Exception:
        pass
    
    return -np.sum(r) + penalty_weight * penalty

def _objective_knn(params, penalty_weight, smooth_eps=1e-4, k_neighbors=8):
    """
    Vectorized objective using k-NN for overlap checking (faster than full pairwise).
    params: flat array [x0,y0,r0, x1,y1,r1, ...]
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
    
    # k-NN overlaps
    centers = np.column_stack([x, y])
    try:
        tree = cKDTree(centers)
        # Query k nearest neighbors for each point
        dists, indices = tree.query(centers, k=min(k_neighbors+1, N))
        # dists[i][0] is 0 (self), so skip index 0
        for i in range(N):
            for j_idx in range(1, min(k_neighbors+1, N)):
                j = indices[i, j_idx]
                if j <= i:
                    continue  # avoid double-counting
                overlap = r[i] + r[j] - dists[i, j_idx]
                penalty += _softplus(overlap, smooth_eps)**2
    except Exception:
        # Fallback to full pairwise
        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _objective_full(params, penalty_weight, smooth_eps=1e-4):
    """
    Full pairwise objective for final polish (most accurate but slower).
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
    
    # Full pairwise overlaps
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx*dx + dy*dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    penalty += np.sum(_softplus(overlap, smooth_eps)**2)
    
    return -np.sum(r) + penalty_weight * penalty

def _initial_voronoi_like(seed=0):
    """Voronoi-inspired: place circles at cell centroids of a perturbed grid."""
    rng = np.random.default_rng(seed)
    positions = []
    base_r = 0.085
    for i in range(5):
        for j in range(5):
            if i == 2 and j == 2:
                continue
            x = (i + 0.5) / 5.0 + rng.uniform(-0.02, 0.02)
            y = (j + 0.5) / 5.0 + rng.uniform(-0.02, 0.02)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            positions.append([x, y, base_r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    for _ in range(N - len(positions)):
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        positions.append([x, y, base_r * 0.9])
    return np.array(positions[:N])

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

def _fix_boundaries(xyz):
    """Project circles inside unit square."""
    xyz = xyz.copy()
    for i in range(N):
        r = max(0, xyz[i,2])
        xyz[i,0] = np.clip(xyz[i,0], r, 1-r) if r < 0.5 else 0.5
        xyz[i,1] = np.clip(xyz[i,1], r, 1-r) if r < 0.5 else 0.5
        xyz[i,2] = r
    return xyz

def _initial_hex():
    """Hexagonal packing initial config - fixed spacing."""
    positions = []
    r = 0.09
    # 5 columns, 6 rows with hex offset
    dx = 1.0 / 5.0
    dy = 1.0 / 6.0
    for row in range(6):
        for col in range(5):
            if row % 2 == 1:
                x = (col + 1.0) * dx
            else:
                x = (col + 0.5) * dx
            y = (row + 0.5) * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])

def _initial_grid():
    """Grid initial config - 5x5 with center removed + extras."""
    positions = []
    r = 0.09
    # 5x5 grid = 25 positions, remove center, add 1 more
    for i in range(5):
        for j in range(5):
            if i == 2 and j == 2:
                continue
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 5.0
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    # Add one more circle in the center
    positions.append([0.5, 0.5, r * 0.85])
    return np.array(positions[:N])

def _initial_corner_weighted(seed=0):
    """Structured config with corner/edge preference."""
    rng = np.random.default_rng(seed)
    positions = []
    # Corners with larger radius
    corners = [[0.07, 0.07], [0.93, 0.07], [0.07, 0.93], [0.93, 0.93]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.10])
    # Edge midpoints
    edges = [[0.5, 0.07], [0.5, 0.93], [0.07, 0.5], [0.93, 0.5],
             [0.25, 0.07], [0.75, 0.07], [0.25, 0.93], [0.75, 0.93],
             [0.07, 0.25], [0.07, 0.75], [0.93, 0.25], [0.93, 0.75]]
    for cx, cy in edges:
        positions.append([cx, cy, 0.085])
        if len(positions) >= N:
            break
    # Fill rest randomly in interior
    for _ in range(N - len(positions)):
        x = rng.uniform(0.15, 0.85)
        y = rng.uniform(0.15, 0.85)
        positions.append([x, y, rng.uniform(0.07, 0.095)])
    return np.array(positions[:N])

def _initial_ring(seed=0):
    """Ring-shaped initial config with better spacing."""
    rng = np.random.default_rng(seed)
    positions = [[0.5, 0.5, 0.09]]
    # Ring 1: 6 circles at radius ~0.18 from center
    for k in range(6):
        angle = 2*np.pi*k/6 + 0.1
        dist = 0.19
        x = 0.5 + dist*np.cos(angle)
        y = 0.5 + dist*np.sin(angle)
        x = np.clip(x, 0.08, 0.92)
        y = np.clip(y, 0.08, 0.92)
        positions.append([x, y, 0.09])
    # Ring 2: 10 circles at radius ~0.32 from center
    for k in range(10):
        angle = 2*np.pi*k/10 + 0.15
        dist = 0.33
        x = 0.5 + dist*np.cos(angle)
        y = 0.5 + dist*np.sin(angle)
        x = np.clip(x, 0.07, 0.93)
        y = np.clip(y, 0.07, 0.93)
        positions.append([x, y, 0.08])
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), 0.07])
    return np.array(positions[:N])

def _optimize_single(xyz0, deadline):
    """Optimize from a single initial config with adaptive penalty and Delaunay-based objective."""
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    # Phase 1: Delaunay-based with moderate penalty (fast, broad)
    remaining = deadline - time.time()
    if remaining > 8:
        penalty = 5.0
        try:
            res = minimize(_objective_delaunay, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-3),
                          options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective_delaunay(params, penalty, 1e-3):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: Delaunay with higher penalty
    remaining = deadline - time.time()
    if remaining > 6:
        penalty = 20.0
        try:
            res = minimize(_objective_delaunay, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4),
                          options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-11})
            if res.fun < _objective_delaunay(params, penalty, 1e-4):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 3: k-NN with higher penalty (more thorough constraint checking)
    remaining = deadline - time.time()
    if remaining > 5:
        penalty = 50.0
        try:
            res = minimize(_objective_knn, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-4, 10),
                          options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-11})
            if res.fun < _objective_knn(params, penalty, 1e-4, 10):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 4: Full pairwise with high penalty
    remaining = deadline - time.time()
    if remaining > 4:
        penalty = 200.0
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-5),
                          options={'maxiter': 350, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective_full(params, penalty, 1e-5):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 5: Very high penalty, full pairwise
    remaining = deadline - time.time()
    if remaining > 3:
        penalty = 1000.0
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-6),
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13})
            if res.fun < _objective_full(params, penalty, 1e-6):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 6: Ultra-precise final polish
    remaining = deadline - time.time()
    if remaining > 2:
        penalty = 5000.0
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-7),
                          options={'maxiter': 250, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 30})
            if res.fun < _objective_full(params, penalty, 1e-7):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 7: Final tiny-eps polish
    remaining = deadline - time.time()
    if remaining > 1.5:
        penalty = 10000.0
        try:
            res = minimize(_objective_full, params, method='L-BFGS-B',
                          bounds=bounds, args=(penalty, 1e-8),
                          options={'maxiter': 200, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 40})
            if res.fun < _objective_full(params, penalty, 1e-8):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    return best_params.reshape(N, 3), best_score

def construct():
    global_start = time.time()
    deadline = global_start + 148.0  # 2s safety margin before 150s kill
    
    best_xyz = _initial_grid()
    best_score = np.sum(best_xyz[:, 2])
    
    # Generate diverse initial configurations
    inits = []
    inits.append(_initial_hex())
    inits.append(_initial_corner_weighted(0))
    inits.append(_initial_corner_weighted(1))
    inits.append(_initial_corner_weighted(2))
    inits.append(_initial_ring(0))
    inits.append(_initial_ring(1))
    inits.append(_initial_voronoi_like(0))
    inits.append(_initial_voronoi_like(1))
    
    # Optimize from each
    n_inits = len(inits)
    for idx, xyz0 in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
        # Allocate time: earlier inits get more time
        time_budget = deadline - 5.0 - remaining * 0.05
        xyz_opt, score = _optimize_single(xyz0, time_budget)
        
        # Fix if invalid
        if not _validate(xyz_opt):
            xyz_fix = xyz_opt.copy()
            for _ in range(200):
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
    
    # Final validation
    if not _validate(best_xyz):
        best_xyz = _initial_grid()
    
    return best_xyz

# EVOLVE-BLOCK-END
