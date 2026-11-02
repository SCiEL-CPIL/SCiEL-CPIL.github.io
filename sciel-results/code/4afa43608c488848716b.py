import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay

N = 26

def _softplus(z, eps=1e-4):
    return np.logaddexp(0, z / eps) * eps

def _objective(params, penalty_weight, smooth_eps=1e-4, active_pairs=None):
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary violations
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
    
    # Overlap violations
    if active_pairs is None:
        idx_i, idx_j = np.triu_indices(N, k=1)
    else:
        idx_i, idx_j = active_pairs
        
    if len(idx_i) > 0:
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx*dx + dy*dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        overlap = r[idx_i] + r[idx_j] - dist
        penalty += np.sum(_softplus(overlap, smooth_eps)**2)
        
    return -np.sum(r) + penalty_weight * penalty

def _get_active_pairs(params, tol=0.05):
    x = params[0::3]
    y = params[1::3]
    centers = np.column_stack([x, y])
    
    # Use Delaunay to find neighbors
    try:
        tri = Delaunay(centers)
        pairs = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    pairs.add((simplex[i], simplex[j]))
        # Also add boundary-adjacent? Delaunay might miss distant ones that could overlap if radii are large.
        # But we are using a penalty method. If a pair is not in Delaunay, they are likely far apart.
        # However, if radii grow, they might overlap.
        # A safer approach: use cKDTree to find pairs within (r_i + r_j + tol).
        
        # Let's stick to cKDTree for robustness in dynamic radii scenarios.
        # Delaunay is static geometry based on centers.
        
    except Exception:
        pass
        
    # cKDTree approach
    try:
        tree = cKDTree(centers)
        # Query for neighbors within max_radius * 2 + tol
        # We don't know max_radius exactly, but we can estimate or just use a generous radius.
        # Actually, let's just compute all pairs if N is small? N=26, 325 pairs.
        # The inspiration said Delaunay cuts O(N^2) to O(N).
        # Let's try to implement a hybrid: use Delaunay edges + pairs that are currently overlapping or close.
        
        # For simplicity and speed, let's use the Delaunay edges plus a check for current overlaps.
        # If a pair is not in Delaunay, it's separated by a convex hull edge or interior triangle.
        # If they overlap, they must be close, so likely Delaunay neighbors? Not necessarily if one is inside the hull of others?
        # Actually, if two circles overlap, their centers are close. In a Delaunay triangulation, close points are usually connected.
        
        # Let's just use the Delaunay edges.
        if 'tri' in locals():
            pairs_list = [tuple(sorted(p)) for p in pairs]
            # Convert to indices
            idx_i = [p[0] for p in pairs_list]
            idx_j = [p[1] for p in pairs_list]
            return (np.array(idx_i), np.array(idx_j))
    except Exception:
        pass
        
    # Fallback to all pairs
    return np.triu_indices(N, k=1)

def _optimize_single(xyz0, deadline):
    params = xyz0.flatten().copy()
    best_params = params.copy()
    best_score = np.sum(params[2::3])
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
        
    # Phase 1: Low penalty, focus on layout, full pairs for safety
    remaining = deadline - time.time()
    if remaining > 4:
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(5.0, 1e-3, None),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective(params, 5.0, 1e-3, None):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
    
    # Phase 2: Medium penalty, use active pairs for speed
    remaining = deadline - time.time()
    if remaining > 4:
        active_pairs = _get_active_pairs(params)
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(50.0, 1e-4, active_pairs),
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < _objective(params, 50.0, 1e-4, active_pairs):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
            
    # Phase 3: High penalty, fine tuning, active pairs
    remaining = deadline - time.time()
    if remaining > 2:
        active_pairs = _get_active_pairs(params)
        try:
            res = minimize(_objective, params, method='L-BFGS-B',
                          bounds=bounds, args=(500.0, 1e-5, active_pairs),
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < _objective(params, 500.0, 1e-5, active_pairs):
                params = res.x
                best_params = params.copy()
                best_score = np.sum(params[2::3])
        except Exception:
            pass
            
    return best_params, best_score

def _validate(xyz, tol=1e-7):
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
    xyz = xyz.copy()
    for i in range(N):
        r = max(0, xyz[i,2])
        xyz[i,0] = np.clip(xyz[i,0], r, 1-r) if r < 0.5 else 0.5
        xyz[i,1] = np.clip(xyz[i,1], r, 1-r) if r < 0.5 else 0.5
        xyz[i,2] = r
    return xyz

def _initial_hex():
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

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    inits = [
        _initial_hex(),
        _initial_grid(),
        _initial_corner_weighted(1),
        _initial_corner_weighted(2),
        _initial_ring(1),
        _initial_ring(2),
    ]
    
    best_params = None
    best_score = -np.inf
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Allocate time for each restart
        # Simple allocation: equal time minus overhead
        time_left = deadline - time.time()
        if time_left < 5:
            break
            
        # Run optimizer
        params, score = _optimize_single(init, deadline)
        
        if score > best_score:
            best_score = score
            best_params = params
            
    if best_params is None:
        # Fallback
        best_params = _initial_grid().flatten()
        
    # Final validation and fixing
    xyz = best_params.reshape(N, 3)
    
    # Fix boundaries
    xyz = _fix_boundaries(xyz)
    
    # Check validity
    if not _validate(xyz):
        # Try to fix overlaps by shrinking radii
        for _ in range(10):
            valid = True
            x, y, r = xyz[:,0], xyz[:,1], xyz[:,2]
            # Check boundary
            for i in range(N):
                min_dist = min(x[i], 1-x[i], y[i], 1-y[i])
                if r[i] > min_dist:
                    r[i] = min_dist * 0.99
                    valid = False
            # Check overlap
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        total_r = r[i] + r[j]
                        if total_r > 0:
                            scale = (dist * 0.99) / total_r
                            r[i] *= scale
                            r[j] *= scale
                        else:
                            r[i] = 0
                            r[j] = 0
                        valid = False
            xyz[:,2] = r
            if valid:
                break
            if not _validate(xyz):
                # If still invalid, try a more aggressive shrink
                max_violation = 0
                for i in range(N):
                    min_dist = min(x[i], 1-x[i], y[i], 1-y[i])
                    if r[i] > min_dist:
                        max_violation = max(max_violation, r[i] - min_dist)
                for i in range(N):
                    r[i] -= max_violation * 0.1
                    r[i] = max(r[i], 0.01)
                xyz[:,2] = r
                
    return xyz
