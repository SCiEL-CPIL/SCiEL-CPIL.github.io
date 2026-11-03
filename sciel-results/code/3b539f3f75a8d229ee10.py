import numpy as np
import time
from scipy.optimize import minimize

N = 11
SQRT3 = np.sqrt(3)

# Precompute indices for all unique triples (i, j, k) with i < j < k
_triples = []
for i in range(N - 2):
    for j in range(i + 1, N - 1):
        for k in range(j + 1, N):
            _triples.append((i, j, k))
_TRIPLES = np.array(_triples, dtype=np.int32)

def _min_area_bary(p):
    """
    p: array of shape (N, 3) with barycentric coordinates.
    Returns the minimum normalized triangle area.
    """
    r1 = p[_TRIPLES[:, 0]]
    r2 = p[_TRIPLES[:, 1]]
    r3 = p[_TRIPLES[:, 2]]
    
    # dets = det([r1, r2, r3]) = r1 dot (r2 cross r3)
    cross_r2_r3 = np.cross(r2, r3)
    dets = np.einsum('ij,ij->i', r1, cross_r2_r3)
    
    areas = 0.5 * np.abs(dets)
    return np.min(areas)

def _bary_to_cart(bary):
    """Convert barycentric (u,v,w) to Cartesian (x,y)."""
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = (SQRT3 / 2) * bary[:, 2]
    return np.column_stack([x, y])

def _project_bary(bary):
    """
    Project barycentric coordinates to the simplex.
    """
    b = bary.copy()
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    b = np.clip(b, 0, 1)
    b_sum = np.sum(b, axis=1, keepdims=True)
    b_sum[b_sum == 0] = 1.0
    b = b / b_sum
    return b

def _get_min_triple_index(p):
    """Returns the index of the triple with minimum area."""
    r1 = p[_TRIPLES[:, 0]]
    r2 = p[_TRIPLES[:, 1]]
    r3 = p[_TRIPLES[:, 2]]
    cross_r2_r3 = np.cross(r2, r3)
    dets = np.einsum('ij,ij->i', r1, cross_r2_r3)
    areas = 0.5 * np.abs(dets)
    return np.argmin(areas)

def _objective_neg_min_area(x_flat):
    """
    Objective for Nelder-Mead: minimize negative of min area.
    x_flat: flat array of barycentric coords.
    """
    bary = x_flat.reshape(N, 3)
    bary = _project_bary(bary)
    val = _min_area_bary(bary)
    return -val

def _generate_structured_inits():
    """
    Generate a diverse set of initial configurations.
    """
    inits = []
    
    # 1. Random uniform
    rng = np.random.default_rng(42)
    for _ in range(5):
        bary = rng.dirichlet([1, 1, 1], size=N)
        inits.append(bary)
        
    # 2. Symmetric-ish: 1 center, 6 on inner hexagon, 4 near corners
    center = np.array([1/3, 1/3, 1/3])
    # Inner hexagon vertices in barycentric
    # Scale factor for inner hexagon
    s = 0.3
    hex_verts = []
    for k in range(6):
        angle = k * np.pi / 3
        # In cartesian, center is (0.5, sqrt(3)/6)
        # Barycentric conversion is tricky, let's just use random perturbations around center
        pass
    
    # Let's use a simpler structured init: Grid-like
    # 3x3 grid + 2 extra
    # Map grid to barycentric
    # x in [0,1], y in [0, sqrt(3)/2]
    # x = v + w/2, y = sqrt(3)/2 * w
    # w = 2*y/sqrt(3), v = x - w/2, u = 1 - v - w
    grid_pts = []
    for i in range(3):
        for j in range(3):
            x = (i + 0.5) / 3.0
            y = (j + 0.5) / 3.0 * (SQRT3 / 2)
            w = 2 * y / SQRT3
            v = x - w / 2
            u = 1 - v - w
            if u >= 0 and v >= 0 and w >= 0:
                grid_pts.append([u, v, w])
    if len(grid_pts) >= N:
        inits.append(np.array(grid_pts[:N]))
    else:
        # Pad with center
        while len(grid_pts) < N:
            grid_pts.append([1/3, 1/3, 1/3])
        inits.append(np.array(grid_pts))
        
    # 3. Boundary heavy
    corners = np.array([
        [0.95, 0.025, 0.025],
        [0.025, 0.95, 0.025],
        [0.025, 0.025, 0.95]
    ])
    rest = rng.dirichlet([1, 1, 1], size=N-3)
    inits.append(np.vstack([corners, rest]))
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_area = 0.0
    best_bary = None
    
    inits = _generate_structured_inits()
    rng = np.random.default_rng(123)
    
    # We will iterate through inits, and for each, do a local search with "kicks"
    # to escape local minima.
    
    for init in inits:
        if time.time() > deadline - 10.0:
            break
            
        current_bary = _project_bary(init.copy())
        current_area = _min_area_bary(current_bary)
        
        if current_area > best_area:
            best_area = current_area
            best_bary = current_bary.copy()
            
        # Local search phase with restarts
        # We use Nelder-Mead for the local refinement as it handles the non-smooth min well.
        # We also use a "kick" strategy: if stuck, move the critical points.
        
        for kick_iter in range(10): # Max kicks per init
            if time.time() > deadline - 5.0:
                break
                
            # Refine with Nelder-Mead
            # Flatten for NM
            x0 = current_bary.flatten()
            
            try:
                res = minimize(
                    _objective_neg_min_area,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                refined_bary = _project_bary(res.x.reshape(N, 3))
                refined_area = _min_area_bary(refined_bary)
                
                if refined_area > current_area:
                    current_bary = refined_bary
                    current_area = refined_area
                    if current_area > best_area:
                        best_area = current_area
                        best_bary = current_bary.copy()
                else:
                    # Stuck? Apply a kick
                    # Identify the min area triple
                    min_idx = _get_min_triple_index(current_bary)
                    i, j, k = _TRIPLES[min_idx]
                    
                    # Randomly perturb these 3 points
                    # Add random noise
                    noise_scale = 0.1
                    current_bary[i] += rng.normal(0, noise_scale, 3)
                    current_bary[j] += rng.normal(0, noise_scale, 3)
                    current_bary[k] += rng.normal(0, noise_scale, 3)
                    
                    current_bary = _project_bary(current_bary)
                    current_area = _min_area_bary(current_bary)
                    
                    # If the kick made it worse, we might want to revert? 
                    # But the point is to explore. 
                    # However, if it's much worse, we might just keep the previous best 
                    # and try a different kick or move to next init.
                    # Let's keep the kicked version for the next NM run to see if it escapes.
                    
            except Exception:
                pass
                
    if best_bary is None:
        rng = np.random.default_rng(0)
        best_bary = rng.dirichlet([1, 1, 1], size=N)
        best_bary = _project_bary(best_bary)
        
    return _bary_to_cart(best_bary)
