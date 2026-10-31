import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11

def _min_area_bary(points_bary):
    """Compute minimum triangle area from barycentric coords (N,2) where w=1-u-v."""
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    n = len(u)
    
    # Vectorized computation of all triangle areas
    # For each triple (i,j,k), area = |det| / 2 but we just need relative ordering
    # det = u_i*(v_j*w_k - w_j*v_k) - v_i*(u_j*w_k - w_j*u_k) + w_i*(u_j*v_k - v_j*u_k)
    
    min_a = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                det = (u[i]*(v[j]*w[k] - w[j]*v[k]) 
                      - v[i]*(u[j]*w[k] - w[j]*u[k]) 
                      + w[i]*(u[j]*v[k] - v[j]*u[k]))
                a = abs(det)
                if a < min_a:
                    min_a = a
    return min_a

def _min_area_bary_fast(points_bary):
    """Faster vectorized version for minimum triangle area."""
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    n = len(u)
    
    # Compute all pairwise cross products in 3D (u,v,w) space
    # The area of triangle (i,j,k) is 0.5 * |cross(p_j - p_i, p_k - p_i)|
    # In barycentric coords, the determinant gives twice the signed area relative to the reference triangle
    
    # More efficient: compute all C(n,3) determinants
    min_a = float('inf')
    for i in range(n):
        ui, vi, wi = u[i], v[i], w[i]
        for j in range(i+1, n):
            uj, vj, wj = u[j], v[j], w[j]
            for k in range(j+1, n):
                uk, vk, wk = u[k], v[k], w[k]
                det = ui*(vj*wk - wj*vk) - vi*(uj*wk - wj*uk) + wi*(uj*vk - vj*uk)
                a = abs(det)
                if a < min_a:
                    min_a = a
    return min_a

def _soft_min_area(points_bary, beta=100.0):
    """Soft minimum of triangle areas for smooth optimization."""
    u = points_bary[:, 0]
    v = points_bary[:, 1]
    w = 1.0 - u - v
    n = len(u)
    
    areas = []
    for i in range(n):
        ui, vi, wi = u[i], v[i], w[i]
        for j in range(i+1, n):
            uj, vj, wj = u[j], v[j], w[j]
            for k in range(j+1, n):
                uk, vk, wk = u[k], v[k], w[k]
                det = ui*(vj*wk - wj*vk) - vi*(uj*wk - wj*uk) + wi*(uj*vk - vj*uk)
                areas.append(abs(det))
    
    areas = np.array(areas)
    # Soft minimum: -log(sum(exp(-beta*a))) / beta
    # For stability, subtract max
    a_max = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - a_max))))
    soft_min = a_max - log_sum / beta
    return soft_min

def _clamp_to_simplex(uv):
    """Clamp (u,v) coordinates to valid barycentric region."""
    u = uv[:, 0].copy()
    v = uv[:, 1].copy()
    
    # Ensure u >= 0, v >= 0
    u = np.maximum(u, 0)
    v = np.maximum(v, 0)
    
    # Ensure u + v <= 1
    sum_uv = u + v
    mask = sum_uv > 1
    if np.any(mask):
        factor = 1.0 / sum_uv[mask]
        u[mask] *= factor
        v[mask] *= factor
    
    return np.column_stack([u, v])

def _objective(params):
    """Objective for optimization: minimize negative min area."""
    uv = params.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1 - u - v
    
    # Penalty for invalid points
    penalty = 0
    if np.any(w < -1e-10):
        penalty += 1000 * np.sum(w[w < 0]**2)
    if np.any(u < -1e-10):
        penalty += 1000 * np.sum(u[u < 0]**2)
    if np.any(v < -1e-10):
        penalty += 1000 * np.sum(v[v < 0]**2)
    
    min_a = _min_area_bary_fast(uv)
    return -min_a + penalty

def _generate_initializations():
    """Generate diverse initial configurations."""
    inits = []
    rng = np.random.RandomState(12345)
    
    # 1. Symmetric grid-based initialization
    # Place points on a triangular lattice pattern
    grid_points = []
    # 4x4 grid in barycentric coords, select 11
    for i in range(5):
        for j in range(5 - i):
            u = i / 4.0
            v = j / 4.0
            w = 1 - u - v
            if w >= -1e-10:
                grid_points.append((u, v))
    
    # Select 11 points from the grid, trying to maximize spread
    # Use a greedy approach: start with corners, then add farthest points
    selected = []
    remaining = list(range(len(grid_points)))
    
    # Start with three corners
    selected.append(0)  # (0,0)
    selected.append(4)  # (1,0) - index depends on ordering
    selected.append(8)  # (0,1) - need to find correct indices
    
    # Actually, let's just use a fixed good initial configuration
    # Based on known good solutions for small n
    base_config = np.array([
        [0.0, 0.0],    # corner A
        [1.0, 0.0],    # corner B
        [0.0, 1.0],    # corner C
        [0.5, 0.0],    # midpoint AB
        [0.0, 0.5],    # midpoint AC
        [0.5, 0.5],    # midpoint BC
        [1/3, 1/3],    # centroid
        [2/3, 0.0],    # 2/3 along AB
        [0.0, 2/3],    # 2/3 along AC
        [2/3, 1/3],    # interior
        [1/3, 2/3],    # interior
    ])
    inits.append(base_config)
    
    # 2. Random Dirichlet samples
    for _ in range(30):
        bary = rng.dirichlet(np.ones(3), size=N)
        uv = bary[:, :2]
        inits.append(uv)
    
    # 3. Perturbed base configurations
    for _ in range(20):
        noise = rng.uniform(-0.1, 0.1, size=base_config.shape)
        perturbed = base_config + noise
        perturbed = _clamp_to_simplex(perturbed)
        inits.append(perturbed)
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    best_score = -float('inf')
    best_points = None
    
    # Phase 1: Differential Evolution for global exploration
    bounds = [(0, 1)] * (2 * N)
    
    # Use differential evolution with multiple restarts
    de_kwargs = {
        'bounds': bounds,
        'objective': _objective,
        'maxiter': 300,
        'popsize': 30,
        'tol': 1e-8,
        'mut': (0.5, 1.0),
        'recon': 'rand1bin',
        'seed': 42,
        'init': 'latinhypercube',
        'polish': False,
        'updating': 'immediate',
        'workers': -1,
    }
    
    try:
        # Time-limited DE
        elapsed = time.time() - start_time
        if elapsed < deadline - 30:
            de_result = differential_evolution(**de_kwargs)
            uv_de = de_result.x.reshape(N, 2)
            uv_de = _clamp_to_simplex(uv_de)
            score_de = _min_area_bary_fast(uv_de)
            
            if score_de > best_score:
                best_score = score_de
                best_points = uv_de.copy()
    except Exception:
        pass
    
    # Phase 2: Local refinement from best DE result and initializations
    inits = _generate_initializations()
    
    if best_points is not None:
        inits.append(best_points.copy())
    
    # Add some more diverse starts
    rng = np.random.RandomState(99)
    for _ in range(10):
        bary = rng.dirichlet(np.ones(3), size=N)
        inits.append(bary[:, :2])
    
    # Local optimization with Nelder-Mead and L-BFGS-B
    for init in inits:
        if time.time() > deadline - 10:
            break
        
        init = _clamp_to_simplex(init)
        
        # Try Nelder-Mead
        try:
            nm_result = minimize(
                _objective,
                init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            uv_nm = nm_result.x.reshape(N, 2)
            uv_nm = _clamp_to_simplex(uv_nm)
            score_nm = _min_area_bary_fast(uv_nm)
            
            if score_nm > best_score:
                best_score = score_nm
                best_points = uv_nm.copy()
        except Exception:
            pass
        
        # Try L-BFGS-B from the Nelder-Mead result if available
        if time.time() > deadline - 10:
            break
        
        try:
            lbfgs_result = minimize(
                _objective,
                uv_nm.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            uv_lbfgs = lbfgs_result.x.reshape(N, 2)
            uv_lbfgs = _clamp_to_simplex(uv_lbfgs)
            score_lbfgs = _min_area_bary_fast(uv_lbfgs)
            
            if score_lbfgs > best_score:
                best_score = score_lbfgs
                best_points = uv_lbfgs.copy()
        except Exception:
            pass
    
    # Phase 3: Coordinate descent refinement
    if best_points is not None and time.time() < deadline - 5:
        current = best_points.copy()
        improved = True
        max_iter = 50
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            if not improved:
                break
            
            improved = False
            for i in range(N):
                for coord in range(2):
                    # Optimize one coordinate at a time
                    best_val = current[i, coord]
                    best_obj = _objective(current.flatten())
                    
                    # Try small perturbations
                    for delta in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                        trial = current.copy()
                        trial[i, coord] += delta
                        trial = _clamp_to_simplex(trial)
                        obj = _objective(trial.flatten())
                        if obj < best_obj:
                            best_obj = obj
                            best_val = trial[i, coord]
                            improved = True
                    
                    current[i, coord] = best_val
            
            # Full re-clamp
            current = _clamp_to_simplex(current)
            score = _min_area_bary_fast(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
    
    if best_points is None:
        # Fallback
        best_points = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
            [1/3, 1/3], [2/3, 0.0], [0.0, 2/3],
            [2/3, 1/3], [1/3, 2/3]
        ])
    
    # Convert barycentric to Cartesian
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    pts = []
    for i in range(N):
        u, v = best_points[i]
        w = 1 - u - v
        p = u * A + v * B + w * C
        pts.append(p)
    
    return np.array(pts)
