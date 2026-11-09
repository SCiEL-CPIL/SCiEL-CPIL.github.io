import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
H = np.sqrt(3) / 2

# Precompute indices for vectorized area calculation
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)

idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def min_triangle_area_vec(points):
    """
    points: (N, 2) array
    Returns the minimum area of any triangle formed by 3 points.
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def min_pairwise_distance(points):
    """
    Returns the minimum pairwise distance between any two points.
    """
    n = len(points)
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(points[i] - points[j])
            min_dist = min(min_dist, d)
    return min_dist

def bary_to_cartesian(bary):
    """
    Convert barycentric coordinates to Cartesian in the unit equilateral triangle.
    bary: (N, 3) array where each row sums to 1 and all components >= 0
    """
    # Vertices of equilateral triangle
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, H])
    
    # Cartesian = alpha*A + beta*B + gamma*C
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * H
    
    return np.column_stack((x, y))

def params_to_bary(params):
    """
    Convert flat params to barycentric coordinates.
    params: flat array of 2*N values (u_i, v_i) in [0,1]
    Returns (N, 3) barycentric coordinates.
    """
    params = np.array(params).reshape(N, 2)
    u = params[:, 0]
    v = params[:, 1]
    
    # Ensure u+v <= 1 by normalization
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u = u.copy()
        v = v.copy()
        u[mask] = u[mask] / s[mask]
        v[mask] = v[mask] / s[mask]
    
    # Barycentric: alpha = 1-u-v, beta = u, gamma = v
    alpha = 1.0 - u - v
    beta = u
    gamma = v
    
    return np.column_stack((alpha, beta, gamma))

def bary_to_points(params):
    """
    Convert flat params to Cartesian points in the triangle.
    """
    bary = params_to_bary(params)
    return bary_to_cartesian(bary)

def objective_min_area(params):
    """
    Objective to minimize: negative of min triangle area.
    """
    points = bary_to_points(params)
    return -min_triangle_area_vec(points)

def objective_min_dist(params):
    """
    Objective to minimize: negative of min pairwise distance (smooth proxy).
    """
    points = bary_to_points(params)
    return -min_pairwise_distance(points)

def lattice_search(rng, deadline):
    """
    1D lattice search over barycentric coordinates.
    For each point, we search over a 1D parameter that controls position
    along a radial line from centroid, with angular offset.
    """
    best_params = None
    best_score = -np.inf
    
    # Try different lattice structures
    # Strategy: place points on concentric "rings" with angular offsets
    
    # Ring 0: center point
    # Ring 1: 6 points around center
    # Ring 2: 4 more points (or adjust)
    
    # Actually, let's try a simpler approach:
    # Use a grid in (u,v) space where u+v <= 1
    
    grid_sizes = [5, 6, 7, 8, 9, 10, 12, 14, 16, 20]
    
    for g in grid_sizes:
        if time.time() > deadline - 5.0:
            break
            
        # Generate grid points in (u,v) space
        # u from 0 to 1, v from 0 to 1-u
        # We'll sample g points along each dimension and filter for u+v <= 1
        
        u_vals = np.linspace(0, 1, g)
        v_vals = np.linspace(0, 1, g)
        
        # Generate all (u,v) pairs where u+v <= 1
        candidates = []
        for u in u_vals:
            for v in v_vals:
                if u + v <= 1.0:
                    candidates.append((u, v))
        
        candidates = np.array(candidates)
        
        if len(candidates) < N:
            continue
        
        # We need to select N points from candidates to maximize min pairwise distance
        # This is a combinatorial problem. Use a greedy approach:
        # Start with the point farthest from centroid, then iteratively add
        # the point that maximizes the minimum distance to already-selected points.
        
        centroid = np.array([1.0/3, 1.0/3])
        
        # Sort by distance from centroid (farthest first for diversity)
        dists_to_centroid = np.linalg.norm(candidates - centroid, axis=1)
        
        # Greedy selection
        selected = [np.argmax(dists_to_centroid)]
        
        while len(selected) < N and len(selected) < len(candidates):
            # For each candidate not yet selected, compute min distance to selected
            min_dists = np.full(len(candidates), np.inf)
            for idx in selected:
                d = np.linalg.norm(candidates - candidates[idx], axis=1)
                min_dists = np.minimum(min_dists, d)
            
            # Exclude already selected
            for idx in selected:
                min_dists[idx] = 0
            
            # Pick the one with max min_dist
            next_idx = np.argmax(min_dists)
            selected.append(next_idx)
        
        if len(selected) < N:
            continue
        
        # Build params from selected candidates
        params = np.zeros(2*N)
        for i, idx in enumerate(selected[:N]):
            params[2*i] = candidates[idx, 0]
            params[2*i+1] = candidates[idx, 1]
        
        score = -objective_min_area(params)
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
    
    return best_params, best_score

def local_refine(params, deadline):
    """
    Local refinement using Nelder-Mead on the min-area objective.
    """
    if time.time() > deadline - 10.0:
        return params, -objective_min_area(params)
    
    best_params = params.copy()
    best_score = -objective_min_area(params)
    
    # Multiple restarts with small perturbations
    rng = np.random.default_rng(123)
    
    n_restarts = 20
    for _ in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Perturb the current best
        noise = rng.normal(0, 0.02, size=(2*N,))
        start_params = np.clip(best_params + noise, 0.0, 1.0)
        
        try:
            result = minimize(
                objective_min_area,
                start_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            score = -result.fun
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
        except Exception:
            pass
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    rng = np.random.default_rng(42)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Lattice search for structured initialization
    if time.time() < deadline - 60.0:
        lattice_params, lattice_score = lattice_search(rng, deadline)
        if lattice_params is not None:
            best_params = lattice_params
            best_score = lattice_score
    
    # Phase 2: Differential Evolution for global search
    if time.time() < deadline - 30.0:
        bounds = [(0.0, 1.0)] * (2 * N)
        
        try:
            # Use the lattice result as initial population if available
            if best_params is not None:
                # Create initial population around the lattice solution
                init_pop = []
                # Include the lattice solution
                init_pop.append(best_params.copy())
                # Add random points
                for _ in range(14):
                    init_pop.append(rng.random(2*N))
                init_pop = np.array(init_pop)
            else:
                init_pop = rng.random((15, 2*N))
            
            result = differential_evolution(
                objective_min_area, 
                bounds, 
                maxiter=80, 
                popsize=15, 
                seed=42, 
                tol=1e-6,
                polish=False,
                init=init_pop
            )
            
            de_score = -result.fun
            if de_score > best_score:
                best_score = de_score
                best_params = result.x.copy()
                
        except Exception as e:
            print(f"DE failed: {e}")
    
    # Phase 3: Local refinement with Nelder-Mead
    if best_params is not None and time.time() < deadline - 10.0:
        refined_params, refined_score = local_refine(best_params, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_params = refined_params
    
    # Fallback if nothing worked
    if best_params is None:
        best_params = rng.random(2*N)
    
    # Convert to points
    points = bary_to_points(best_params)
    
    return points
