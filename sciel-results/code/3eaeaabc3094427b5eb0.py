"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(points):
    """Compute the packing radius for given centers."""
    n = len(points)
    if n == 0:
        return 0.0
    # Distance to boundary
    radii = 1.0 - np.linalg.norm(points, axis=1)
    r = np.min(radii)
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    if len(pair_dists) > 0:
        r = min(r, np.min(pair_dists))
    return r

def _hexagonal_init(n):
    """
    Initialize with a hexagonal lattice structure.
    This is a strong starting point for circle packing.
    """
    # Generate hexagonal grid points
    # We need to select 'n' points from an infinite hex grid centered at origin
    # Hex grid coordinates (q, r) -> Cartesian
    # x = size * 3/2 * q
    # y = size * sqrt(3) * (r + q/2)
    
    # Let's generate a large enough grid and pick the n closest to origin
    # Or generate rings of hexagons.
    # Number of points in hex ring k is 6*k.
    # Total points up to ring k: 1 + 6 * (1 + ... + k) = 1 + 3k(k+1)
    # For N=50:
    # k=0: 1
    # k=1: 7 (total 8)
    # k=2: 13 (total 21)
    # k=3: 19 (total 40)
    # k=4: 25 (total 65) -> We need 50, so we take all up to k=3 (40 points) 
    # and 10 points from ring 4.
    
    points = []
    # Center
    points.append([0.0, 0.0])
    
    # Generate hex rings
    # We need to determine the scale 's' such that the points fit nicely.
    # The final optimization will adjust positions, so initial scale doesn't need to be perfect,
    # but it should be reasonable. Let's assume a target radius r ~ 0.12.
    # Distance between centers ~ 2r ~ 0.24.
    # Hex grid spacing 'a' = 2r.
    # Let's use a=1 for generation, then scale at the end.
    
    a = 1.0
    sqrt3 = np.sqrt(3)
    
    def hex_to_cart(q, r):
        x = a * 1.5 * q
        y = a * sqrt3 * (r + q / 2.0)
        return np.array([x, y])
    
    # Generate rings
    # Ring k consists of 6k points.
    # Directions: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) in hex coords
    # Actually, easier to generate all (q,r) such that max(|q|,|r|,|q+r|) == k
    
    max_ring = 5 # Enough to cover 50 points
    candidates = []
    for q in range(-max_ring, max_ring + 1):
        for r in range(-max_ring, max_ring + 1):
            # Hex distance
            dist_hex = max(abs(q), abs(r), abs(q+r))
            # Actually, standard hex distance is (|q| + |r| + |q+r|) / 2
            # Let's use the max norm for simplicity of ring definition, or just generate all and sort by distance
            candidates.append((q, r))
            
    # Convert to Cartesian and sort by distance to origin
    cart_points = []
    for q, r in candidates:
        # Avoid duplicates? The loop generates unique (q,r).
        p = hex_to_cart(q, r)
        d = np.linalg.norm(p)
        cart_points.append((d, p))
        
    # Sort by distance
    cart_points.sort(key=lambda x: x[0])
    
    # Take the first N points
    selected_points = [p for d, p in cart_points[:N]]
    
    points = np.array(selected_points)
    
    # Scale the points so they fit within the unit circle roughly
    # The max radius of the hex cluster
    max_r = np.max(np.linalg.norm(points, axis=1))
    if max_r > 0:
        # We want the outer points to be near the boundary, but not on it.
        # Let's scale so that max_r * (1 + 2*0.12) ~ 1? 
        # Actually, just scale to max_r = 0.8 to leave room for optimization.
        scale = 0.8 / max_r
        points = points * scale
        
    return points

def _smooth_min(dists, b=100.0):
    """Compute a smooth approximation of the minimum of a set of distances."""
    # min(x) ≈ -1/b * log(sum(exp(-b*x)))
    # To avoid overflow, subtract max
    if len(dists) == 0:
        return 0.0
    neg_dists = -dists
    max_neg = np.max(neg_dists)
    # If max_neg is very large positive (dists very negative), exp will underflow/overflow
    # But dists are distances, so they are positive. -dists are negative.
    # max_neg is the least negative number (closest to 0).
    # exp(neg_dists - max_neg) is exp(negative - less_negative) = exp(more_negative) < 1. Safe.
    log_sum = max_neg + np.log(np.sum(np.exp(neg_dists - max_neg)))
    return -log_sum / b

def _objective(params, b=100.0):
    """
    Objective function for L-BFGS-B.
    Minimizes the negative smooth minimum of all relevant distances.
    params: flattened array of 2*N coordinates.
    """
    pts = params.reshape(-1, 2)
    n = len(pts)
    
    # Boundary distances
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, 1)
    pair_dists = dists[iu] / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([boundary, pair_dists])
    
    # Smooth min
    smooth_min_val = _smooth_min(all_dists, b)
    
    # We want to maximize the min, so minimize -min
    return -smooth_min_val

def _anneal(points, time_limit=20.0):
    """
    Simulated Annealing to improve the packing.
    Uses a potential energy that penalizes overlaps and boundary violations.
    """
    n = len(points)
    pts = points.copy()
    
    # Initial temperature
    T = 0.1
    T_min = 0.001
    cooling_rate = 0.999
    
    # Precompute indices for efficiency
    iu = np.triu_indices(n, 1)
    
    start_time = time.time()
    steps = 0
    best_score = _score(pts)
    best_pts = pts.copy()
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Generate a random perturbation
        # Move one point or a few points
        idx = np.random.randint(n)
        # Random step size
        step_size = 0.05 * T + 0.01
        delta = np.random.randn(2) * step_size
        
        new_pts = pts.copy()
        new_pts[idx] += delta
        
        # Check if new points are valid (finite)
        if not np.all(np.isfinite(new_pts)):
            T *= cooling_rate
            continue
            
        # Calculate energy change
        # Energy = Sum of repulsive forces
        # We use a soft potential: E = sum exp(-alpha * dist)
        # Minimizing E maximizes distances.
        # Let's use alpha = 1/T to control sharpness
        
        alpha = 1.0 / T
        
        def calc_energy(p):
            # Boundary term
            b_dist = 1.0 - np.linalg.norm(p, axis=1)
            # Clip to avoid log/exp issues if too far out
            b_dist = np.clip(b_dist, 1e-6, None)
            e_b = np.sum(np.exp(-alpha * b_dist))
            
            # Pairwise term
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            pd = dists[iu]
            # Avoid division by zero or exp overflow
            pd = np.clip(pd, 1e-6, None)
            e_p = np.sum(np.exp(-alpha * pd))
            
            return e_b + e_p
        
        E_old = calc_energy(pts)
        E_new = calc_energy(new_pts)
        
        dE = E_new - E_old
        
        # Metropolis criterion
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            pts = new_pts
            current_score = _score(pts)
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        
        T *= cooling_rate
        steps += 1
        
        # If time is running out, break
        if time.time() - start_time > time_limit - 2.0:
            break
            
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    # 1. Initialize with hexagonal structure
    points = _hexagonal_init(N)
    best_points = points.copy()
    best_score = _score(best_points)
    
    # 2. Simulated Annealing
    # Run SA for a portion of the time
    sa_time = 30.0
    if time.time() < deadline - sa_time - 10.0:
        points = _anneal(points, time_limit=sa_time)
        if _score(points) > best_score:
            best_score = _score(points)
            best_points = points.copy()
    
    # 3. Local Optimization using L-BFGS-B
    # Multiple rounds with increasing b (sharpness)
    b_values = [50.0, 100.0, 200.0, 500.0]
    
    x0 = best_points.flatten()
    
    for b in b_values:
        if time.time() > deadline - 15.0:
            break
            
        def obj_func(params, b_val=b):
            return _objective(params, b_val)
        
        try:
            res = minimize(obj_func, x0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if time.time() > deadline - 10.0:
                break
                
            cand = res.x.reshape(-1, 2)
            if np.all(np.isfinite(cand)):
                cand_score = _score(cand)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand.copy()
                    x0 = cand.flatten()
        except Exception as e:
            pass
    
    # 4. Multi-start local optimization from best point
    # Try a few perturbations to escape local minima if stuck
    n_restarts = 0
    max_restarts = 5
    
    while n_restarts < max_restarts and time.time() < deadline - 10.0:
        # Perturb the best point
        perturbation_scale = 0.02
        x0 = best_points.flatten() + np.random.randn(2 * N) * perturbation_scale
        
        n_restarts += 1
        
        b = 200.0
        def obj_func(params, b_val=b):
            return _objective(params, b_val)
        
        try:
            res = minimize(obj_func, x0, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if time.time() > deadline - 5.0:
                break
                
            cand = res.x.reshape(-1, 2)
            if np.all(np.isfinite(cand)):
                cand_score = _score(cand)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand.copy()
        except Exception:
            pass
            
    # 5. Final high-precision polish
    if time.time() < deadline - 5.0:
        x0 = best_points.flatten()
        b = 1000.0
        
        def obj_func(params, b_val=b):
            return _objective(params, b_val)
        
        try:
            res = minimize(obj_func, x0, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
            
            cand = res.x.reshape(-1, 2)
            if np.all(np.isfinite(cand)):
                cand_score = _score(cand)
                if cand_score > best_score:
                    best_score = cand_score
                    best_points = cand.copy()
        except Exception:
            pass
            
    return best_points
# EVOLVE-BLOCK-END
