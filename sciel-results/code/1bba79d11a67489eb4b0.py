import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return -np.inf
    
    # Distance from origin
    dist_origin = np.linalg.norm(P, axis=1)
    r_bound = np.min(1.0 - dist_origin)
    
    # Pairwise distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return min(r_pair, r_bound)

def _smooth_objective(P_flat, r_target):
    """
    Smooth objective function for L-BFGS-B.
    Minimizes the sum of squared overlaps.
    Overlap = max(2*r - dist, 0) for pairs.
    Overlap = max(dist_origin + r - 1, 0) for boundary.
    """
    P = P_flat.reshape(N, 2)
    
    # Pairwise
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Soft max approximation or just hard max (L-BFGS-B handles non-diff points okay-ish, 
    # but smooth is better. Here we use hard max but the gradient will be 0 where no overlap, 
    # which is fine for local refinement if we start close).
    # Actually, for L-BFGS-B, we provide the gradient.
    overlaps_cc = np.maximum(2.0 * r_target - dist, 0.0)
    
    # Boundary
    dist_origin = np.linalg.norm(P, axis=1)
    overlaps_cb = np.maximum(dist_origin + r_target - 1.0, 0.0)
    
    return 0.5 * (np.sum(overlaps_cc**2) + np.sum(overlaps_cb**2))

def _smooth_gradient(P_flat, r_target):
    """
    Gradient of the smooth objective.
    """
    P = P_flat.reshape(N, 2)
    grad = np.zeros_like(P)
    
    # Pairwise
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    V_cc = np.maximum(2.0 * r_target - dist, 0.0)
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 1e-12)
    
    # grad_i = sum_j (2*r - d_ij) * (p_i - p_j) / d_ij
    # Note: The term is - (2*r - d) * (p_i - p_j)/d for the objective 0.5 * sum (2r-d)^2
    # d/dp_i 0.5 * (2r - ||p_i - p_j||)^2 = -(2r - d) * (p_i - p_j) / d
    W = V_cc * inv_dist
    # sum_j W_ij * (p_i - p_j) = p_i * sum_j W_ij - sum_j W_ij * p_j
    sum_W = np.sum(W, axis=1)
    term1 = P * sum_W[:, np.newaxis]
    term2 = (P.T @ W).T
    grad -= (term1 - term2)
    
    # Boundary
    dist_origin = np.linalg.norm(P, axis=1)
    V_cb = np.maximum(r_target + dist_origin - 1.0, 0.0)
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 1e-12)
    
    # grad_i = (r + ||p_i|| - 1) * p_i / ||p_i||
    grad += (V_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad.flatten()

def _local_refine(P_init, r_target, deadline):
    """
    Use L-BFGS-B to minimize the smooth objective for a fixed r_target.
    Returns the optimized P and the energy.
    """
    def objective(P_flat):
        return _smooth_objective(P_flat, r_target)
    
    def grad(P_flat):
        return _smooth_gradient(P_flat, r_target)
        
    try:
        res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        P_opt = res.x.reshape(N, 2)
        return P_opt
    except Exception:
        return P_init

def _simulate_annealing(P_init, start_temp, end_temp, steps, deadline):
    """
    Simulated Annealing to maximize the minimum distance.
    We minimize the "energy" which is related to the negative of the score or overlaps.
    Here we define energy as sum of overlaps for a fixed radius? 
    No, SA usually works better if we directly try to maximize the minimum distance.
    However, maximizing min(dist) is non-differentiable and hard for SA moves.
    
    Alternative: SA minimizes a penalty function for a FIXED radius r.
    If we can fit all circles with radius r, energy is 0.
    We can binary search on r.
    
    Strategy:
    1. Start with a low r.
    2. Run SA to minimize overlaps for that r.
    3. If overlaps are 0, increase r.
    4. If overlaps > 0, decrease r or try harder.
    
    This is complex. Simpler SA:
    Objective: Maximize r_current.
    Move: Randomly perturb a point.
    Acceptance: If new r > old r, accept. Else accept with prob exp((new_r - old_r)/T).
    Problem: r changes very little, making the probability calculation noisy.
    
    Better SA Objective: Minimize Sum of Squared Overlaps for a GIVEN r.
    We will use a "Fixed Radius" approach.
    """
    P_curr = P_init.copy()
    curr_score = _compute_score(P_curr)
    
    # We will try to increase the radius.
    # Let's use a "Radius Annealing" approach.
    # Start with r = curr_score * 0.9
    # Try to fit into that r using SA.
    
    best_P = P_curr.copy()
    best_r = curr_score
    
    # Define a helper to run SA for a specific target radius
    def sa_for_radius(P_start, r_target, steps, deadline):
        P = P_start.copy()
        T = start_temp
        cooling = (end_temp / start_temp) ** (1.0 / steps)
        
        # Calculate initial energy
        def energy(P):
            return _smooth_objective(P.flatten(), r_target)
            
        E_curr = energy(P)
        P_best = P.copy()
        E_best = E_curr
        
        for i in range(steps):
            if time.time() > deadline - 5.0:
                break
            
            # Random move: pick a random point and perturb it
            idx = np.random.randint(N)
            # Step size proportional to temperature
            step_size = T * 0.1
            dP = np.random.normal(0, step_size, 2)
            P_new = P.copy()
            P_new[idx] += dP
            
            E_new = energy(P_new)
            
            if E_new < E_curr or np.random.random() < np.exp((E_curr - E_new) / T):
                P = P_new
                E_curr = E_new
                if E_new < E_best:
                    E_best = E_new
                    P_best = P.copy()
            
            T *= cooling
            
        return P_best, E_best

    # Phase 1: Find a good radius via bisection + SA
    # Start with a guess
    r_low = curr_score * 0.9
    r_high = curr_score * 1.1
    
    # First, try to stabilize at current r
    P_stable, _ = sa_for_radius(P_curr, curr_score, 500, deadline)
    best_P = P_stable
    best_r = _compute_score(P_stable)
    
    # Now try to increase r
    # We do a few iterations of increasing r and refining
    for iter_num in range(10):
        if time.time() > deadline - 10.0:
            break
            
        # Try a slightly larger radius
        r_try = best_r * 1.01 + 0.0001
        
        # Run SA to see if we can fit r_try
        P_try, E_try = sa_for_radius(best_P, r_try, 1000, deadline)
        
        # Check if it actually fits (E_try should be close to 0, but let's check real score)
        score_try = _compute_score(P_try)
        
        if score_try > best_r:
            best_r = score_try
            best_P = P_try
        else:
            # If we can't increase, maybe we can refine the current best more
            # Or break if we're stuck
            if iter_num > 3:
                break

    return best_P, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    np.random.seed(42)
    
    # Initializations
    # 1. Hexagonal Lattice
    def get_hex_lattice():
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            margin = spacing / 2.0 
            points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
            if len(points) >= N:
                indices = np.argsort(radius, kind="stable")
                return points[indices[:N]]
            spacing *= 0.98
            
    P_init_hex = get_hex_lattice()
    
    # 2. Random Jittered Grid
    def get_random_grid():
        # Create a grid and jitter
        side = int(np.ceil(np.sqrt(N)))
        x = np.linspace(-0.8, 0.8, side)
        y = np.linspace(-0.8, 0.8, side)
        X, Y = np.meshgrid(x, y)
        P = np.column_stack((X.ravel(), Y.ravel()))[:N]
        P += np.random.normal(0, 0.05, P.shape)
        return P
        
    P_init_rand = get_random_grid()
    
    # 3. Ring Initialization
    def get_ring_init():
        P = np.zeros((N, 2))
        # 1 center, 6 ring 1, 12 ring 2, 18 ring 3, 13 ring 4
        rings = [1, 6, 12, 18, 13]
        radii = [0.0, 0.25, 0.5, 0.75, 0.9]
        count = 0
        for i, n in enumerate(rings):
            if n == 1:
                P[count] = [0,0]
                count += 1
            else:
                angles = np.linspace(0, 2*np.pi, n, endpoint=False)
                P[count:count+n, 0] = radii[i] * np.cos(angles)
                P[count:count+n, 1] = radii[i] * np.sin(angles)
                count += n
        return P
        
    P_init_ring = get_ring_init()
    
    inits = [P_init_hex, P_init_rand, P_init_ring]
    
    best_P = None
    best_score = -np.inf
    
    # Run SA + Refine for each initialization
    for P_init in inits:
        if time.time() > deadline - 30.0:
            break
            
        # 1. Initial Local Refine (L-BFGS-B) to get a clean starting point
        r_guess = _compute_score(P_init)
        P_refined = _local_refine(P_init, r_guess, deadline)
        score_refined = _compute_score(P_refined)
        
        # 2. Simulated Annealing to explore and increase radius
        P_sa, score_sa = _simulate_annealing(P_refined, start_temp=0.5, end_temp=0.01, steps=2000, deadline=deadline)
        
        # 3. Final Local Refine on the SA result
        P_final = _local_refine(P_sa, score_sa, deadline)
        score_final = _compute_score(P_final)
        
        if score_final > best_score:
            best_score = score_final
            best_P = P_final
            
        # Update best_P to the best found so far for next iterations? 
        # No, keep them independent to avoid bias, but we can use the best one for the next SA start if time permits.
        
    # If we have time, do a final high-precision polish on the best solution
    if best_P is not None and time.time() < deadline - 10.0:
        # Try to squeeze a bit more out of the best solution
        # Increase r slightly and refine
        r_try = best_score * 1.005
        P_squeeze = _local_refine(best_P, r_try, deadline)
        score_squeeze = _compute_score(P_squeeze)
        
        if score_squeeze > best_score:
            best_score = score_squeeze
            best_P = P_squeeze

    # Ensure points are inside
    norms = np.linalg.norm(best_P, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_P[mask] *= (1.0 - 1e-9) / norms[mask, np.newaxis]
        
    return best_P
