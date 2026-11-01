import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import itertools

N = 14
D = 3
DEADLINE = 140.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to min pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _generate_icosahedron_grid():
    """
    Generate high-symmetry initial points based on icosahedral geometry.
    """
    phi = (1 + 5 ** 0.5) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1], [0, phi, -1], [0, -phi, 1], [0, -phi, -1],
        [1, 0, phi], [1, 0, -phi], [-1, 0, phi], [-1, 0, -phi],
        [phi, 1, 0], [phi, -1, 0], [-phi, 1, 0], [-phi, -1, 0]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # We need 14 points. 
    # Strategy: Use 12 icosahedron vertices + 2 points from a dual structure (dodecahedron centers) 
    # or just perturb some vertices.
    # For N=14, a good structure might be an icosahedron with 2 vertices removed and 2 new ones added?
    # Or simply take 14 distinct vertices from a denser set.
    
    # Let's use the 12 vertices of the icosahedron and the centers of 2 opposite faces?
    # Face centers of icosahedron are vertices of a dodecahedron.
    # Dodecahedron vertices:
    dodec_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ], dtype=float)
    dodec_verts = dodec_verts / np.linalg.norm(dodec_verts, axis=1, keepdims=True)
    
    # Combine and normalize
    all_cands = np.vstack([verts, dodec_verts])
    # Remove duplicates (some might be close)
    unique_cands = []
    for p in all_cands:
        is_dup = False
        for u in unique_cands:
            if np.linalg.norm(p - u) < 0.1:
                is_dup = True
                break
        if not is_dup:
            unique_cands.append(p)
            
    unique_cands = np.array(unique_cands)
    
    # Select 14 distinct points from this set that maximize min distance (greedy)
    # This provides a good "structured" start.
    selected = []
    remaining = np.arange(len(unique_cands))
    
    # Start with a vertex
    selected.append(remaining[0])
    remaining = np.delete(remaining, 0)
    
    for _ in range(N - 1):
        sel_pts = unique_cands[selected]
        dists = cdist(unique_cands[remaining], sel_pts)
        min_dists = dists.min(axis=1)
        best_local_idx = np.argmax(min_dists)
        best_global_idx = remaining[best_local_idx]
        selected.append(best_global_idx)
        remaining = np.delete(remaining, best_local_idx)
        
    return unique_cands[selected]

def _soft_log_ratio(G, beta=50.0):
    """
    Smoothed objective function based on Gram matrix G.
    G is (N, N). Distances squared are d_ij^2 = G_ii + G_jj - 2 G_ij.
    We want to minimize (max d / min d)^2.
    Let u = d^2. Minimize (max u / min u).
    Using log-sum-exp approximation for max and min (via -max(-u)).
    """
    diag = np.diag(G)
    # d^2 matrix
    d2 = diag[:, None] + diag[None, :] - 2 * G
    np.fill_diagonal(d2, 0.0)
    
    # Ensure positive distances (numerical stability)
    d2 = np.maximum(d2, 1e-9)
    
    # Soft max of d2
    max_d2 = np.log(np.sum(np.exp(beta * d2))) / beta
    # Soft min of d2: - soft max of -d2
    min_d2 = -np.log(np.sum(np.exp(-beta * d2))) / beta
    
    if min_d2 < 1e-6:
        return 1e6
    return max_d2 / min_d2

def _objective_and_grad(G_flat, beta=50.0):
    """
    Objective function for L-BFGS-B.
    Input: G_flat (N*N)
    Returns: f, grad
    """
    N_local = int(np.sqrt(len(G_flat)))
    G = G_flat.reshape(N_local, N_local)
    
    # Ensure PSD by projecting? 
    # L-BFGS-B doesn't have PSD constraint. 
    # Strategy: Parameterize G = X X^T where X is (N, D).
    # This is the "Cholesky-like" or "Factor" parameterization.
    # Let's switch to optimizing X directly in _optimize_X.
    # This function is a placeholder if we stick to G, but we will use X.
    pass

def _optimize_X(X_init, beta=50.0, time_limit=5.0):
    """
    Optimize the coordinate matrix X (N, D) to minimize the soft ratio.
    """
    start_time = time.time()
    
    def objective(X_flat):
        if time.time() - start_time > time_limit:
            return 1e10, np.zeros_like(X_flat)
        X = X_flat.reshape(N, D)
        # Compute Gram matrix from X
        G = X @ X.T
        # Compute soft ratio
        diag = np.diag(G)
        d2 = diag[:, None] + diag[None, :] - 2 * G
        np.fill_diagonal(d2, 0.0)
        d2 = np.maximum(d2, 1e-9)
        
        max_d2 = np.log(np.sum(np.exp(beta * d2))) / beta
        min_d2 = -np.log(np.sum(np.exp(-beta * d2))) / beta
        
        if min_d2 < 1e-6:
            return 1e6, np.zeros_like(X_flat)
        
        val = max_d2 / min_d2
        
        # Gradient calculation is complex for soft-max/min.
        # We can use numerical gradient or analytic.
        # Analytic:
        # Let M = beta * d2
        # max_d2 = (1/beta) log(sum(exp(M)))
        # d(max_d2)/d(G_ij) = ...
        # This is tedious. Let's use scipy's numerical gradient initially or just finite differences?
        # Finite differences on 42 variables is 42 evaluations per step. 
        # L-BFGS-B might take 100 steps. 4200 evals. Each eval is O(N^2). N=14. Fast.
        
        return val, None # None means use numerical gradient
    
    res = minimize(objective, X_init.flatten(), method='L-BFGS-B', 
                   options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5, 'disp': False})
    
    if res.success:
        return res.x.reshape(N, D)
    else:
        return X_init

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = np.inf
    
    # 1. Generate diverse initializations
    inits = []
    
    # A. Icosahedral structure
    ico_pts = _generate_icosahedron_grid()
    inits.append(ico_pts)
    
    # B. Fibonacci sphere
    i = np.arange(N)
    phi_angle = np.arccos(1 - 2 * (i + 0.5) / N)
    theta_angle = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi_angle) * np.cos(theta_angle)
    y = np.sin(phi_angle) * np.sin(theta_angle)
    z = np.cos(phi_angle)
    fib_pts = np.column_stack([x, y, z])
    inits.append(fib_pts)
    
    # C. Random points on sphere
    for _ in range(5):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
    
    # D. Perturbed icosahedron
    for _ in range(5):
        pts = ico_pts + rng.normal(scale=0.1, size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        inits.append(pts)
        
    # 2. Optimization Loop
    # We will use L-BFGS-B on the coordinates X.
    # To handle the non-differentiability of max/min, we use a smooth approximation (soft-min/max).
    # We will increase beta (sharpness) gradually (annealing) to refine the solution.
    
    betas = [10.0, 50.0, 100.0, 200.0]
    
    for init in inits:
        if time.time() - _t0 > DEADLINE - 20:
            break
            
        current_pts = init.copy()
        
        for beta in betas:
            if time.time() - _t0 > DEADLINE - 10:
                break
                
            # Allocate time based on remaining budget
            time_left = DEADLINE - (time.time() - _t0)
            # Use a small fraction of remaining time for each beta stage
            # Earlier stages (low beta) need less time, later stages (high beta) need more?
            # Actually, low beta is easier to converge. High beta is harder.
            # Let's give more time to high beta.
            alloc_time = min(5.0, max(1.0, (time_left / len(betas))))
            
            refined_pts = _optimize_X(current_pts, beta=beta, time_limit=alloc_time)
            
            # Evaluate true score
            score = _true_score(refined_pts)
            
            if score < best_score:
                best_score = score
                best_pts = refined_pts.copy()
                current_pts = refined_pts.copy()
            else:
                # If it got worse, keep the previous best for this init chain
                current_pts = best_pts.copy() if best_pts is not None else current_pts
                
            # Early stop if we hit a very good score
            if best_score < 1.15:
                break

    # 3. Final Polish with Higher Beta and More Iterations
    if best_pts is not None:
        # Try one more round with very high beta and tighter tolerances
        final_pts = _optimize_X(best_pts, beta=500.0, time_limit=min(10.0, DEADLINE - (time.time() - _t0) - 5))
        final_score = _true_score(final_pts)
        if final_score < best_score:
            best_score = final_score
            best_pts = final_pts

    # Fallback
    if best_pts is None:
        best_pts = fib_pts

    return best_pts
