import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.linalg import eigh
from scipy.special import logsumexp

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def gram_to_points(G, target_dim=D):
    """
    Convert Gram matrix to points in target_dim dimensions.
    Uses eigendecomposition and projects to lowest-dimensional subspace.
    """
    eigvals, eigvecs = eigh(G)
    # Sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take top min(target_dim, N) eigenvalues
    k = min(target_dim, len(eigvals))
    eigvals_k = np.maximum(eigvals[:k], 0)
    eigvecs_k = eigvecs[:, :k]
    
    # Points = eigvecs_k^T * sqrt(eigvals_k)
    points = eigvecs_k.T * np.sqrt(eigvals_k)
    return points

def raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return np.inf
    return (np.max(sq_dists) / min_d)

def normalize_points(points):
    """
    Center and normalize points to unit mean squared norm.
    """
    points = points - np.mean(points, axis=0)
    mean_norm_sq = np.mean(np.sum(points**2, axis=1))
    if mean_norm_sq > 1e-12:
        points = points / np.sqrt(mean_norm_sq)
    return points

def objective_smooth(x, tau=100.0):
    """
    Smooth approximation of the log ratio of max to min squared distance.
    Uses scipy's logsumexp for numerical stability.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max using logsumexp
    log_sum_exp_max = logsumexp(tau * sq_dists) / tau
    
    # Soft min using logsumexp on negated values
    log_sum_exp_min = -logsumexp(-tau * sq_dists) / tau
    
    # Minimize log(soft_max) - log(soft_min)
    obj_val = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    return obj_val

def objective_gram_smooth(g_vec, tau=100.0, rank_penalty=1000.0):
    """
    Objective for Gram matrix optimization with rank constraint.
    g_vec contains the upper triangle of the Gram matrix G (size N x N).
    """
    # Reconstruct Gram matrix from upper triangle
    G = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(i, N):
            G[i, j] = g_vec[idx]
            G[j, i] = g_vec[idx]
            idx += 1
    
    # Ensure PSD by clipping eigenvalues
    eigvals, eigvecs = eigh(G)
    eigvals_clipped = np.maximum(eigvals, 0)
    
    # Rank penalty: penalize eigenvalues beyond D-th
    if len(eigvals_clipped) > D:
        rank_pen = rank_penalty * np.sum(eigvals_clipped[D:]**2)
    else:
        rank_pen = 0.0
    
    # Trace normalization penalty
    trace = np.sum(eigvals_clipped)
    trace_pen = 10.0 * (trace - N)**2
    
    # Extract points
    points = gram_to_points(G, target_dim=D)
    points = normalize_points(points)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max and min
    log_sum_exp_max = logsumexp(tau * sq_dists) / tau
    log_sum_exp_min = -logsumexp(-tau * sq_dists) / tau
    
    obj_val = np.log(log_sum_exp_max) - np.log(log_sum_exp_min) + rank_pen + trace_pen
    
    return obj_val

def objective_epigraph(x, t, tau=100.0):
    """
    Epigraph formulation with penalty method.
    """
    pts = x[:N*D].reshape(N, D)
    t = x[N*D]
    
    if t < 1e-6:
        t = 1e-6
    
    pts = normalize_points(pts)
    sq_dists = get_pairwise_sq_dists(pts)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Penalty for d^2 < 1/t
    viol_low = np.maximum(1.0/t - sq_dists, 0)**2
    # Penalty for d^2 > t
    viol_high = np.maximum(sq_dists - t, 0)**2
    
    lam = 500.0
    return t + lam * (np.sum(viol_low) + np.sum(viol_high))

def objective_direct(x):
    """
    Direct non-smooth objective for SLSQP or other methods that can handle it.
    Minimizes log(max) - log(min) or just ratio.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return 1e10
    return np.log(np.max(sq_dists)) - np.log(np.min(sq_dists))

def objective_gram(g_vec, tau=100.0):
    """
    Objective for Gram matrix optimization.
    g_vec contains the upper triangle of the Gram matrix G (size N x N).
    We reconstruct G, check PSD, extract points, and compute the smooth ratio.
    """
    # Reconstruct Gram matrix from upper triangle
    G = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(i, N):
            G[i, j] = g_vec[idx]
            G[j, i] = g_vec[idx]
            idx += 1
    
    # Ensure PSD by clipping eigenvalues
    eigvals, eigvecs = eigh(G)
    eigvals_clipped = np.maximum(eigvals, 0)
    
    # Extract points
    points = gram_to_points(G, target_dim=D)
    points = normalize_points(points)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-15)
    
    # Soft max and min
    log_sum_exp_max = logsumexp(tau * sq_dists) / tau
    log_sum_exp_min = -logsumexp(-tau * sq_dists) / tau
    
    obj_val = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    return obj_val

def points_to_gram_vec(points):
    """Convert points to Gram matrix upper triangle vector."""
    points = normalize_points(points)
    G = points.T @ points
    # Extract upper triangle including diagonal
    iu = np.triu_indices(N)
    return G[iu]

def fibonacci_sphere(n, seed=42):
    """Generate n points on a sphere using Fibonacci lattice."""
    rng = np.random.default_rng(seed)
    golden_angle = np.pi * (3 - np.sqrt(5))
    points = np.zeros((n, D))
    for i in range(n):
        z = 1 - 2 * i / (n - 1)
        r = np.sqrt(max(0, 1 - z**2))
        theta = golden_angle * i + rng.uniform(0, 2*np.pi) * 0.01
        points[i, 0] = r * np.cos(theta)
        points[i, 1] = r * np.sin(theta)
        points[i, 2] = z
    return points

def icosahedron_points():
    """Return 12 vertices of a regular icosahedron on unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    return ico

def generate_diverse_initializations(rng, num_candidates=20):
    """Generate diverse initial point configurations."""
    candidates = []
    
    # 1. Fibonacci sphere
    candidates.append(fibonacci_sphere(N))
    
    # 2. Icosahedron + poles
    ico = icosahedron_points()
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    candidates.append(np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)]))
    
    # 3. Tetrahedron-based
    tetra = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    tetra = tetra / np.linalg.norm(tetra, axis=1, keepdims=True)
    
    edge_mids = []
    for i in range(4):
        for j in range(i+1, 4):
            mid = (tetra[i] + tetra[j]) / 2
            mid = mid / np.linalg.norm(mid)
            edge_mids.append(mid)
    
    face_centers = []
    for i in range(4):
        for j in range(i+1, 4):
            for k in range(j+1, 4):
                center = (tetra[i] + tetra[j] + tetra[k]) / 3
                center = center / np.linalg.norm(center)
                face_centers.append(center)
    
    candidates.append(np.vstack([tetra, np.array(edge_mids), np.array(face_centers[:4])]))
    
    # 4. Random sphere points
    for _ in range(num_candidates - 3):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts)
    
    return candidates

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin for 150s limit
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        pts = normalize_points(pts)
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    candidates = generate_diverse_initializations(rng, num_candidates=20)
    
    # Optimization Loop
    n_restarts = len(candidates)
    
    # Phase 1: Smooth Log-Ratio Annealing with multiple tau values
    taus = [10.0, 50.0, 200.0, 1000.0]
    
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 30:
            break
            
        x0 = candidates[i].copy()
        x0 = normalize_points(x0)
        x0 = x0.flatten()
        
        try:
            current_x = x0
            
            # Phase 1: Standard smooth optimization with annealing
            for tau in taus:
                if time.time() - start_time > timeout - 30:
                    break
                
                res = minimize(
                    objective_smooth, 
                    current_x, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-11}
                )
                
                if res.x is not None:
                    current_x = res.x
                    pts_current = current_x.reshape(N, D)
                    update_best(pts_current)
            
            # Phase 2: Gram matrix optimization with rank penalty
            if time.time() - start_time < timeout - 25:
                pts_current = current_x.reshape(N, D)
                pts_current = normalize_points(pts_current)
                g0 = points_to_gram_vec(pts_current)
                
                # Try with different rank penalties
                for rank_pen in [500.0, 2000.0]:
                    if time.time() - start_time > timeout - 20:
                        break
                    
                    res_gram = minimize(
                        objective_gram_smooth,
                        g0,
                        method='L-BFGS-B',
                        args=(200.0, rank_pen),
                        options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-11}
                    )
                    
                    if res_gram.x is not None:
                        # Reconstruct points from optimized Gram
                        G = np.zeros((N, N))
                        idx = 0
                        for ii in range(N):
                            for jj in range(ii, N):
                                G[ii, jj] = res_gram.x[idx]
                                G[jj, ii] = res_gram.x[idx]
                                idx += 1
                        
                        points_gram = gram_to_points(G, target_dim=D)
                        points_gram = normalize_points(points_gram)
                        update_best(points_gram)
            
            # Phase 3: Epigraph refinement with SLSQP
            if time.time() - start_time < timeout - 15:
                pts_current = current_x.reshape(N, D)
                pts_current = normalize_points(pts_current)
                sq_dists = get_pairwise_sq_dists(pts_current)
                
                # Initial t estimate: sqrt(max/min)
                t0 = np.sqrt(np.max(sq_dists) / np.min(sq_dists))
                
                x_epigraph = np.concatenate([pts_current.flatten(), [t0]])
                
                # Bounds for t
                bounds = [(None, None)] * (N * D) + [(1.0, 20.0)]
                
                res = minimize(
                    objective_epigraph, 
                    x_epigraph, 
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-13}
                )
                
                if res.x is not None:
                    pts_epigraph = res.x[:N*D].reshape(N, D)
                    update_best(pts_epigraph)
            
            # Phase 4: Direct non-smooth polish with SLSQP
            if time.time() - start_time < timeout - 10:
                pts_current = current_x.reshape(N, D)
                pts_current = normalize_points(pts_current)
                
                res = minimize(
                    objective_direct, 
                    pts_current.flatten(), 
                    method='SLSQP',
                    options={'maxiter': 200, 'ftol': 1e-15}
                )
                
                if res.x is not None:
                    pts_direct = res.x.reshape(N, D)
                    update_best(pts_direct)
        
        except Exception:
            pass
    
    # Final high-precision polish pass on best point
    if best_points is not None and time.time() - start_time < timeout - 10:
        try:
            x_polish = best_points.flatten()
            # Use very high tau for precise max/min
            res = minimize(
                objective_smooth, 
                x_polish, 
                method='L-BFGS-B', 
                args=(5000.0,),
                options={'maxiter': 1000, 'ftol': 1e-17, 'gtol': 1e-13}
            )
            if res.x is not None:
                pts_polish = res.x.reshape(N, D)
                update_best(pts_polish)
                
            # Another SLSQP pass on the result
            if time.time() - start_time < timeout - 5:
                res = minimize(
                    objective_direct, 
                    best_points.flatten(), 
                    method='SLSQP',
                    options={'maxiter': 500, 'ftol': 1e-17}
                )
                if res.x is not None:
                    pts_polish2 = res.x.reshape(N, D)
                    update_best(pts_polish2)
        except Exception:
            pass

    if best_points is None:
        best_points = normalize_points(candidates[0])
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
