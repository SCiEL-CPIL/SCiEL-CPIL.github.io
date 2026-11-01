import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """Calculate squared pairwise distances for N points in D dimensions."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """Calculate the actual score: (max_dist / min_dist)^2."""
    sq_dists = get_pairwise_sq_dists(points)
    min_sq = np.min(sq_dists)
    if min_sq < 1e-12:
        return np.inf
    max_sq = np.max(sq_dists)
    return (max_sq / min_sq)

def objective_smooth(x, tau=100.0):
    """
    Smoothed objective function using log-sum-exp for max and min.
    Minimizes log(max_sq_dist) - log(min_sq_dist).
    """
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    
    # Avoid log(0)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max of squared distances
    m_max = np.max(sq_dists)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    
    # Soft min of squared distances
    m_min = np.min(sq_dists)
    log_sum_exp_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    # Objective is log(max) - log(min)
    return np.log(log_sum_exp_max) - np.log(log_sum_exp_min)

def gram_objective(v, tau=50.0, rank_penalty=100.0):
    """
    Objective in terms of Gram matrix vector.
    v is a flattened symmetric matrix of size N x N.
    We enforce PSD and rank <= 3 via eigenvalue penalties.
    """
    G = v.reshape(N, N)
    # Symmetrize
    G = (G + G.T) / 2.0
    
    # Compute eigenvalues for PSD and rank check
    eigvals = np.linalg.eigvalsh(G)
    
    # Penalty for negative eigenvalues (PSD constraint)
    neg_eigs = np.clip(-eigvals, 0, None)
    psd_penalty = rank_penalty * np.sum(neg_eigs**2)
    
    # Penalty for rank > 3 (keep only top 3 eigenvalues, penalize the rest)
    # Sort eigenvalues in descending order
    eigvals_sorted = np.sort(eigvals)[::-1]
    # Penalize eigenvalues beyond the 3rd
    rank_penalty_val = rank_penalty * np.sum(eigvals_sorted[3:]**2)
    
    # Compute squared distances from Gram matrix
    # d_ij^2 = G_ii + G_jj - 2*G_ij
    sq_dists_full = G[np.diag_indices_from(G)][:, np.newaxis] + G[np.diag_indices_from(G)][np.newaxis, :] - 2 * G
    iu = np.triu_indices(N, k=1)
    sq_dists = sq_dists_full[iu]
    
    # Avoid log(0)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max and min
    m_max = np.max(sq_dists)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    
    m_min = np.min(sq_dists)
    log_sum_exp_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    obj = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    return obj + psd_penalty + rank_penalty_val

def gram_to_points(G):
    """Convert Gram matrix to point coordinates via eigendecomposition."""
    G = (G + G.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(G)
    # Keep top 3 eigenvalues
    top3 = np.argsort(eigvals)[::-1][:3]
    eigvals_top = eigvals[top3]
    eigvecs_top = eigvecs[:, top3]
    # Ensure non-negative
    eigvals_top = np.maximum(eigvals_top, 0)
    points = eigvecs_top * np.sqrt(eigvals_top)
    return points

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin before 150s limit
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # --- Strategy 1: Coordinate-space optimization (original approach) ---
    
    # Initial candidates
    # 1. Icosahedron + center + 1 point
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    center = np.array([0.0, 0.0, 0.0])
    v = rng.normal(size=3)
    v = v / np.linalg.norm(v)
    p14 = v * 1.0
    
    init_ico = np.vstack([ico, center, p14])
    coord_candidates = [init_ico.flatten()]
    
    # 2. Random points on a sphere
    for i in range(3):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        coord_candidates.append(pts.flatten())
        
    # 3. Random points in a cube
    for i in range(3):
        pts = rng.uniform(-1, 1, size=(N, D))
        coord_candidates.append(pts.flatten())

    # 4. Fibonacci sphere
    def fibonacci_sphere(n):
        points = []
        golden_angle = np.pi * (3. - np.sqrt(5.))
        for i in range(n):
            y = 1 - (i / float(n - 1)) * 2
            radius = np.sqrt(1 - y**2)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)
    
    fib = fibonacci_sphere(N)
    coord_candidates.append(fib.flatten())

    # Coordinate-space optimization
    n_coord_restarts = 100
    for i in range(n_coord_restarts):
        if time.time() - start_time > timeout - 20:
            break
            
        if i < len(coord_candidates):
            x0 = coord_candidates[i]
        else:
            x0 = rng.uniform(-1, 1, size=(N * D))
            
        try:
            # Multi-stage optimization
            res1 = minimize(
                objective_smooth, 
                x0, 
                method='L-BFGS-B', 
                args=(10.0,),
                options={'maxiter': 80, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            if res1.x is not None:
                pts1 = res1.x.reshape(N, D)
                update_best(pts1)
                
                if time.time() - start_time < timeout - 15:
                    res2 = minimize(
                        objective_smooth, 
                        res1.x, 
                        method='L-BFGS-B', 
                        args=(50.0,),
                        options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-6}
                    )
                    if res2.x is not None:
                        pts2 = res2.x.reshape(N, D)
                        update_best(pts2)
                        
                        if time.time() - start_time < timeout - 10:
                            res3 = minimize(
                                objective_smooth, 
                                res2.x, 
                                method='L-BFGS-B', 
                                args=(200.0,),
                                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                            )
                            if res3.x is not None:
                                pts3 = res3.x.reshape(N, D)
                                update_best(pts3)
                                
        except Exception:
            pass

    # --- Strategy 2: Gram matrix optimization ---
    
    if time.time() - start_time < timeout - 30:
        # Generate initial Gram matrices from candidate point configurations
        gram_candidates = []
        
        for cand in coord_candidates:
            pts = cand.reshape(N, D)
            # Compute Gram matrix: G_ij = p_i . p_j
            G = pts @ pts.T
            # Flatten symmetric part
            iu = np.triu_indices(N)
            gram_candidates.append(G[iu].copy())
        
        # Also add some random PSD matrices
        for i in range(5):
            A = rng.normal(size=(N, 3))
            G = A @ A.T
            iu = np.triu_indices(N)
            gram_candidates.append(G[iu].copy())
        
        n_gram_restarts = 50
        for i in range(n_gram_restarts):
            if time.time() - start_time > timeout - 5:
                break
                
            if i < len(gram_candidates):
                v0 = gram_candidates[i]
            else:
                # Random initial Gram matrix (PSD)
                A = rng.normal(size=(N, 3))
                G = A @ A.T
                iu = np.triu_indices(N)
                v0 = G[iu].copy()
                
            try:
                # Multi-stage Gram optimization
                res_g1 = minimize(
                    gram_objective, 
                    v0, 
                    method='L-BFGS-B', 
                    args=(10.0, 50.0),
                    options={'maxiter': 80, 'ftol': 1e-6, 'gtol': 1e-4}
                )
                if res_g1.x is not None:
                    G1 = res_g1.x.reshape(N, N)
                    G1 = (G1 + G1.T) / 2.0
                    pts_g1 = gram_to_points(G1)
                    update_best(pts_g1)
                    
                    if time.time() - start_time < timeout - 3:
                        res_g2 = minimize(
                            gram_objective, 
                            res_g1.x, 
                            method='L-BFGS-B', 
                            args=(50.0, 100.0),
                            options={'maxiter': 150, 'ftol': 1e-8, 'gtol': 1e-6}
                        )
                        if res_g2.x is not None:
                            G2 = res_g2.x.reshape(N, N)
                            G2 = (G2 + G2.T) / 2.0
                            pts_g2 = gram_to_points(G2)
                            update_best(pts_g2)
                            
                            if time.time() - start_time < timeout - 1:
                                res_g3 = minimize(
                                    gram_objective, 
                                    res_g2.x, 
                                    method='L-BFGS-B', 
                                    args=(200.0, 200.0),
                                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                                )
                                if res_g3.x is not None:
                                    G3 = res_g3.x.reshape(N, N)
                                    G3 = (G3 + G3.T) / 2.0
                                    pts_g3 = gram_to_points(G3)
                                    update_best(pts_g3)
                                    
            except Exception:
                pass

    # --- Final polish: use the best points found and do a high-precision coordinate optimization ---
    if best_points is not None and time.time() - start_time < timeout - 2:
        try:
            res_final = minimize(
                objective_smooth, 
                best_points.flatten(), 
                method='L-BFGS-B', 
                args=(500.0,),
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_final.x is not None:
                pts_final = res_final.x.reshape(N, D)
                update_best(pts_final)
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = coord_candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
