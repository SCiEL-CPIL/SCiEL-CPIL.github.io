import time
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist

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

def _gram_to_points(G):
    """
    Convert a Gram matrix G to points in R^D.
    G is N x N. We assume G is PSD (or close enough).
    We perform eigendecomposition G = V diag(lam) V^T.
    Points are rows of V[:, :D] * sqrt(lam[:D]).
    """
    # Ensure symmetry
    G = (G + G.T) / 2.0
    
    # Eigendecomposition
    # eigh returns eigenvalues in ascending order
    vals, vecs = eigh(G)
    
    # Clip small negative eigenvalues to 0
    vals = np.maximum(vals, 0)
    
    # Take the top D eigenvalues/vectors
    # Since we want rank <= D, we take the largest D eigenvalues
    # eigh gives ascending, so largest are at the end
    idx = np.argsort(vals)[::-1][:D]
    top_vals = vals[idx]
    top_vecs = vecs[:, idx]
    
    # Reconstruct points: X = V_top * sqrt(diag(lam_top))
    # X is N x D
    pts = top_vecs * np.sqrt(top_vals)
    
    return pts

def _objective_from_gram(G):
    """
    Compute score directly from Gram matrix without explicit point reconstruction 
    for speed during search, but for SA we might need points for exact score.
    However, calculating distances from G is O(N^2) which is fast.
    d2_ij = G_ii + G_jj - 2 G_ij
    """
    # G is N x N
    # d2 matrix
    diag_G = np.diag(G)
    d2 = diag_G[:, None] + diag_G[None, :] - 2 * G
    np.fill_diagonal(d2, 0)
    
    # We only care about i < j
    iu = np.triu_indices(N, k=1)
    d2_flat = d2[iu]
    
    mn = d2_flat.min()
    mx = d2_flat.max()
    
    if mn < 1e-9:
        return np.inf
        
    return (np.sqrt(mx) / np.sqrt(mn)) ** 2

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    rng = np.random.default_rng(42)
    
    # 1. Generate an initial Gram matrix
    # Start with a known good configuration: Fibonacci sphere points
    def get_fib_sphere_pts():
        i = np.arange(N)
        phi = np.arccos(1 - 2 * (i + 0.5) / N)
        theta = np.pi * (1 + 5 ** 0.5) * i
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack([x, y, z])
    
    pts_init = get_fib_sphere_pts()
    G_init = pts_init @ pts_init.T
    
    # Center the points? Gram matrix is invariant to translation if we center points first.
    # Actually, G = (X - mean) (X - mean)^T is the covariance-like matrix.
    # The distances depend on G.
    # Let's ensure G is centered: subtract mean of rows/cols? No, G is defined by centered points.
    # If we generate G from centered points, it's fine.
    
    current_G = G_init.copy()
    current_pts = _gram_to_points(current_G)
    current_score = _true_score(current_pts)
    
    best_G = current_G.copy()
    best_pts = current_pts.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T_start = 0.05  # Initial temperature (perturbation scale for G entries)
    T_end = 1e-6
    alpha = 0.9995  # Cooling rate
    
    # We will perturb G, project to PSD (by clipping eigenvalues), and evaluate.
    # To be efficient, we can perturb G directly.
    
    # Number of steps
    # We have ~140s. 
    # Each step: Perturb G (O(N^2)), Eigendecomp (O(N^3)), Reconstruct (O(N*D^2)), Score (O(N^2)).
    # N=14. N^3 = 2744. This is very fast. 
    # We can do many steps.
    
    steps_limit = 50000
    T = T_start
    
    for step in range(steps_limit):
        if time.time() - _t0 > DEADLINE - 5:
            break
            
        # Perturb the Gram matrix
        # Add Gaussian noise to G
        # Scale noise by T
        noise = rng.normal(0, T, size=(N, N))
        # Make it symmetric
        noise = (noise + noise.T) / 2.0
        
        G_new = current_G + noise
        
        # Project G_new to PSD space
        # Eigendecompose
        vals, vecs = eigh(G_new)
        # Clip negative eigenvalues
        vals_clipped = np.maximum(vals, 0)
        # Reconstruct
        G_psd = vecs @ np.diag(vals_clipped) @ vecs.T
        G_psd = (G_psd + G_psd.T) / 2.0
        
        # Get points and score
        pts_new = _gram_to_points(G_psd)
        score_new = _true_score(pts_new)
        
        # Acceptance criterion (Metropolis)
        if score_new < current_score:
            # Improvement
            current_G = G_psd
            current_pts = pts_new
            current_score = score_new
            
            if current_score < best_score:
                best_score = current_score
                best_G = current_G.copy()
                best_pts = current_pts.copy()
        else:
            # Deterioration
            # Accept with probability exp(-(score_new - current_score) / T)
            # Note: score is ratio^2. Large score is bad.
            # Delta E = score_new - current_score > 0
            # We want to minimize score, so "energy" is score.
            delta = score_new - current_score
            if delta < 1e-9:
                accept = True
            else:
                # Avoid overflow in exp
                if delta / T < 700:
                    accept = rng.random() < np.exp(-delta / T)
                else:
                    accept = False
            
            if accept:
                current_G = G_psd
                current_pts = pts_new
                current_score = score_new
        
        # Cool down
        T *= alpha
        
        # If T gets too small, we might be stuck. 
        # We can restart with a slightly higher T if no improvement for a long time?
        # For now, simple cooling.
        
        # Check for convergence or stagnation
        if T < T_end:
            break
            
    # Final Polish: 
    # Use the best Gram matrix found to get points, then do a few steps of 
    # local optimization on the points using the original gradient method 
    # or just return the best points from SA.
    # Given the budget, SA should have found a good spot.
    # Let's do a quick gradient refinement on the best points found by SA.
    
    if best_pts is not None and time.time() - _t0 < DEADLINE - 10:
        # Refine using scipy minimize on the points, starting from SA result
        # Objective: minimize (max_d/min_d)^2
        # We can use the soft-min/soft-max objective from before, but start from SA result.
        
        def _soft_ratio_obj(v):
            x = v.reshape(N, D)
            x_sq = np.sum(x**2, axis=1)
            d2 = x_sq[:, None] + x_sq[None, :] - 2 * (x @ x.T)
            d2 = np.maximum(d2, 0)
            np.fill_diagonal(d2, 0)
            iu = np.triu_indices(N, k=1)
            d2_flat = d2[iu]
            
            if len(d2_flat) == 0:
                return 0.0
            
            # Soft min and max
            k = 0.1
            m_min = np.min(d2_flat)
            exp_terms_min = np.exp(-(d2_flat - m_min) / k)
            sum_exp_min = np.sum(exp_terms_min)
            soft_min_val = m_min + k * np.log(sum_exp_min)
            
            m_max = np.max(d2_flat)
            exp_terms_max = np.exp((d2_flat - m_max) / k)
            sum_exp_max = np.sum(exp_terms_max)
            soft_max_val = m_max + k * np.log(sum_exp_max)
            
            # Ratio squared approx: (soft_max / soft_min)^2
            # To avoid division by zero, add small epsilon
            if soft_min_val < 1e-9:
                return 1e10
            return (soft_max_val / soft_min_val) ** 2
        
        from scipy.optimize import minimize
        v_start = best_pts.flatten()
        try:
            res = minimize(
                _soft_ratio_obj,
                v_start,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            v_opt = res.x
            pts_opt = v_opt.reshape(N, D)
            pts_opt = pts_opt - pts_opt.mean(axis=0)
            score_opt = _true_score(pts_opt)
            
            if score_opt < best_score:
                best_score = score_opt
                best_pts = pts_opt
        except Exception:
            pass

    if best_pts is None:
        best_pts = get_fib_sphere_pts()
        
    return best_pts
