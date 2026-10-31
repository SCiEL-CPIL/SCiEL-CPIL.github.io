import numpy as np
import time
import scipy.spatial as spat
import scipy.optimize
import itertools
import sys

N = 13

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(points) < 3:
        return 0.0
    
    # Compute convex hull area
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    pts = points
    n = len(pts)
    
    # Efficient vectorized triangle area calculation
    # Pre-computed indices for N=13
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    v1 = pts[j_flat] - pts[i_flat] 
    v2 = pts[k_flat] - pts[i_flat]
    
    # Determinant in 2D
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _psd_project(G):
    """
    Project a symmetric matrix G onto the Positive Semidefinite (PSD) cone.
    Uses eigenvalue decomposition.
    """
    eigvals, eigvecs = np.linalg.eigh(G)
    # Clamp negative eigenvalues to 0
    eigvals_clamped = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T

def _sdp_optimize(start_time, deadline):
    """
    Attempts to optimize the Gram matrix G (size 13x13) to maximize the 
    minimum squared triangle area, subject to G being PSD and having rank <= 2 
    (implicitly handled by projection, though rank constraint is non-convex, 
    we rely on the structure of the objective and initialization to stay low rank).
    
    Note: Maximizing min(area^2) is not convex in G directly because area^2 
    involves determinants of submatrices which are not linear in G. 
    However, for points in R^2, the squared area of triangle (i,j,k) is 
    0.25 * ( (G_ii + G_jj + G_kk - 2G_ij - 2G_ik - 2G_jk) * something? No.
    
    Let's use the property:
    Area^2(i,j,k) = 0.25 * det( [ 1 1 1; x_i x_j x_k; y_i y_j y_k ] )^2
    This is complex in terms of G.
    
    Alternative: Use the fact that for centered points, the covariance matrix is G.
    Actually, let's stick to a simpler heuristic inspired by SDP:
    We optimize the Gram matrix G such that the "spread" is maximized.
    
    A more robust structural change:
    Use a **Simulated Annealing** approach on the Gram matrix, with moves that 
    preserve PSDness via projection, and a cooling schedule. This is a global 
    search method distinct from local gradient/perturbation.
    """
    
    # Initialize G with a random low-rank PSD matrix
    # Rank 2 structure: G = X X^T where X is 13x2
    np.random.seed(42)
    X = np.random.randn(N, 2)
    G = X @ X.T
    
    best_score = -1.0
    best_points = None
    
    # We will optimize X directly (the embedding), which ensures G is PSD and Rank <= 2.
    # This is equivalent to optimizing points in R^2, but we can use smarter moves.
    
    # Start with a regular polygon configuration as a strong baseline
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    X_init = np.column_stack((np.cos(angles), np.sin(angles)))
    
    current_X = X_init.copy()
    current_score = calculate_score(current_X)
    best_score = current_score
    best_X = current_X.copy()
    
    # Simulated Annealing on X
    T = 1.0
    T_min = 1e-6
    cooling_rate = 0.99
    max_iter = 5000
    
    # Precompute indices for fast scoring
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    def fast_score(X):
        try:
            hull = spat.ConvexHull(X)
            hull_area = hull.volume
            if hull_area <= 1e-9:
                return 0.0
            v1 = X[j_flat] - X[i_flat] 
            v2 = X[k_flat] - X[i_flat]
            det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(det)
            return np.min(areas) / hull_area
        except Exception:
            return 0.0

    # Move: Perturb one point
    for iter_idx in range(max_iter):
        if time.time() > deadline - 5.0:
            break
            
        # Choose a point to move
        p_idx = np.random.randint(N)
        
        # Gaussian perturbation
        delta = np.random.randn(2) * 0.1 * T
        
        X_new = current_X.copy()
        X_new[p_idx] += delta
        
        score_new = fast_score(X_new)
        
        # Acceptance criterion
        if score_new > current_score:
            current_X = X_new
            current_score = score_new
            if current_score > best_score:
                best_score = current_score
                best_X = current_X.copy()
        else:
            # Metropolis criterion
            delta_score = score_new - current_score
            if delta_score < 0:
                prob = np.exp(delta_score / T)
                if np.random.rand() < prob:
                    current_X = X_new
                    current_score = score_new
            else:
                current_X = X_new
                current_score = score_new
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_X, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_score = -1.0
    best_points = None
    
    # Strategy 1: Simulated Annealing on the point coordinates
    # This is a structural change from the affine-transformed grid search.
    # It explores the space of all 13-point configurations directly.
    
    # Initial guess: Regular polygon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Run SA
    sa_points, sa_score = _sdp_optimize(start_time, deadline)
    
    if sa_score > best_score:
        best_score = sa_score
        best_points = sa_points
        
    # Strategy 2: If SA is slow or stuck, try a quick random restart with local polish
    # (Fallback to ensure we return something reasonable)
    if time.time() < deadline - 10.0:
        for _ in range(5):
            if time.time() > deadline - 5.0:
                break
            # Random perturbation of regular polygon
            p = np.column_stack((np.cos(angles), np.sin(angles)))
            p += np.random.randn(N, 2) * 0.1
            score = calculate_score(p)
            if score > best_score:
                best_score = score
                best_points = p

    # Final polish: Local search on the best point found
    if best_points is not None and time.time() < deadline - 2.0:
        # Simple local search
        current = best_points.copy()
        current_score = best_score
        step = 0.01
        for _ in range(200):
            if time.time() > deadline - 2.0:
                break
            improved = False
            for i in range(N):
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0: continue
                        new_p = current.copy()
                        new_p[i, 0] += dx
                        new_p[i, 1] += dy
                        s = calculate_score(new_p)
                        if s > current_score:
                            current = new_p
                            current_score = s
                            improved = True
                            break
                    if improved: break
                if improved: break
            if not improved:
                step *= 0.8
                if step < 1e-5: break
        if current_score > best_score:
            best_score = current_score
            best_points = current

    if best_points is None:
        # Absolute fallback
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points
