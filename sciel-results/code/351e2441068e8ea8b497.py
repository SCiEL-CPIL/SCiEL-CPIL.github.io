"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 13

# EVOLVE-BLOCK-START

# Pre-compute triangle indices for efficiency
# i < j < k
_idx_i, _idx_j, _idx_k = np.indices((N, N, N)).reshape(3, -1)
_mask = (_idx_i < _idx_j) & (_idx_j < _idx_k)
_TRI_I = _idx_i[_mask]
_TRI_J = _idx_j[_mask]
_TRI_K = _idx_k[_mask]

def _compute_score(pts):
    """
    Compute the minimum triangle area normalized by the convex hull area.
    Returns 0 if invalid.
    """
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return 0.0
        
    # Compute Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Compute all triangle areas vectorized
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _smooth_min_area(pts, beta):
    """
    Compute a smooth approximation of the minimum triangle area using log-sum-exp.
    """
    p_i = pts[_TRI_I]
    p_j = pts[_TRI_J]
    p_k = pts[_TRI_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_a = np.min(areas)
    if min_a < 1e-10:
         return 0.0
         
    y = -beta * (areas - min_a)
    sum_exp = np.sum(np.exp(y))
    if sum_exp == 0:
        return min_a
    log_sum_exp = np.log(sum_exp)
    
    soft_min = min_a - (1.0 / beta) * log_sum_exp
    return soft_min

def _angular_objective(params, beta):
    """
    Objective for angular parameterization.
    params: [r0, ..., r12, theta0, ..., theta12]
    r_i > 0, theta_i sorted.
    """
    radii = np.exp(params[:N]) # Ensure positive
    angles = np.cumsum(np.abs(params[N:]) + 0.1) # Ensure sorted and positive steps
    angles = angles % (2 * np.pi)
    # Sort angles to ensure convex ordering? 
    # Actually, if we define points by angle, they are naturally sorted by angle.
    # But we need to map them to a convex shape.
    # A simpler convex representation: 
    # Points on a convex curve can be represented by their support function or just polar coordinates with sorted angles.
    # If angles are sorted, the polygon is star-shaped with respect to origin.
    # To ensure convexity, we need to be careful.
    # Let's stick to Cartesian for the final polish, but use Angular for global search.
    
    pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    
    # Check convexity
    # Compute cross products of consecutive edges
    if N < 3: return 1e9
    p0 = pts[-1]
    non_convex_count = 0
    for i in range(N):
        p1 = pts[i]
        p2 = pts[(i+1)%N]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if cross < 0: # Assuming CCW
            non_convex_count += 1
        p0 = p1
    # If more than 1 non-convex, it's likely self-intersecting or concave
    # For Heilbronn, we want convex hull to be the set of points.
    # If the points form a convex polygon, all crosses should have the same sign.
    # Let's just check if the area is positive and hull area matches polygon area roughly?
    # Simpler: just use the score. If it's self-intersecting, hull area will be larger than polygon area,
    # but min triangle area might still be computed on the points.
    # The score is min_area / hull_area.
    # If self-intersecting, hull_area > polygon_area, so score is lower.
    # This naturally penalizes non-convex arrangements if we assume the "intended" shape is convex.
    
    h_area = _hull_area(pts)
    if h_area < 1e-9:
        return 1e9
        
    s_min = _smooth_min_area(pts, beta)
    if s_min <= 0:
        return 1e9
        
    return -(s_min / h_area)

def _spectral_objective(params, beta):
    """
    Objective for spectral graph theory approach.
    params: flattened banded SPD matrix L.
    We use a banded matrix of width 2 (tridiagonal) to enforce locality.
    L is N x N.
    We take the eigenvectors of L. The first 2 eigenvectors correspond to the null space (constant vector and maybe another if not connected?).
    For a connected graph, L has 1 zero eigenvalue.
    We want the points to be the eigenvectors? No, that's not right.
    
    Alternative Spectral View:
    Let points be $x_i \in \mathbb{R}^2$.
    Consider the Gram matrix $G = X X^T$.
    If we define a graph Laplacian $L$ on the points, the eigenvectors of $L$ are related to the geometry.
    
    Let's try a different spectral approach:
    Represent the points as the first 13 rows of the eigenvector matrix of a learnable SPD matrix $M$.
    $M$ is $13 \times 13$.
    Eigenvectors $V$.
    Points = $V[:, :2]$? No, $V$ is $13 \times 13$.
    We can take the first 2 columns of $V$ as the coordinates.
    This ensures the points are orthogonal in the spectral sense?
    
    Actually, a common trick for "smooth" manifolds is to use the eigenvectors of a banded matrix.
    Let $L$ be a banded SPD matrix.
    Compute eigenvectors $U$ of $L$.
    Let the points be $U[:, 1:3]$ (skipping the constant vector).
    Scale and shift to unit area.
    
    Let's implement this.
    """
    # Reconstruct banded SPD matrix
    # Band width 2: diagonal and off-diagonals.
    # N=13.
    # Diagonal: 13 params.
    # Off-diagonal 1: 12 params.
    # Off-diagonal 2: 11 params.
    # Total 36 params.
    
    # Ensure SPD by adding large identity.
    L = np.zeros((N, N))
    idx = 0
    # Diagonal
    for i in range(N):
        L[i, i] = params[idx] + 10.0 # Large shift for SPD
        idx += 1
    # Off-diagonal 1
    for i in range(N-1):
        val = params[idx]
        L[i, i+1] = val
        L[i+1, i] = val
        idx += 1
    # Off-diagonal 2
    for i in range(N-2):
        val = params[idx]
        L[i, i+2] = val
        L[i+2, i] = val
        idx += 1
        
    try:
        # Symmetrize just in case
        L = (L + L.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(L)
    except Exception:
        return 1e9
        
    # Take the first 2 non-trivial eigenvectors?
    # The smallest eigenvalue is usually the constant vector.
    # We want the "smooth" modes.
    # Let's take eigenvectors corresponding to the 2nd and 3rd smallest eigenvalues.
    # These should represent the fundamental shape.
    
    # Sort eigenvectors by eigenvalue
    idx_sort = np.argsort(eigvals)
    eigvecs_sorted = eigvecs[:, idx_sort]
    
    # Use columns 1 and 2 (0 is constant)
    pts = eigvecs_sorted[:, 1:3].T # Shape (2, N) -> (N, 2)
    
    # Center and scale
    pts = pts - np.mean(pts, axis=0)
    
    # Check if degenerate
    if np.linalg.matrix_rank(pts) < 2:
        return 1e9
        
    # Scale to unit area?
    # We need to normalize by hull area in the score, so scaling doesn't affect the ratio.
    # But we need to avoid numerical issues.
    # Let's normalize so that the max radius is 1.
    radii = np.linalg.norm(pts, axis=1)
    max_r = np.max(radii)
    if max_r < 1e-9:
        return 1e9
    pts = pts / max_r
    
    h_area = _hull_area(pts)
    if h_area < 1e-9:
        return 1e9
        
    s_min = _smooth_min_area(pts, beta)
    if s_min <= 0:
        return 1e9
        
    return -(s_min / h_area)

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Spectral Approach
    # Initialize banded matrix params
    # Diagonal: 10, Off1: -1, Off2: 0.1
    init_params_spectral = np.zeros(36)
    init_params_spectral[:N] = 10.0
    init_params_spectral[N:N-1] = -1.0
    init_params_spectral[N-1:] = 0.1
    
    try:
        res_spectral = minimize(
            _spectral_objective, 
            init_params_spectral, 
            args=(50.0,),
            method='BFGS', 
            options={'maxiter': 100, 'gtol': 1e-6}
        )
        if res_spectral.fun < 1e8:
            # Reconstruct points from best params
            # We need to extract points from the objective function logic
            # Let's just re-run the extraction
            L = np.zeros((N, N))
            idx = 0
            for i in range(N):
                L[i, i] = res_spectral.x[idx] + 10.0
                idx += 1
            for i in range(N-1):
                val = res_spectral.x[idx]
                L[i, i+1] = val
                L[i+1, i] = val
                idx += 1
            for i in range(N-2):
                val = res_spectral.x[idx]
                L[i, i+2] = val
                L[i+2, i] = val
                idx += 1
            L = (L + L.T) / 2.0
            eigvals, eigvecs = np.linalg.eigh(L)
            idx_sort = np.argsort(eigvals)
            eigvecs_sorted = eigvecs[:, idx_sort]
            pts = eigvecs_sorted[:, 1:3].T
            pts = pts - np.mean(pts, axis=0)
            radii = np.linalg.norm(pts, axis=1)
            max_r = np.max(radii)
            if max_r > 1e-9:
                pts = pts / max_r
                score = _compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
    except Exception:
        pass

    # Phase 2: Angular Parameterization (Fallback/Refinement)
    # This is known to be robust for convex shapes
    if (time.time() - start_time) < deadline - 10.0:
        # Init: regular polygon
        angles_init = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii_init = np.ones(N)
        params_init = np.concatenate([np.log(radii_init), angles_init])
        
        # Optimize
        try:
            res_angular = minimize(
                _angular_objective, 
                params_init, 
                args=(50.0,),
                method='BFGS', 
                options={'maxiter': 200, 'gtol': 1e-6}
            )
            if res_angular.fun < 1e8:
                radii = np.exp(res_angular.x[:N])
                angles = np.cumsum(np.abs(res_angular.x[N:]) + 0.1)
                angles = angles % (2 * np.pi)
                # Ensure angles are sorted and cover 2pi?
                # The cumsum ensures monotonicity.
                # We need to map them to 0..2pi properly.
                # The objective function handles the mapping.
                # Let's re-extract points using the same logic as objective
                pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
                
                # Check if this is better
                score = _compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
        except Exception:
            pass

    # Phase 3: Cartesian Refinement (Nelder-Mead)
    if best_pts is not None and (time.time() - start_time) < deadline - 5.0:
        try:
            def _exact_obj(params):
                pts = params.reshape(N, 2)
                s = _compute_score(pts)
                return -s if s > 0 else 1e9
            
            res_polish = minimize(
                _exact_obj, 
                best_pts.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res_polish.fun < 1e8:
                polish_pts = res_polish.x.reshape(N, 2)
                polish_score = _compute_score(polish_pts)
                
                if polish_score > best_score:
                    best_score = polish_score
                    best_pts = polish_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        return np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_pts

# EVOLVE-BLOCK-END
