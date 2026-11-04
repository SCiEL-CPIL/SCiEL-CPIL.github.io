"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute triangle indices
TRI_INDICES = list(combinations(range(N), 3))
NUM_TRIS = len(TRI_INDICES)  # 286

# Precompute index arrays for vectorized area computation
_I = np.array([t[0] for t in TRI_INDICES])
_J = np.array([t[1] for t in TRI_INDICES])
_K = np.array([t[2] for t in TRI_INDICES])

def _tri_areas(pts):
    """Vectorized triangle areas for all 286 triples."""
    pts = np.asarray(pts, dtype=np.float64)
    pi = pts[_I]
    pj = pts[_J]
    pk = pts[_K]
    v1 = pj - pi
    v2 = pk - pi
    return 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])

def _hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
        return 0.0
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    min_area = np.min(_tri_areas(pts))
    return min_area / ha

def _soft_min_objective(x, beta=50.0):
    """
    Soft-min of triangle areas / hull area, negative for minimization.
    x is a flat array of length 2*N.
    """
    pts = x.reshape(-1, 2)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-10:
        return 1e6
    
    areas = _tri_areas(pts)
    if len(areas) == 0:
        return 1e6
    
    # Soft min using log-sum-exp trick for numerical stability
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    return -soft_min_area / hull_area

def _exact_objective(x):
    """Exact negative score for Nelder-Mead refinement."""
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _regular_polygon(n, radius=1.0, center=(0,0)):
    """Generate regular polygon vertices."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius*np.cos(angles), 
                            center[1] + radius*np.sin(angles)])

def _halton_sequence(n, base):
    """Generate first n Halton sequence values in [0,1)."""
    seq = np.zeros(n)
    f = 1.0
    for i in range(1, n + 1):
        f /= base
        seq[i - 1] = 0.0
        j = i
        while j > 0:
            seq[i - 1] += (j % base) * f
            j //= base
    return seq

def _halton_points(n, base1=2, base2=3):
    """Generate n points using 2D Halton sequence, scaled to unit circle."""
    x = _halton_sequence(n, base1)
    y = _halton_sequence(n, base2)
    pts = np.column_stack([x, y])
    # Normalize to unit disk
    radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    radii[radii < 1e-10] = 1.0
    pts = pts / radii[:, np.newaxis]
    return pts

def _center_ring(n_outer, n_inner, r_outer=1.0, r_inner=0.5):
    """Points on two concentric rings."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
    inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    return np.vstack([outer, inner])

def _simulated_annealing(pts, time_budget, rng):
    """
    Perform Simulated Annealing on the exact objective.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_score = _score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    T = 0.01
    cooling_rate = 0.95
    min_T = 1e-6
    step_size = 0.05
    
    while time.time() - start_time < time_budget - 1 and T > min_T:
        idx = rng.randint(N)
        delta = rng.normal(0, step_size, 2)
        new_pts = current_pts.copy()
        new_pts[idx] += delta
        
        new_score = _score(new_pts)
        
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            delta_score = current_score - new_score
            if delta_score < 1e-12:
                accept = True
            else:
                try:
                    prob = np.exp(-delta_score / T)
                except OverflowError:
                    prob = 0.0
                accept = rng.rand() < prob
            
            if accept:
                current_pts = new_pts
                current_score = new_score
        
        T *= cooling_rate
        
        if rng.rand() < 0.05:
            step_size *= 0.9

    return best_pts, best_score

def _l_bfgs_b_polish(pts, time_budget, beta=50.0):
    """
    Polish using L-BFGS-B on the soft-min objective.
    """
    def obj(x):
        return _soft_min_objective(x, beta)
    
    try:
        res = minimize(obj, pts.flatten(), method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        return cand, s
    except Exception:
        return pts, _score(pts)

def _nelder_mead_polish(pts, time_budget):
    """
    Final polish using Nelder-Mead on exact objective.
    """
    def obj(x):
        return _exact_objective(x)
    
    try:
        res = minimize(obj, pts.flatten(), method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                'initial_simplex': 1e-2})
        cand = res.x.reshape(N, 2)
        s = _score(cand)
        return cand, s
    except Exception:
        return pts, _score(pts)

def _spectral_to_points(spectral_params, n_points=N):
    """
    Convert spectral parameters to 2D points.
    
    spectral_params: flat array of length n_points*(n_points+1)//2
    representing upper triangle of a symmetric matrix M.
    
    We construct M as SPD, compute eigendecomposition, and use
    the first 2 eigenvectors' rows as point coordinates.
    """
    n = n_points
    # Reconstruct symmetric matrix from upper triangle
    idx = np.triu_indices(n)
    M = np.zeros((n, n))
    M[idx] = spectral_params
    M = M + M.T - np.diag(np.diag(M))  # Make symmetric
    
    # Ensure SPD by adding identity with small regularization
    M = M + np.eye(n) * 1e-4
    
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(M)
    
    # Take first 2 eigenvectors (columns 0 and 1)
    # Each eigenvector is length n, so we get n points in 2D
    pts = eigvecs[:, :2]
    
    # Center and scale
    pts = pts - pts.mean(axis=0)
    norm = np.linalg.norm(pts, axis=1)
    norm[norm < 1e-10] = 1.0
    pts = pts / norm[:, np.newaxis]
    
    return pts

def _points_to_spectral(pts, n_points=N):
    """
    Convert 2D points back to spectral parameters (approximate inverse).
    This is used for initialization and isn't a perfect inverse.
    """
    n = n_points
    # Create a random SPD matrix that has similar eigenvector structure
    rng = np.random.RandomState(42)
    M = rng.randn(n, n)
    M = M + M.T
    M = M @ M.T + np.eye(n) * 0.1
    
    # Adjust to better match the given points
    # Use the points to create a correlation-like matrix
    pts_centered = pts - pts.mean(axis=0)
    # Create a matrix where eigenvectors are related to point positions
    # Simple approach: use outer product structure
    for i in range(n):
        for j in range(i+1, n):
            M[i,j] += 0.01 * pts_centered[i,0] * pts_centered[j,0] + 0.01 * pts_centered[i,1] * pts_centered[j,1]
            M[j,i] = M[i,j]
    
    # Extract upper triangle
    idx = np.triu_indices(n)
    return M[idx]

def _spectral_objective(spectral_params, beta=50.0):
    """
    Objective function for spectral parameter optimization.
    Converts spectral params to points, then computes negative soft-min score.
    """
    try:
        pts = _spectral_to_points(spectral_params)
        return _soft_min_objective(pts.flatten(), beta)
    except Exception:
        return 1e6

def _spectral_optimize(initial_params, time_budget, beta=50.0):
    """
    Optimize spectral parameters using L-BFGS-B.
    """
    def obj(x):
        return _spectral_objective(x, beta)
    
    try:
        res = minimize(obj, initial_params, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
        spectral_params = res.x
        pts = _spectral_to_points(spectral_params)
        s = _score(pts)
        return pts, s
    except Exception:
        return None, 0.0

def _spectral_init(rng, n_points=N):
    """
    Generate initial spectral parameters.
    """
    n = n_points
    # Start with a diagonal-dominant SPD matrix
    diag_vals = rng.uniform(0.5, 2.0, n)
    off_diag = rng.normal(0, 0.1, (n, n))
    off_diag = off_diag + off_diag.T
    np.fill_diagonal(off_diag, 0)
    
    M = np.diag(diag_vals) + off_diag
    
    # Ensure SPD
    eigvals = np.linalg.eigvalsh(M)
    if np.min(eigvals) < 1e-3:
        M = M + np.eye(n) * (1e-3 - np.min(eigvals) + 1e-4)
    
    # Extract upper triangle
    idx = np.triu_indices(n)
    return M[idx]

def construct():
    """Main entry point: find optimal 13-point configuration using spectral approach."""
    start_time = time.time()
    total_budget = 145.0  # Safety margin for 150s limit
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_ratio = -1e9
    
    # Generate diverse starting points for traditional approach
    starts = []
    starts.append(_regular_polygon(N, radius=1.0))
    starts.append(_halton_points(N))
    for n_outer in [7, 8, 9]:
        n_inner = N - n_outer
        for r_inner in [0.4, 0.5, 0.6]:
            starts.append(_center_ring(n_outer, n_inner, 1.0, r_inner))
    for _ in range(3):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2*np.pi, N)
        starts.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))

    # Evaluate initial candidates
    for cand in starts:
        if time.time() - start_time > total_budget:
            break
        try:
            s = _score(cand)
            if s > best_ratio:
                best_ratio = s
                best_points = cand.copy()
        except Exception:
            pass
    
    # Sort starts by initial score
    scored_starts = []
    for start in starts:
        try:
            s = _score(start)
            scored_starts.append((s, start))
        except Exception:
            scored_starts.append((0.0, start))
    scored_starts.sort(key=lambda x: -x[0])
    
    # Phase 1: Traditional optimization from best starts
    n_starts = min(5, len(scored_starts))
    base_time = (total_budget * 0.4) / max(1, n_starts)
    
    for i in range(n_starts):
        if time.time() - start_time > total_budget * 0.45:
            break
        
        init_score, start = scored_starts[i]
        remaining_time = total_budget * 0.45 - (time.time() - start_time)
        if remaining_time < 3.0:
            continue
        
        try:
            # SA + L-BFGS-B + NM
            sa_pts, _ = _simulated_annealing(start, remaining_time * 0.3, rng)
            lbfgs_pts, _ = _l_bfgs_b_polish(sa_pts, remaining_time * 0.4)
            nm_pts, nm_score = _nelder_mead_polish(lbfgs_pts, remaining_time * 0.3)
            
            if nm_score > best_ratio:
                best_ratio = nm_score
                best_points = nm_pts.copy()
        except Exception:
            pass
    
    # Phase 2: Spectral optimization
    spectral_time_budget = total_budget * 0.5
    spectral_start = time.time()
    
    # Generate multiple spectral initializations
    n_spectral_starts = 8
    spectral_starts = []
    for i in range(n_spectral_starts):
        if time.time() - spectral_start > spectral_time_budget * 0.1:
            break
        try:
            params = _spectral_init(rng)
            spectral_starts.append(params)
        except Exception:
            pass
    
    # Optimize each spectral start
    per_spectral_time = spectral_time_budget * 0.9 / max(1, len(spectral_starts))
    
    for i, params in enumerate(spectral_starts):
        if time.time() - spectral_start > spectral_time_budget - 5.0:
            break
        
        alloc_time = min(per_spectral_time, spectral_time_budget - (time.time() - spectral_start) - 2.0)
        if alloc_time < 2.0:
            continue
        
        try:
            # Optimize spectral parameters
            pts, score = _spectral_optimize(params, alloc_time, beta=50.0)
            
            if pts is not None and score > best_ratio:
                best_ratio = score
                best_points = pts.copy()
            
            # Also try polishing the resulting points with traditional methods
            if pts is not None and time.time() - spectral_start < spectral_time_budget - 3.0:
                polish_time = min(alloc_time * 0.3, spectral_time_budget - (time.time() - spectral_start) - 1.0)
                if polish_time > 1.0:
                    polished_pts, polished_score = _nelder_mead_polish(pts, polish_time)
                    if polished_score > best_ratio:
                        best_ratio = polished_score
                        best_points = polished_pts.copy()
        except Exception:
            pass
    
    # Phase 3: Final refinement
    if best_points is not None and time.time() - start_time < total_budget - 5.0:
        final_time = total_budget - (time.time() - start_time) - 1.0
        if final_time > 3.0:
            try:
                final_pts, final_score = _nelder_mead_polish(best_points, final_time)
                if final_score > best_ratio:
                    best_ratio = final_score
                    best_points = final_pts.copy()
            except Exception:
                pass
    
    if best_points is None:
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points
# EVOLVE-BLOCK-END
