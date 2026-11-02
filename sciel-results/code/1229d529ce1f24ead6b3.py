"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
DEADLINE = 140.0  # Safety margin before 150s kill

# Pre-compute all triples for fast vectorized scoring
# There are C(13, 3) = 286 triples
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)

_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)

def _compute_areas(pts):
    """Compute all triangle areas for a point set."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-12

def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    
    areas = _compute_areas(pts)
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area

def _soft_min_score(pts, beta=100.0):
    """
    Compute a smooth approximation of the min triangle area / hull area.
    Uses log-sum-exp for the min to make it differentiable.
    S = -1/beta * ln(sum(exp(-beta * A_i))) / H
    This is approx -min(A_i) / H for large beta.
    We want to maximize this, so we minimize the negative.
    """
    pts = np.asarray(pts, dtype=np.float64)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return -np.inf
        
    areas = _compute_areas(pts)
    
    # Stable logsumexp calculation for -min
    # log(sum(exp(-beta * A))) = max(-beta*A) + log(sum(exp(-beta*A - max(-beta*A))))
    vals = -beta * areas
    v_max = np.max(vals)
    if v_max == -np.inf:
        return -np.inf
        
    log_sum = v_max + np.log(np.sum(np.exp(vals - v_max)))
    soft_min_val = -log_sum / beta
    
    return soft_min_val / hull_area

def _neg_soft_score(params, beta):
    """Negative soft score for minimization."""
    pts = params.reshape(N, 2)
    return -_soft_min_score(pts, beta)

def _neg_hard_score(params):
    """Negative hard score for minimization."""
    pts = params.reshape(N, 2)
    return -_score_points(pts)

def _make_initial_guesses():
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p1 = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(p1.flatten())
    
    # 2. 1 center + 12 outer
    p2 = np.zeros((N, 2))
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    p2[1:, 0] = np.cos(angles)
    p2[1:, 1] = np.sin(angles)
    inits.append(p2.flatten())
    
    # 3. 1 center + 6 mid + 6 outer
    p3 = np.zeros((N, 2))
    angles_m = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    p3[1:7, 0] = 0.5 * np.cos(angles_m)
    p3[1:7, 1] = 0.5 * np.sin(angles_m)
    angles_o = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    p3[7:, 0] = np.cos(angles_o)
    p3[7:, 1] = np.sin(angles_o)
    inits.append(p3.flatten())
    
    # 4. Random
    rng = np.random.RandomState(42)
    p4 = rng.uniform(-1, 1, size=(N, 2)).flatten()
    inits.append(p4)
    
    # 5. 1 center + 5 inner + 7 outer (asymmetric layers)
    p5 = np.zeros((N, 2))
    angles_in = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    p5[1:6, 0] = 0.4 * np.cos(angles_in)
    p5[1:6, 1] = 0.4 * np.sin(angles_in)
    angles_out = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    p5[6:, 0] = np.cos(angles_out)
    p5[6:, 1] = np.sin(angles_out)
    inits.append(p5.flatten())
    
    # 6. 2 center-ish + 11 outer
    p6 = np.zeros((N, 2))
    p6[0] = [0.2, 0]
    p6[1] = [-0.2, 0]
    angles = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    p6[2:, 0] = np.cos(angles)
    p6[2:, 1] = np.sin(angles)
    inits.append(p6.flatten())
    
    return inits

def construct():
    t_start = time.time()
    
    best_params = None
    best_score = -np.inf
    
    # 1. Evaluate initial guesses
    inits = _make_initial_guesses()
    for p in inits:
        pts = p.reshape(N, 2)
        s = _score_points(pts)
        if s > best_score:
            best_score = s
            best_params = p.copy()
            
    # 2. Simulated Annealing with Local Refinement
    # We use a smooth surrogate for the SA steps and local optimization.
    # The hard score is used for final evaluation and acceptance of "best".
    
    # SA Parameters
    T0 = 0.05  # Initial temperature
    T_end = 1e-5
    cooling_rate = 0.995
    max_iters = 20000  # We will break early based on time
    
    current_params = best_params.copy()
    current_soft_score = _soft_min_score(current_params.reshape(N, 2), beta=50.0)
    
    # We will increase beta over time to sharpen the objective
    # Start with low beta (smooth) to find good basin, increase to refine.
    
    rng = np.random.RandomState(123)
    
    # Adaptive step size
    step_size = 0.1
    
    # We will use a fixed beta for the main SA loop to ensure smoothness,
    # then do a separate refinement phase with higher beta.
    sa_beta = 50.0
    
    for i in range(max_iters):
        # Time check
        if (time.time() - t_start) > (DEADLINE - 20.0):
            break
            
        # Update temperature
        T = T0 * (cooling_rate ** i)
        
        # Generate neighbor
        # Gaussian perturbation
        perturbation = rng.normal(0, step_size, size=(2 * N))
        neighbor_params = current_params + perturbation
        neighbor_params = np.clip(neighbor_params, -1.0, 1.0)
        
        neighbor_pts = neighbor_params.reshape(N, 2)
        neighbor_soft_score = _soft_min_score(neighbor_pts, sa_beta)
        
        # Acceptance criterion
        if neighbor_soft_score > current_soft_score:
            accept = True
        else:
            delta = current_soft_score - neighbor_soft_score
            if delta < 1e-12:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta / T)
                
        if accept:
            current_params = neighbor_params
            current_soft_score = neighbor_soft_score
            
            # Occasionally perform a local refinement on the soft score
            # This helps climb the smooth hills efficiently
            if i % 100 == 0 and i > 0:
                try:
                    res = minimize(
                        lambda x: _neg_soft_score(x, sa_beta),
                        current_params,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                    if res.fun < -current_soft_score:
                        current_params = res.x
                        current_soft_score = -res.fun
                except Exception:
                    pass
            
            # Check hard score periodically
            if i % 50 == 0:
                hard_score = _score_points(current_params.reshape(N, 2))
                if hard_score > best_score:
                    best_score = hard_score
                    best_params = current_params.copy()
        else:
            # If rejected, slightly reduce step size if stuck? 
            # Or just keep same step size.
            pass
            
        # Adaptive step size: if we haven't improved best_score in a while, increase step size
        # This is a simple heuristic.
        if i % 500 == 0 and i > 0:
             # If stuck, expand search
             if best_score < _score_points(current_params.reshape(N,2)) + 1e-6:
                 step_size *= 1.2
             else:
                 step_size *= 0.9
             step_size = np.clip(step_size, 0.01, 0.5)

    # 3. Multi-stage Refinement Phase
    # Start from the best SA result and other good inits
    # Use increasing beta for the surrogate to sharpen the min-area constraint
    
    refinement_starts = [best_params.copy()]
    for p in inits[:3]: # Add a few diverse inits
        refinement_starts.append(p.copy())
        
    # Deduplicate starts
    unique_starts = []
    for s in refinement_starts:
        is_unique = True
        for u in unique_starts:
            if np.allclose(s, u, atol=1e-3):
                is_unique = False
                break
        if is_unique:
            unique_starts.append(s)

    # Stage 1: Moderate beta (50)
    for start in unique_starts:
        if (time.time() - t_start) > (DEADLINE - 10.0):
            break
        try:
            res = minimize(
                lambda x: _neg_soft_score(x, 50.0),
                start,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 300, 'maxfun': 5000}
            )
            score = _score_points(res.x.reshape(N, 2))
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass

    # Stage 2: High beta (200)
    for start in unique_starts[:2]: # Only refine the best ones further
        if (time.time() - t_start) > (DEADLINE - 5.0):
            break
        try:
            res = minimize(
                lambda x: _neg_soft_score(x, 200.0),
                start,
                method='L-BFGS-B',
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 300, 'maxfun': 5000}
            )
            score = _score_points(res.x.reshape(N, 2))
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass
            
    # Stage 3: Final Hard Score Polish
    if (time.time() - t_start) < (DEADLINE - 2.0):
        try:
            res = minimize(
                _neg_hard_score,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            score = _score_points(res.x.reshape(N, 2))
            if score > best_score:
                best_score = score
                best_params = res.x.copy()
        except Exception:
            pass

    return best_params.reshape(N, 2)
