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
    
    # Avoid log(0) or overflow
    # Shift areas to ensure numerical stability? 
    # The areas are typically small (O(1/100) or less).
    # If beta is large, exp(-beta * A) can underflow for large A, which is fine (becomes 0).
    # But if A is very small, exp(-beta*A) is close to 1.
    
    # To avoid underflow issues with very large beta * A, we can scale.
    # However, for N=13, areas are not huge. 
    # Let's use a stable logsumexp calculation.
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
    # 6 mid
    angles_m = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    p3[1:7, 0] = 0.5 * np.cos(angles_m)
    p3[1:7, 1] = 0.5 * np.sin(angles_m)
    # 6 outer
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
    
    # 7. Symmetric Ring Configuration (1-6-6) with optimized radii
    # This is a strong candidate for the optimal structure.
    # We try a few variations of radii.
    for r_mid in [0.4, 0.5, 0.6, 0.7]:
        p7 = np.zeros((N, 2))
        angles_m = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        p7[1:7, 0] = r_mid * np.cos(angles_m)
        p7[1:7, 1] = r_mid * np.sin(angles_m)
        # Outer points at radius 1, offset by pi/6
        angles_o = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
        p7[7:, 0] = np.cos(angles_o)
        p7[7:, 1] = np.sin(angles_o)
        inits.append(p7.flatten())
        
    # 8. Symmetric Ring Configuration (1-5-7)
    for r_mid in [0.4, 0.5, 0.6]:
        p8 = np.zeros((N, 2))
        angles_m = np.linspace(0, 2 * np.pi, 5, endpoint=False)
        p8[1:6, 0] = r_mid * np.cos(angles_m)
        p8[1:6, 1] = r_mid * np.sin(angles_m)
        angles_o = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        p8[6:, 0] = np.cos(angles_o)
        p8[6:, 1] = np.sin(angles_o)
        inits.append(p8.flatten())

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
    
    stagnation_counter = 0
    last_best_score = best_score
    
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
                    if res.fun < _neg_soft_score(current_params, sa_beta):
                        current_params = res.x
                        current_soft_score = _soft_min_score(current_params.reshape(N, 2), sa_beta)
                except Exception:
                    pass
            
            # Update global best if this is better in terms of HARD score
            # We don't want to waste time calculating hard score every step.
            # But we need to track the best hard score.
            if i % 20 == 0:
                hard_s = _score_points(current_params.reshape(N, 2))
                if hard_s > best_score:
                    best_score = hard_s
                    best_params = current_params.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 20
                    
        else:
            pass
            
        # Adaptive step size: if we are accepting a lot, increase step?
        # If we are rejecting a lot, decrease step?
        # Simple heuristic:
        # If T is high, step can be larger.
        step_size = 0.05 + 0.05 * (T / T0)
        
        # Stagnation breaker: if no improvement in hard score for a while, jump
        if stagnation_counter > 200:
            # Large random jump
            jump = rng.normal(0, 0.5, size=(2 * N))
            current_params = np.clip(current_params + jump, -1.0, 1.0)
            current_soft_score = _soft_min_score(current_params.reshape(N, 2), sa_beta)
            stagnation_counter = 0

    # 3. Final Local Refinement on Hard Score
    # Use the best soft-scored point as a starting point for hard score optimization.
    # Also try the current best_params from hard score tracking.
    
    starts_to_refine = [best_params]
    
    # Add current SA state
    if not np.allclose(best_params, current_params):
        starts_to_refine.append(current_params)
        
    # Add initial bests
    for p in inits:
        starts_to_refine.append(p)
        
    # Deduplicate
    unique_starts = []
    for s in starts_to_refine:
        is_unique = True
        for u in unique_starts:
            if np.allclose(s, u, atol=1e-3):
                is_unique = False
                break
        if is_unique:
            unique_starts.append(s)
            
    # Refinement Phase 1: Medium Beta
    # Refinement Phase 2: High Beta
    # Refinement Phase 3: Very High Beta (Polish)
    
    betas = [100.0, 300.0, 1000.0, 3000.0]
    
    for start in unique_starts:
        if (time.time() - t_start) > (DEADLINE - 10.0):
            break
            
        current_refine_params = start.copy()
        
        for beta in betas:
            if (time.time() - t_start) > (DEADLINE - 5.0):
                break
                
            try:
                # Use L-BFGS-B for better convergence on smooth functions
                res = minimize(
                    lambda x: _neg_soft_score(x, beta),
                    current_refine_params,
                    method='L-BFGS-B',
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-14}
                )
                if res.success or res.fun < _neg_soft_score(current_refine_params, beta):
                    current_refine_params = res.x
                    
                # Evaluate hard score
                score2 = _score_points(current_refine_params.reshape(N, 2))
                if score2 > best_score:
                    best_score = score2
                    best_params = current_refine_params.copy()
                    
            except Exception:
                pass

    return best_params.reshape(N, 2)
