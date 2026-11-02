import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
DEADLINE = 145.0  # Safety margin before 150s kill

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

def _neg_score(params):
    """Negative score for minimization."""
    pts = params.reshape(N, 2)
    return -_score_points(pts)

def _soft_min_score(params, beta=50.0):
    """
    Compute a smooth approximation of the min triangle area / hull area.
    Uses log-sum-exp for the min to make it differentiable.
    S = -1/beta * ln(sum(exp(-beta * A_i))) / H
    This is approx -min(A_i) / H for large beta.
    We want to maximize this, so we minimize the negative.
    """
    pts = params.reshape(N, 2)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 1e10
        
    areas = _compute_areas(pts)
    
    # Avoid log(0) or overflow
    # Shift areas to ensure numerical stability? 
    # The areas are typically small, so we can work directly
    # But exp(-beta * A) can underflow if beta*A is large
    # Let's use a shifted version: let m = min(areas)
    # Then soft_min = m - 1/beta * ln(sum(exp(-beta*(A_i - m))))
    
    m = np.min(areas)
    if m < 1e-15:
        return 1e10
    
    # Compute log-sum-exp in a stable way
    # sum(exp(-beta * (A_i - m))) = sum(exp(-beta * A_i + beta * m))
    # = exp(beta * m) * sum(exp(-beta * A_i))
    # So ln(sum(exp(-beta * (A_i - m)))) = beta * m + ln(sum(exp(-beta * A_i)))
    # This doesn't help directly.
    
    # Let's just use: soft_min = -1/beta * ln(sum(exp(-beta * A_i / H)))
    # This is the soft min of the normalized areas.
    
    norm_areas = areas / hull_area
    # To avoid underflow, subtract the max exponent (which is -beta * min(norm_areas))
    # max_exponent = -beta * np.min(norm_areas)
    # sum_exp = np.sum(np.exp(-beta * norm_areas - max_exponent)) * np.exp(max_exponent)
    # log_sum_exp = max_exponent + np.log(np.sum(np.exp(-beta * norm_areas - max_exponent)))
    # soft_min = -1/beta * log_sum_exp
    
    # Simpler: since we want to minimize -soft_min, and soft_min approx min(norm_areas)
    # Let's just compute -1/beta * log(sum(exp(-beta * norm_areas)))
    # But if norm_areas are small, exp(-beta * norm_areas) is close to 1.
    # sum(exp(-beta * norm_areas)) is approx N_triples * exp(-beta * avg(norm_areas))
    
    # Let's try a different approach: minimize the sum of powers
    # sum(A_i^p) is approximated by min(A_i)^p * N for large p
    # So minimizing sum(A_i^p) is like minimizing min(A_i)
    # Let's use p = 10 or so.
    
    # Actually, the log-sum-exp trick is standard.
    # Let x_i = norm_areas
    # soft_min = -1/beta * log(sum(exp(-beta * x_i)))
    # Let M = min(x_i)
    # sum(exp(-beta * x_i)) = exp(-beta * M) * sum(exp(-beta * (x_i - M)))
    # log(sum(exp(-beta * x_i))) = -beta * M + log(sum(exp(-beta * (x_i - M))))
    # soft_min = -1/beta * (-beta * M + log(sum(exp(-beta * (x_i - M)))))
    # soft_min = M - 1/beta * log(sum(exp(-beta * (x_i - M))))
    
    x = norm_areas
    M = np.min(x)
    # x - M >= 0
    # exp(-beta * (x - M)) <= 1
    # sum(exp(-beta * (x - M))) is between 1 and N
    # log(sum(...)) is between 0 and log(N)
    # soft_min = M - 1/beta * log(sum(exp(-beta * (x - M))))
    
    # This is stable.
    shifted = x - M
    # If beta * shifted is large, exp underflows to 0, which is fine.
    exp_terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = M - (1.0 / beta) * log_sum
    
    # We want to minimize -soft_min_val
    return -soft_min_val

def _make_symmetric_inits():
    """
    Generate diverse initial configurations based on symmetry.
    Optimal Heilbronn configurations often have points on concentric regular polygons.
    """
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p1 = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(p1.flatten())
    
    # 2. 1 center + 12 outer (Regular 12-gon)
    p2 = np.zeros((N, 2))
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    p2[1:, 0] = np.cos(angles)
    p2[1:, 1] = np.sin(angles)
    inits.append(p2.flatten())
    
    # 3. 1 center + 4 inner (Square) + 8 outer (Octagon)
    p3 = np.zeros((N, 2))
    # 4 inner
    angles_i = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    p3[1:5, 0] = 0.5 * np.cos(angles_i)
    p3[1:5, 1] = 0.5 * np.sin(angles_i)
    # 8 outer
    angles_o = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    p3[5:, 0] = np.cos(angles_o)
    p3[5:, 1] = np.sin(angles_o)
    inits.append(p3.flatten())
    
    # 4. 1 center + 6 inner (Hexagon) + 6 outer (Hexagon)
    p4 = np.zeros((N, 2))
    # 6 inner
    angles_i = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    p4[1:7, 0] = 0.5 * np.cos(angles_i)
    p4[1:7, 1] = 0.5 * np.sin(angles_i)
    # 6 outer
    angles_o = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    p4[7:, 0] = np.cos(angles_o)
    p4[7:, 1] = np.sin(angles_o)
    inits.append(p4.flatten())
    
    # 5. 1 center + 3 inner (Triangle) + 9 outer (Nonagon)
    p5 = np.zeros((N, 2))
    angles_i = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    p5[1:4, 0] = 0.5 * np.cos(angles_i)
    p5[1:4, 1] = 0.5 * np.sin(angles_i)
    angles_o = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    p5[4:, 0] = np.cos(angles_o)
    p5[4:, 1] = np.sin(angles_o)
    inits.append(p5.flatten())
    
    # 6. Random uniform
    rng = np.random.RandomState(42)
    p6 = rng.uniform(-1, 1, size=(N, 2)).flatten()
    inits.append(p6)
    
    # 7. Random symmetric perturbation of regular 13-gon
    p7 = p1.copy()
    p7 += rng.normal(0, 0.1, size=p7.shape)
    inits.append(p7.flatten())
    
    # 8. Random symmetric perturbation of 1+12
    p8 = p2.copy()
    p8 += rng.normal(0, 0.1, size=p8.shape)
    inits.append(p8.flatten())
    
    # 9. Fermat spiral
    p9 = np.zeros((N, 2))
    for i in range(N):
        r = np.sqrt(i + 1) / np.sqrt(N)
        theta = i * np.pi * (3 - np.sqrt(5))
        p9[i, 0] = r * np.cos(theta)
        p9[i, 1] = r * np.sin(theta)
    inits.append(p9.flatten())
    
    return inits

def construct():
    t_start = time.time()
    
    best_params = None
    best_score = -np.inf
    
    # 1. Evaluate initial guesses
    inits = _make_symmetric_inits()
    for p in inits:
        pts = p.reshape(N, 2)
        s = _score_points(pts)
        if s > best_score:
            best_score = s
            best_params = p.copy()
            
    # 2. Multi-start Nelder-Mead optimization
    # We will try to refine each initial guess using Nelder-Mead.
    # Nelder-Mead is derivative-free and handles non-differentiable functions well.
    
    # Sort inits by score to try the best ones first
    init_scores = []
    for p in inits:
        s = _score_points(p.reshape(N, 2))
        init_scores.append((s, p))
    init_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Use a mix of hard score and soft score for optimization
    # Start with hard score (Nelder-Mead) for the best few
    for s, p in init_scores:
        if (time.time() - t_start) > (DEADLINE - 10.0):
            break
            
        # Run Nelder-Mead from this start
        try:
            res = minimize(
                _neg_score,
                p,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            # Evaluate the result
            res_score = _score_points(res.x.reshape(N, 2))
            if res_score > best_score:
                best_score = res_score
                best_params = res.x.copy()
                
        except Exception:
            pass
            
    # 3. Refine with soft score using BFGS or L-BFGS-B
    # The soft score is differentiable, so gradient-based methods can work better
    if best_params is not None:
        if (time.time() - t_start) < (DEADLINE - 5.0):
            try:
                # Use a moderate beta for the soft min
                res = minimize(
                    _soft_min_score,
                    best_params,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'gtol': 1e-8, 'maxfun': 5000}
                )
                
                res_score = _score_points(res.x.reshape(N, 2))
                if res_score > best_score:
                    best_score = res_score
                    best_params = res.x.copy()
                    
            except Exception:
                pass
    
    # 4. Final local refinement with higher precision
    # Try to polish the best result found so far
    if best_params is not None:
        if (time.time() - t_start) < (DEADLINE - 2.0):
            try:
                res = minimize(
                    _neg_score,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                res_score = _score_points(res.x.reshape(N, 2))
                if res_score > best_score:
                    best_score = res_score
                    best_params = res.x.copy()
            except Exception:
                pass
    
    return best_params.reshape(N, 2)
