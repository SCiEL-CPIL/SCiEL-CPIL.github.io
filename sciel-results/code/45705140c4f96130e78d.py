import numpy as np
import time
import itertools
from scipy.optimize import minimize, differential_evolution

N = 16
D = 2
DEADLINE = 140.0  # Safety margin (150s limit)

def _pairwise_dists_sq(P):
    """Return vector of squared pairwise distances for upper triangle."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, 1)
    return dists_sq[iu]

def _score(P):
    """Calculate (max_dist / min_dist)^2."""
    d_sq = _pairwise_dists_sq(P)
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    if min_d_sq < 1e-12:
        return 1e10
    return max_d_sq / min_d_sq

def _soft_spread_objective(params, temp=1.0):
    """
    Smooth surrogate objective using centered points.
    Minimizes the soft spread of squared distances.
    params: flattened array of shape (N*2,).
    """
    P = params.reshape(N, 2)
    P = P - np.mean(P, axis=0)  # Center to remove translational invariance
    
    d_sq = _pairwise_dists_sq(P)
    d_sq = np.maximum(d_sq, 1e-12)
    
    # Soft max and soft min using log-sum-exp with temperature
    m = np.max(d_sq)
    exp_max = np.exp((d_sq - m) / temp)
    soft_max = m + temp * np.log(np.sum(exp_max))
    
    m_neg = np.max(-d_sq)
    exp_min = np.exp((-d_sq + m_neg) / temp)
    soft_min = -m_neg - temp * np.log(np.sum(exp_min))
    
    if soft_min < 1e-6:
        return 1e10
    
    return soft_max / soft_min

def _get_initial_candidates():
    """
    Generate diverse initial configurations.
    Returns list of (N, 2) arrays.
    """
    candidates = []
    
    # 1. Hexagonal Lattice Subset - evaluate all removal combinations
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    hex_points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 4.0 + 1e-9:
                hex_points.append(p)
    hex_points = np.array(hex_points)
    
    # Identify outer ring (dist^2 == 4)
    d_sqs = np.sum(hex_points**2, axis=1)
    outer_indices = np.where(np.isclose(d_sqs, 4.0))[0]
    
    # Evaluate all combinations of 3 removals from outer ring
    if len(outer_indices) >= 3:
        best_hex_score = float('inf')
        best_hex_pts = None
        for removal_combo in itertools.combinations(outer_indices, 3):
            removal_set = set(removal_combo)
            keep_idx = [i for i in range(len(hex_points)) if i not in removal_set]
            pts_hex = hex_points[keep_idx]
            s = _score(pts_hex)
            if s < best_hex_score:
                best_hex_score = s
                best_hex_pts = pts_hex
        if best_hex_pts is not None:
            candidates.append(best_hex_pts)
    
    # 2. 4x4 Grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)])
    candidates.append(grid)
    
    # 3. Two Concentric Circles (8 inner, 8 outer)
    pts_circ = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        pts_circ.append([2 * np.cos(ang), 2 * np.sin(ang)])
    for k in range(8):
        ang = 2 * np.pi * (k + 0.5) / 8
        pts_circ.append([1 * np.cos(ang), 1 * np.sin(ang)])
    candidates.append(np.array(pts_circ))
    
    # 4. Regular 16-gon
    pts_poly = []
    for k in range(16):
        ang = 2 * np.pi * k / 16
        pts_poly.append([np.cos(ang), np.sin(ang)])
    candidates.append(np.array(pts_poly))
    
    # 5. Two-ring circle: 6 inner at r=0.6, 10 outer at r=1.4
    pts_two_ring = []
    for k in range(6):
        ang = 2 * np.pi * k / 6
        pts_two_ring.append([0.6 * np.cos(ang), 0.6 * np.sin(ang)])
    for k in range(10):
        ang = 2 * np.pi * (k + 0.5) / 10
        pts_two_ring.append([1.4 * np.cos(ang), 1.4 * np.sin(ang)])
    candidates.append(np.array(pts_two_ring))
    
    # 6. Random Perturbed Grid
    rng = np.random.default_rng(42)
    perturbed = grid + rng.normal(0, 0.1, grid.shape)
    candidates.append(perturbed)
    
    return candidates

def _refine_candidate(init_pts, t0):
    """
    Refine a single candidate using annealed L-BFGS-B on the soft objective.
    Returns (best_pts, best_score).
    """
    best_pts = init_pts.copy()
    best_score = _score(init_pts)
    current_params = init_pts.flatten()
    
    # Annealing schedule: start wide, narrow down
    temps = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
    
    for temp in temps:
        if time.time() - t0 > DEADLINE - 5:
            break
            
        try:
            res = minimize(
                lambda p: _soft_spread_objective(p, temp=temp),
                current_params,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            new_pts = res.x.reshape(N, D)
            new_score = _score(new_pts)
            
            if new_score < best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                current_params = res.x
        except Exception:
            continue
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    
    # Phase 1: Generate initial candidates
    candidates = _get_initial_candidates()
    
    best_pts = None
    best_score = float('inf')
    
    # Phase 2: Global Search with Differential Evolution
    # Use a moderate temperature for DE to make the landscape smoother
    de_candidates = []
    if time.time() - start_time < DEADLINE - 30:
        try:
            bounds = [(-10, 10)] * (N * D)
            de_res = differential_evolution(
                _soft_spread_objective,
                bounds,
                args=(0.1,),
                maxiter=100,
                popsize=10,
                tol=1e-4,
                seed=42,
                polish=False
            )
            if de_res.success or de_res.x is not None:
                de_candidates.append(de_res.x.reshape(N, D))
                # Add perturbations of the DE result
                rng = np.random.default_rng(123)
                for _ in range(3):
                    perturbed = de_res.x.reshape(N, D) + rng.randn(N, D) * 0.1
                    de_candidates.append(perturbed)
        except Exception:
            pass
    
    candidates.extend(de_candidates)
    
    # Phase 3: Local Refinement on all candidates
    for i, init_pts in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        try:
            pts, score = _refine_candidate(init_pts, start_time)
            
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Phase 4: Final Polish with very fine tolerance
    if best_pts is not None and time.time() - start_time < DEADLINE - 2:
        try:
            res_final = minimize(
                lambda p: _soft_spread_objective(p, temp=0.0001),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            pts_final = res_final.x.reshape(N, D)
            score_final = _score(pts_final)
            if score_final < best_score:
                best_pts = pts_final
        except Exception:
            pass
    
    # Fallback
    if best_pts is None:
        best_pts = candidates[0]
        
    return best_pts
