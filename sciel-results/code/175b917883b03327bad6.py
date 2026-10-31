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

def _epigraph_objective(params, t, temp=0.01):
    """
    Epigraph-style smooth objective for min/max ratio.
    Minimizes (soft_max_d2 - t * soft_min_d2) where t is a variable.
    This encourages all distances to be close to each other.
    """
    P = params.reshape(N, 2)
    P = P - np.mean(P, axis=0)
    
    d_sq = _pairwise_dists_sq(P)
    d_sq = np.maximum(d_sq, 1e-12)
    
    # Soft max of d_sq
    m = np.max(d_sq)
    exp_max = np.exp((d_sq - m) / temp)
    soft_max = m + temp * np.log(np.sum(exp_max))
    
    # Soft min of d_sq
    m_neg = np.max(-d_sq)
    exp_min = np.exp((-d_sq + m_neg) / temp)
    soft_min = -m_neg - temp * np.log(np.sum(exp_min))
    
    if soft_min < 1e-6:
        return 1e10
    
    # We want to minimize (max/min)^2 = max^2/min^2
    # Equivalent to minimizing (max - sqrt(t)*min)^2 + penalty for t deviation
    # Better: minimize soft_max/soft_min directly with tighter temp
    return soft_max / soft_min

def _get_lattice_subset():
    """
    Get the best 16-point subset from triangular lattice.
    Uses systematic search over removal combinations.
    """
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    
    # Generate lattice points within radius 2.5
    hex_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            if np.sum(p**2) <= 6.25 + 1e-9:  # radius 2.5
                hex_points.append(p)
    hex_points = np.array(hex_points)
    
    # We need to choose 16 points from this set
    # Strategy: start with all points, remove worst ones iteratively
    if len(hex_points) <= 16:
        return hex_points
    
    # Use greedy removal: repeatedly remove the point whose removal
    # most improves the score
    current_pts = hex_points.copy()
    
    for _ in range(len(current_pts) - 16):
        best_score = float('inf')
        best_remove_idx = -1
        
        for idx in range(len(current_pts)):
            # Try removing this point
            remaining = np.delete(current_pts, idx, axis=0)
            if len(remaining) != 16:
                continue
            s = _score(remaining)
            if s < best_score:
                best_score = s
                best_remove_idx = idx
        
        if best_remove_idx >= 0:
            current_pts = np.delete(current_pts, best_remove_idx, axis=0)
    
    return current_pts

def _get_initial_candidates():
    """
    Generate diverse initial configurations.
    Returns list of (N, 2) arrays.
    """
    candidates = []
    
    # 1. Best triangular lattice subset
    lattice_pts = _get_lattice_subset()
    candidates.append(lattice_pts)
    
    # 2. Perturbed lattice subsets for diversity
    rng = np.random.default_rng(42)
    for noise_scale in [0.01, 0.05, 0.1]:
        perturbed = lattice_pts + rng.normal(0, noise_scale, lattice_pts.shape)
        candidates.append(perturbed)
    
    # 3. 4x4 Grid
    grid = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)])
    candidates.append(grid)
    
    # 4. Two Concentric Circles (8 inner, 8 outer)
    pts_circ = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        pts_circ.append([2 * np.cos(ang), 2 * np.sin(ang)])
    for k in range(8):
        ang = 2 * np.pi * (k + 0.5) / 8
        pts_circ.append([1 * np.cos(ang), 1 * np.sin(ang)])
    candidates.append(np.array(pts_circ))
    
    # 5. Regular 16-gon
    pts_poly = []
    for k in range(16):
        ang = 2 * np.pi * k / 16
        pts_poly.append([np.cos(ang), np.sin(ang)])
    candidates.append(np.array(pts_poly))
    
    # 6. Two-ring circle: 6 inner at r=0.6, 10 outer at r=1.4
    pts_two_ring = []
    for k in range(6):
        ang = 2 * np.pi * k / 6
        pts_two_ring.append([0.6 * np.cos(ang), 0.6 * np.sin(ang)])
    for k in range(10):
        ang = 2 * np.pi * (k + 0.5) / 10
        pts_two_ring.append([1.4 * np.cos(ang), 1.4 * np.sin(ang)])
    candidates.append(np.array(pts_two_ring))
    
    return candidates

def _refine_candidate(init_pts, t0):
    """
    Refine a single candidate using annealed L-BFGS-B on the soft objective.
    Returns (best_pts, best_score).
    """
    best_pts = init_pts.copy()
    best_score = _score(init_pts)
    current_params = init_pts.flatten()
    
    # Aggressive annealing schedule
    temps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for temp in temps:
        if time.time() - t0 > DEADLINE - 5:
            break
            
        try:
            res = minimize(
                lambda p: _epigraph_objective(p, None, temp=temp),
                current_params,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12}
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

def _final_polish(pts, t0):
    """
    Very tight final optimization with multiple restarts from tiny perturbations.
    """
    best_pts = pts.copy()
    best_score = _score(pts)
    current_params = pts.flatten()
    
    # Multiple restarts with tiny perturbations
    rng = np.random.default_rng(999)
    for restart in range(5):
        if time.time() - t0 > DEADLINE - 2:
            break
        
        if restart > 0:
            current_params = best_pts.flatten() + rng.normal(0, 0.001, (N, D)).flatten()
        
        try:
            res = minimize(
                lambda p: _epigraph_objective(p, None, temp=0.0001),
                current_params,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            
            new_pts = res.x.reshape(N, D)
            new_score = _score(new_pts)
            
            if new_score < best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        except Exception:
            continue
    
    return best_pts

def construct():
    start_time = time.time()
    
    # Phase 1: Generate initial candidates
    candidates = _get_initial_candidates()
    
    best_pts = None
    best_score = float('inf')
    
    # Phase 2: Global Search with Differential Evolution (optional, time permitting)
    if time.time() - start_time < DEADLINE - 40:
        try:
            bounds = [(-5, 5)] * (N * D)
            de_res = differential_evolution(
                lambda p: _epigraph_objective(p, None, temp=0.05),
                bounds,
                maxiter=50,
                popsize=8,
                tol=1e-3,
                seed=42,
                polish=False
            )
            if de_res.x is not None:
                candidates.append(de_res.x.reshape(N, D))
        except Exception:
            pass
    
    # Phase 3: Local Refinement on all candidates
    for i, init_pts in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 15:
            break
            
        try:
            pts, score = _refine_candidate(init_pts, start_time)
            
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
        except Exception:
            continue
    
    # Phase 4: Final Polish with very tight tolerances
    if best_pts is not None and time.time() - start_time < DEADLINE - 3:
        try:
            best_pts = _final_polish(best_pts, start_time)
        except Exception:
            pass
    
    # Fallback
    if best_pts is None:
        best_pts = candidates[0]
        
    return best_pts
