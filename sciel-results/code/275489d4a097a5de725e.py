"""Heilbronn convex regions n=14: maximize min triangle area / hull area."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _all_triangle_areas(points):
    """Compute all triangle areas vectorized. Returns array of shape (C(n,3),)."""
    n = len(points)
    i_idx, j_idx, k_idx = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_idx.append(i)
                j_idx.append(j)
                k_idx.append(k)
    i_idx = np.array(i_idx)
    j_idx = np.array(j_idx)
    k_idx = np.array(k_idx)
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    areas = _all_triangle_areas(points)
    min_tri = np.min(areas)
    return min_tri / hull_area

def _soft_score(points, temperature=0.01):
    """Smooth surrogate using log-sum-exp soft-min."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-12:
        return 0.0
    areas = _all_triangle_areas(points)
    max_a = np.max(areas)
    shifted = -areas / temperature
    shifted -= np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = -temperature * (log_sum + np.max(shifted))
    return soft_min / hull_area

def _regular_polygon(n, radius=1.0):
    """Generate points on a regular polygon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _two_rings(n, inner_frac=0.5):
    """Generate points on two concentric rings."""
    n_outer = n // 2
    n_inner = n - n_outer
    
    outer_angles = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    inner_angles = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    inner_pts = inner_frac * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    return np.vstack([outer_pts, inner_pts])

def _hexagonal_packing(n):
    """Generate hexagonal packing arrangement."""
    points = []
    points.append([0, 0])
    for i in range(6):
        angle = i * np.pi / 3
        points.append([np.cos(angle), np.sin(angle)])
    for i in range(7):
        angle = i * 2 * np.pi / 7 + np.pi / 7
        points.append([2 * np.cos(angle), 2 * np.sin(angle)])
    return np.array(points[:n])

def _optimize_nelder_mead(points, deadline, use_soft=False, temperature=0.01):
    """Optimize using Nelder-Mead with time management."""
    if use_soft:
        def objective(x):
            pts = x.reshape(-1, 2)
            return -_soft_score(pts, temperature)
    else:
        def objective(x):
            pts = x.reshape(-1, 2)
            return -_score(pts)
    
    x0 = points.flatten()
    
    if time.time() > deadline - 5:
        return points, -objective(x0)
    
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={
            'maxiter': 5000,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True
        }
    )
    
    best_points = result.x.reshape(-1, 2)
    best_score = -result.fun
    
    return best_points, best_score

def _optimize_local_fine(points, deadline, max_iters=2000, step_size=0.005):
    """Fine-tuning local optimization with small steps."""
    points = points.copy()
    current_score = _score(points)
    best_points = points.copy()
    best_score = current_score
    rng = np.random.RandomState(123)
    
    for it in range(max_iters):
        if time.time() > deadline - 2:
            break
        
        idx = rng.randint(len(points))
        delta = rng.normal(0, step_size, 2)
        new_points = points.copy()
        new_points[idx] += delta
        
        new_score = _score(new_points)
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            if it % 200 == 0:
                step_size *= 0.9
    
    return best_points, best_score

def _epigraph_objective(x, t):
    """
    Epigraph formulation: minimize -t subject to all triangle areas >= t * hull_area.
    We reformulate: minimize -(t / hull_area) subject to areas >= t.
    Actually, let's use a penalty approach for the epigraph.
    """
    pts = x[:-1].reshape(-1, 2)
    t = x[-1]
    
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-12:
        return 1e10
    
    areas = _all_triangle_areas(pts)
    
    # Penalty for violating areas >= t
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations**2)
    
    # Objective: maximize t / hull_area, so minimize -(t / hull_area) + penalty
    return -(t / hull_area) + 100.0 * penalty

def _optimize_epigraph(points, deadline):
    """Optimize using epigraph formulation with SLSQP or trust-constr."""
    pts = points.copy()
    initial_score = _score(pts)
    # Initial t: set to min triangle area
    areas = _all_triangle_areas(pts)
    t0 = np.min(areas)
    
    x0 = np.concatenate([pts.flatten(), [t0]])
    
    def objective(x):
        return _epigraph_objective(x, x[-1])
    
    if time.time() > deadline - 10:
        return pts, initial_score
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        best_pts = result.x[:-1].reshape(-1, 2)
        best_score = _score(best_pts)
        
        if best_score > initial_score:
            return best_pts, best_score
        else:
            return pts, initial_score
    except Exception:
        return pts, initial_score

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    # Generate diverse initial candidates
    candidates = []
    
    # Regular polygon (best known starting point)
    candidates.append(_regular_polygon(N))
    
    # Two rings with different ratios
    for frac in [0.4, 0.5, 0.6, 0.7]:
        candidates.append(_two_rings(N, frac))
    
    # Hexagonal packing
    candidates.append(_hexagonal_packing(N))
    
    # Slightly perturbed regular polygons
    rng = np.random.RandomState(42)
    for _ in range(5):
        base = _regular_polygon(N)
        perturbed = base + rng.normal(0, 0.08, (N, 2))
        candidates.append(perturbed)
    
    # Evaluate initial candidates and find the best
    best_points = None
    best_score = -1
    scored_candidates = []
    
    for pts in candidates:
        if time.time() > deadline - 15:
            break
        score = _score(pts)
        scored_candidates.append((score, pts))
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    if best_points is None:
        return _regular_polygon(N)
    
    # Sort candidates by score descending
    scored_candidates.sort(key=lambda x: -x[0])
    
    # Phase 1: Nelder-Mead optimization with soft objective (smoother landscape)
    if time.time() < deadline - 25:
        nm_points, nm_score = _optimize_nelder_mead(
            best_points, deadline, use_soft=True, temperature=0.05
        )
        if nm_score > best_score:
            best_score = nm_score
            best_points = nm_points
    
    # Phase 2: Nelder-Mead with exact objective (refine)
    if time.time() < deadline - 20:
        nm_points, nm_score = _optimize_nelder_mead(
            best_points, deadline, use_soft=False
        )
        if nm_score > best_score:
            best_score = nm_score
            best_points = nm_points
    
    # Phase 3: Try epigraph formulation
    if time.time() < deadline - 20:
        epi_points, epi_score = _optimize_epigraph(best_points, deadline)
        if epi_score > best_score:
            best_score = epi_score
            best_points = epi_points
    
    # Phase 4: Try Nelder-Mead from second-best initial candidate
    if time.time() < deadline - 20:
        for score, pts in scored_candidates[1:]:
            if time.time() > deadline - 20:
                break
            nm_points, nm_score = _optimize_nelder_mead(
                pts, deadline, use_soft=True, temperature=0.05
            )
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_points
                # Refine with exact objective
                nm_points2, nm_score2 = _optimize_nelder_mead(
                    best_points, deadline, use_soft=False
                )
                if nm_score2 > best_score:
                    best_score = nm_score2
                    best_points = nm_points2
    
    # Phase 5: Fine-tuning with small random steps
    if time.time() < deadline - 8:
        fine_points, fine_score = _optimize_local_fine(
            best_points, deadline, max_iters=5000, step_size=0.003
        )
        if fine_score > best_score:
            best_score = fine_score
            best_points = fine_points
    
    # Phase 6: Final Nelder-Mead polish with tighter tolerance
    if time.time() < deadline - 5:
        def objective(x):
            pts = x.reshape(-1, 2)
            return -_score(pts)
        
        x0 = best_points.flatten()
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': 3000,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
        )
        polish_points = result.x.reshape(-1, 2)
        polish_score = -result.fun
        if polish_score > best_score:
            best_score = polish_score
            best_points = polish_points
    
    # Phase 7: Another fine-tuning pass
    if time.time() < deadline - 3:
        fine_points, fine_score = _optimize_local_fine(
            best_points, deadline, max_iters=3000, step_size=0.001
        )
        if fine_score > best_score:
            best_score = fine_score
            best_points = fine_points
    
    return best_points if best_points is not None else _regular_polygon(N)
# EVOLVE-BLOCK-END
