import numpy as np
import time
import warnings
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

warnings.filterwarnings('ignore')

N = 13

# Pre-compute indices for all triples (i, j, k) with i < j < k
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    """
    Calculate areas of all triangles formed by the points.
    points: (N, 2) array
    returns: (286,) array of areas
    """
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        if len(points) < 3:
            return 0.0
        # Check for degeneracy
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / hull area.
    """
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _neg_score(points):
    """
    Negative score for minimization.
    """
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

# ---------- Radial Coordinate Helpers ----------

def _radial_to_cartesian(params):
    """
    Convert [r1,...,rN, theta1,...,thetaN] to (N,2) Cartesian points.
    r is clipped to [1e-6, 1.0] to ensure points are well-behaved and hull is non-degenerate.
    """
    r = np.clip(params[:N], 1e-6, 1.0)
    theta = params[N:]
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

def _cartesian_to_radial(points):
    """
    Convert (N,2) Cartesian to [r1,...,rN, theta1,...,thetaN].
    """
    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    theta = np.arctan2(points[:, 1], points[:, 0]) % (2 * np.pi)
    return np.concatenate([r, theta])

# ---------- Penalty-based Formulation ----------
# Objective: Maximize t / hull_area subject to area_i >= t for all i.
# We minimize: -t / hull_area + mu * sum(max(0, t - area_i)^2)
# x = [t, r1,...,rN, theta1,...,thetaN]

def _penalty_objective_radial(x, mu):
    """
    Penalty objective in radial coordinates.
    """
    t = x[0]
    r = np.clip(x[1:1+N], 1e-6, 1.0)
    theta = x[1+N:1+2*N]
    
    # Convert to Cartesian
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    # Calculate areas and hull
    areas = _triangle_areas_vectorized(pts)
    hull_a = _hull_area(pts)
    
    if hull_a < 1e-12:
        return 1e10
        
    # Penalty for violating area_i >= t
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    
    # We want to maximize t/hull_a, so we minimize -t/hull_a
    # But t is absolute area. To make it scale-invariant in the objective landscape,
    # we effectively push t up as much as possible.
    # The penalty term ensures t doesn't exceed the min area.
    # As mu increases, t approaches min_area.
    # The term -t/hull_a then becomes -min_area/hull_a = -score.
    
    return -t / hull_a + mu * penalty

def _penalty_optimize_radial(points, deadline, mu_start=10.0, mu_end=1e8, n_mu_steps=12,
                              max_iter_per_mu=150):
    """
    Penalty-based optimization in radial coordinates.
    Returns best points and best score.
    """
    best = points.copy()
    best_s = _score(best)
    
    # Initial t: slightly below current min area
    areas = _triangle_areas_vectorized(best)
    t_init = float(np.min(areas)) * 0.85
    
    # Convert to radial
    r0 = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    theta0 = np.arctan2(points[:, 1], points[:, 0]) % (2 * np.pi)
    r0 = np.clip(r0, 1e-6, 1.0)
    
    x0 = np.zeros(1 + 2 * N)
    x0[0] = t_init
    x0[1:1+N] = r0
    x0[1+N:] = theta0
    
    # Geometric progression of penalty multipliers
    mus = np.geomspace(mu_start, mu_end, n_mu_steps)
    
    for mu in mus:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(
                lambda x: _penalty_objective_radial(x, mu),
                x0,
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_mu, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            x0 = res.x
            r = np.clip(x0[1:1+N], 1e-6, 1.0)
            theta = x0[1+N:]
            cand_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best = cand_pts.copy()
        except Exception:
            pass
    
    # Final polish pass with highest mu to tighten constraints
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _penalty_objective_radial(x, mu_end),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            x0 = res.x
            r = np.clip(x0[1:1+N], 1e-6, 1.0)
            theta = x0[1+N:]
            cand_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best = cand_pts.copy()
        except Exception:
            pass
    
    return best, best_s

def _coordinate_descent_polish(points, deadline, max_iter=500):
    """
    Coordinate descent on the hard objective in Cartesian space for final precision.
    """
    best_pts = points.copy()
    best_s = _score(best_pts)
    
    steps = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for step in steps:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                for coord in range(2):
                    # Try positive and negative moves
                    for sign in [1, -1]:
                        trial = best_pts.copy()
                        trial[i, coord] += sign * step
                        trial_s = _score(trial)
                        if trial_s > best_s + 1e-14:
                            best_pts = trial
                            best_s = trial_s
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
    return best_pts, best_s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # 1. Generate diverse initial candidates
    candidates = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    candidates.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Two-ring: 6 outer, 6 inner, 1 center
    ang_outer = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pts_outer = np.column_stack([np.cos(ang_outer), np.sin(ang_outer)])
    ang_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    pts_inner = np.column_stack([0.5 * np.cos(ang_inner), 0.5 * np.sin(ang_inner)])
    pts_center = np.array([[0.0, 0.0]])
    candidates.append(np.vstack([pts_outer, pts_inner, pts_center]))
    
    # Three-ring: 5 outer, 5 middle, 3 inner
    a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    a2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
    a3 = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    p1 = np.column_stack([np.cos(a1), np.sin(a1)])
    p2 = np.column_stack([0.6 * np.cos(a2), 0.6 * np.sin(a2)])
    p3 = np.column_stack([0.3 * np.cos(a3), 0.3 * np.sin(a3)])
    candidates.append(np.vstack([p1, p2, p3]))
    
    # Random perturbation of regular polygon
    rng = np.random.RandomState(42)
    ang_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r_reg = 1.0 + 0.1 * rng.randn(N)
    r_reg = np.clip(r_reg, 0.5, 1.5)
    candidates.append(np.column_stack([r_reg * np.cos(ang_reg), r_reg * np.sin(ang_reg)]))

    # 2. Evaluate and select best initial candidate
    best_points = None
    best_score = 0.0
    for c in candidates:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_points = c.copy()

    # 3. Optimization Phase 1: Penalty-based Radial Optimization
    # Run on the best initial candidate
    if time.time() < deadline - 30:
        try:
            opt_pts, opt_score = _penalty_optimize_radial(best_points, deadline)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_pts.copy()
        except Exception:
            pass

    # 4. Optimization Phase 2: Local Refinement (Nelder-Mead on Hard Objective)
    # Nelder-Mead can handle the non-smooth min() better than L-BFGS for final steps
    if time.time() < deadline - 10:
        try:
            res_nm = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            nm_points = res_nm.x.reshape(N, 2)
            nm_score = _score(nm_points)
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_points.copy()
        except Exception:
            pass

    # 5. Optimization Phase 3: Coordinate Descent Polish
    if time.time() < deadline - 2:
        try:
            cd_points, cd_score = _coordinate_descent_polish(best_points, deadline)
            if cd_score > best_score:
                best_score = cd_score
                best_points = cd_points.copy()
        except Exception:
            pass

    return best_points

if __name__ == "__main__":
    points = construct()
    score = _score(points)
    print(f"Score: {score}")
