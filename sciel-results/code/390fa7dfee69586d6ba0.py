"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _all_areas(points):
    """Vectorized computation of all triangle areas."""
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute min triangle area / convex hull area."""
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 3:
        return 0.0
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = np.min(_all_areas(points))
    return min_area / hull_area

def _support_function_to_points(params):
    """
    Convert support function parameters to a set of points inside a convex polygon.
    
    params: shape (n_lines, 3) where each row is [angle, offset, weight]
    Actually, let's use a simpler radial support representation:
    We define a convex polygon by its support function h(theta) = max_i (r_i * cos(theta - alpha_i))
    But to get points, we can just use the vertices of the convex hull defined by these supports.
    
    Alternative: Define the convex region by 14 supporting lines.
    Let's use a representation where we have 14 normals n_i and offsets d_i.
    The region is {x | n_i . x <= d_i for all i}.
    To place points, we place them on the boundary or inside.
    For maximizing min triangle area, points should be on the boundary of the convex hull.
    
    Let's define the convex hull by 14 vertices in a star-shaped manner from the origin.
    p_i = r_i * (cos(alpha_i), sin(alpha_i))
    To ensure convexity, the angles must be sorted and the radii must satisfy certain conditions.
    Actually, any set of points in convex position defines a convex hull.
    The constraint is that the points must be in convex position.
    
    Let's stick to the symmetric parameterization but use a DIFFERENT optimization strategy.
    The current program uses DE + Nelder-Mead/Powell.
    Let's use a **simulated annealing** or **basin hopping** approach with a different objective.
    
    Wait, the directive says "different design or modelling family, a different optimization or search method, or a different representation".
    
    Let's try **optimizing the angles and radii of a convex polygon directly**, ensuring convexity by construction.
    We represent the 14 points as a convex polygon.
    Points are given by polar coordinates (r_i, theta_i).
    To ensure convexity, theta_i must be sorted and the polygon must not self-intersect.
    For a convex polygon, the cross product of consecutive edge vectors must have the same sign.
    
    Let's use the **support function** approach properly.
    A convex body is defined by its support function h(u) = max_{x in K} x.u.
    If we approximate h(u) by a trigonometric polynomial or a set of cosines, we can get a convex shape.
    h(theta) = sum_{k=0}^{M} (a_k cos(k*theta) + b_k sin(k*theta))
    For h to be a valid support function, it must be convex as a function on the circle (i.e., h'' + h >= 0).
    
    This is complex. Let's try a simpler structural change:
    **Optimize the points directly but use a different search algorithm: Basin Hopping with L-BFGS inner loops.**
    Basin hopping is good for global optimization of non-smooth objectives.
    
    Or, even better: **Use the fact that the optimal configuration is likely symmetric.**
    The current program uses central symmetry. Let's try **cyclic symmetry** or **no symmetry** but with a better global optimizer.
    
    Let's try **Differential Evolution** in the full 28D space, but with a **penalty function** for non-convexity (if any point is inside the hull of others, penalize it).
    Actually, if we maximize min triangle area, points will naturally move to the hull.
    
    Let's try a **genetic algorithm** with **tournament selection** and **crossover** in the full 28D space.
    
    Or, let's try **optimizing the angles of a regular 14-gon** with **radial perturbations**.
    Start with a regular 14-gon, and perturb the radii.
    This keeps the points in convex position (if perturbations are small).
    
    Let's implement a **radial perturbation** strategy:
    1. Start with regular 14-gon angles.
    2. Optimize 14 radii r_i.
    3. Points are p_i = r_i * (cos(theta_i), sin(theta_i)).
    4. This guarantees points are in "star-shaped" position. If r_i are close to each other, they are in convex position.
    5. We can add a penalty if the polygon is not convex (check cross products).
    
    This reduces the dimensionality from 28 to 14 (if we fix angles) or keeps it 28 (if we optimize angles and radii).
    Fixing angles to a regular 14-gon is a strong constraint, but it might be close to optimal.
    
    Let's try optimizing **14 radii** for a fixed set of angles (regular 14-gon angles).
    This is a 14D optimization problem.
    
    Then, we can also try **optimizing angles and radii** (28D) but with a convexity penalty.
    
    Let's implement the **fixed-angle radial optimization** first, then refine in full space.
    """
    # This function is a placeholder, we will implement the strategy in construct()
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def _is_convex(points):
        """Check if points form a convex polygon (in order)."""
        n = len(points)
        if n < 3:
            return False
        # Compute cross products of consecutive edges
        signs = []
        for i in range(n):
            p0 = points[i]
            p1 = points[(i + 1) % n]
            p2 = points[(i + 2) % n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            if abs(cross) < 1e-10:
                return False
            signs.append(cross > 0)
        # All crosses must have the same sign
        return all(signs) or not any(signs)

    def _convexity_penalty(points):
        """Penalize non-convex configurations."""
        if _is_convex(points):
            return 0.0
        # Compute min absolute cross product
        n = len(points)
        min_cross = np.inf
        for i in range(n):
            p0 = points[i]
            p1 = points[(i + 1) % n]
            p2 = points[(i + 2) % n]
            v1 = p1 - p0
            v2 = p2 - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            min_cross = min(min_cross, abs(cross))
        return 100.0 / (min_cross + 1e-6)

    # --- Strategy: Optimize radii for fixed regular angles ---
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)

    def _points_from_radii(radii):
        """Convert radii to points using fixed angles."""
        return np.column_stack((radii * cos_angles, radii * sin_angles))

    def _objective_radii(radii):
        """Objective for radii optimization: -score + convexity penalty."""
        pts = _points_from_radii(radii)
        if np.any(radii < 0.1):  # Avoid collapse
            return 1e10
        score = _score(pts)
        penalty = _convexity_penalty(pts)
        return -score + penalty

    # Phase 1: Differential Evolution on 14D radii space
    if time.time() < deadline - 60.0:
        try:
            bounds = [(0.5, 2.0)] * N
            de_res = differential_evolution(_objective_radii, bounds, seed=42,
                                            maxiter=200, popsize=20, tol=1e-6,
                                            polish=True, init='latinhypercube')
            if de_res.fun < 1e9:
                radii_opt = de_res.x
                pts_opt = _points_from_radii(radii_opt)
                score_opt = _score(pts_opt)
                update_best(pts_opt, score_opt)
        except Exception:
            pass

    # Phase 2: Refine in full 28D space using L-BFGS-B with soft-min
    def _objective_soft_full(flat, beta=50.0):
        """Soft-min objective for full 28D parameterization."""
        pts = flat.reshape(N, 2)
        if np.any(~np.isfinite(pts)):
            return 1e10
        areas = _all_areas(pts)
        if np.min(areas) < 1e-12:
            return 1e10
        z = -beta * areas
        max_z = np.max(z)
        if max_z < -500:
            log_sum = max_z
        else:
            log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
        soft_min = -log_sum / beta
        if soft_min < 1e-12:
            return 1e10
        hull_area = _compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1e10
        # Add convexity penalty
        penalty = _convexity_penalty(pts)
        return hull_area / soft_min + penalty

    def _objective_full(flat):
        """Hard objective for full 28D parameterization."""
        pts = flat.reshape(N, 2)
        score = _score(pts)
        penalty = _convexity_penalty(pts)
        return -score + penalty

    # Initialize candidates
    candidates = []
    rng = np.random.RandomState(42)

    # Regular 14-gon
    regular = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates.append(regular.flatten())

    # From Phase 1 if successful
    if best_points is not None:
        candidates.append(best_points.flatten())

    # Random starts
    for _ in range(5):
        rand_pts = rng.uniform(-1, 1, size=(N, 2))
        candidates.append(rand_pts.flatten())

    # Refine each candidate in full space
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 20:
            break

        # Soft-min refinement
        for beta in [50.0, 100.0]:
            try:
                res = minimize(lambda x: _objective_soft_full(x, beta=beta), cand,
                               method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
                pts = res.x.reshape(N, 2)
                score = _score(pts)
                update_best(pts, score)
                if score > best_score:
                    cand = res.x  # Update candidate for next round
            except Exception:
                pass

        # Hard refinement with Powell
        try:
            res = minimize(_objective_full, cand, method='Powell',
                           options={'maxiter': 1000, 'xtol': 1e-11, 'ftol': 1e-13})
            pts = res.x.reshape(N, 2)
            score = _score(pts)
            update_best(pts, score)
        except Exception:
            pass

    # Phase 3: Final polish with Nelder-Mead
    if best_points is not None and time.time() < deadline - 10:
        try:
            res = minimize(_objective_full, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            pts = res.x.reshape(N, 2)
            score = _score(pts)
            update_best(pts, score)
        except Exception:
            pass

    if best_points is None:
        # Fallback: regular 14-gon
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
# EVOLVE-BLOCK-END
