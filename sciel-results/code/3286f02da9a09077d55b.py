# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

_DEADLINE_SECONDS = 630.0
_SAFETY_MARGIN = 30.0


def _score(points: np.ndarray) -> float:
    """Compute (d_min / d_max)^2, with safeguards for degenerate cases."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if not (np.isfinite(d_min) and np.isfinite(d_max)):
        return 0.0
    if d_max <= 1e-15:
        return 0.0
    if d_min <= 1e-15:
        return 0.0
    ratio = d_min / d_max
    return float(ratio * ratio)


def _soft_score(points: np.ndarray, alpha: float = 50.0) -> float:
    """
    Smooth surrogate for (d_min / d_max)^2 using log-sum-exp.
    
    soft_min(dists) ≈ -logsumexp(-alpha * dists) / alpha
    soft_max(dists) ≈ logsumexp(alpha * dists) / alpha
    
    This gives a differentiable approximation that's easier to optimize.
    """
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    if not np.all(np.isfinite(dists)):
        return 0.0
    if np.min(dists) <= 1e-15 or np.max(dists) <= 1e-15:
        return 0.0
    
    # Soft min and max using log-sum-exp trick
    # For numerical stability
    d = dists
    a = alpha
    
    # soft_min = -1/a * log(sum(exp(-a*d)))
    neg_ad = -a * d
    max_neg = np.max(neg_ad)
    soft_min = -(max_neg + np.log(np.sum(np.exp(neg_ad - max_neg)))) / a
    
    # soft_max = 1/a * log(sum(exp(a*d)))
    ad = a * d
    max_ad = np.max(ad)
    soft_max = (max_ad + np.log(np.sum(np.exp(ad - max_ad)))) / a
    
    if soft_max <= 1e-15 or soft_min <= 1e-15:
        return 0.0
    
    ratio = soft_min / soft_max
    return float(ratio * ratio)


def _neg_soft_score(points: np.ndarray, alpha: float = 50.0) -> float:
    """Negative soft score for minimization."""
    return -_soft_score(points, alpha)


def _neg_score(points: np.ndarray) -> float:
    """Negative of score for minimization."""
    return -_score(points)


def _initial_icosahedron() -> np.ndarray:
    """
    Return 12 vertices of a regular icosahedron in 3D.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1],
    ]
    pts = np.array(verts, dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts


def _tetrahedron() -> np.ndarray:
    """Return 4 vertices of a regular tetrahedron in 3D."""
    pts = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts


def _cube() -> np.ndarray:
    """Return 8 vertices of a cube in 3D."""
    pts = np.array([
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ], dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts


def _octahedron() -> np.ndarray:
    """Return 6 vertices of a regular octahedron in 3D."""
    pts = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ], dtype=np.float64)
    return pts


def _generate_initial_guesses() -> list:
    """
    Generate diverse initial guesses for 14 points in 3D.
    """
    guesses = []
    
    # 1. Icosahedron + 2 optimized points
    ico = _initial_icosahedron()
    rng = np.random.default_rng(42)
    pts = ico.copy()
    for _ in range(2):
        best_candidate = None
        best_min_dist = -1
        for _ in range(5000):
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            r = np.sqrt(1 - z**2)
            candidate = np.array([r * np.cos(theta), r * np.sin(theta), z])
            dists = np.linalg.norm(pts - candidate, axis=1)
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = candidate.copy()
        if best_candidate is not None:
            pts = np.vstack([pts, best_candidate])
    guesses.append(pts)
    
    # 2. Cube + tetrahedron (8 + 4 = 12, need 2 more)
    cube = _cube()
    tetra = _tetrahedron()
    tetra_scaled = tetra * 0.5
    combined = np.vstack([cube, tetra_scaled])
    rng2 = np.random.default_rng(123)
    for _ in range(2):
        best_candidate = None
        best_min_dist = -1
        for _ in range(5000):
            theta = rng2.uniform(0, 2 * np.pi)
            z = rng2.uniform(-1, 1)
            r = np.sqrt(1 - z**2)
            candidate = np.array([r * np.cos(theta), r * np.sin(theta), z])
            dists = np.linalg.norm(combined - candidate, axis=1)
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = candidate.copy()
        if best_candidate is not None:
            combined = np.vstack([combined, best_candidate])
    guesses.append(combined)
    
    # 3. Octahedron + 8 cube vertices
    octa = _octahedron()
    cube = _cube()
    combined = np.vstack([octa, cube])
    guesses.append(combined)
    
    # 4. Random points with good spread
    rng3 = np.random.default_rng(456)
    random_pts = np.zeros((14, 3))
    for i in range(14):
        theta = rng3.uniform(0, 2 * np.pi)
        z = rng3.uniform(-1, 1)
        r = np.sqrt(1 - z**2)
        random_pts[i] = [r * np.cos(theta), r * np.sin(theta), z]
    guesses.append(random_pts)
    
    # 5. Symmetric arrangement: 2 poles + 12 around equator
    pts = np.zeros((14, 3))
    pts[0] = [0, 0, 1]
    pts[1] = [0, 0, -1]
    for i in range(12):
        angle = 2 * np.pi * i / 12
        pts[i + 2] = [np.cos(angle), np.sin(angle), 0]
    guesses.append(pts)
    
    # 6. Another symmetric: 3 rings
    pts = np.zeros((14, 3))
    pts[0] = [0, 0, 1]
    pts[1] = [0, 0, -1]
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = np.sqrt(1 - 0.25)
        pts[i + 2] = [r * np.cos(angle), r * np.sin(angle), 0.5]
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        r = np.sqrt(1 - 0.25)
        pts[i + 8] = [r * np.cos(angle), r * np.sin(angle), -0.5]
    guesses.append(pts)
    
    return guesses


def _nelder_mead_optimize(points: np.ndarray, deadline: float, max_iter: int = 2000, 
                           use_soft: bool = True, alpha: float = 50.0) -> np.ndarray:
    """
    Optimize using Nelder-Mead method on the flattened 42D vector.
    Uses soft surrogate if use_soft=True.
    """
    start = time.time()
    x0 = points.flatten()
    
    if use_soft:
        def objective(x):
            if time.time() > deadline:
                return 0.0
            pts = x.reshape(14, 3)
            return _neg_soft_score(pts, alpha)
    else:
        def objective(x):
            if time.time() > deadline:
                return 0.0
            pts = x.reshape(14, 3)
            return _neg_score(pts)
    
    try:
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        if result.success or result.fun < 0:
            return result.x.reshape(14, 3)
        else:
            return points
    except Exception:
        return points


def _refine_with_gradient(points: np.ndarray, deadline: float) -> np.ndarray:
    """
    High-precision polish pass using coordinate descent with tight tolerances.
    """
    best_points = points.copy()
    best_score = _score(best_points)
    start = time.time()
    
    n = len(points)
    step_sizes = np.full((n, 3), 0.01)
    
    no_improve_count = 0
    threshold = 1e-14
    
    for iteration in range(3000):
        if time.time() > deadline:
            break
        
        improved = False
        
        for i in range(n):
            for j in range(3):
                if time.time() > deadline:
                    return best_points
                
                # Try positive step
                pts_plus = best_points.copy()
                pts_plus[i, j] += step_sizes[i, j]
                score_plus = _score(pts_plus)
                
                # Try negative step
                pts_minus = best_points.copy()
                pts_minus[i, j] -= step_sizes[i, j]
                score_minus = _score(pts_minus)
                
                best_new_score = max(score_plus, score_minus)
                
                if best_new_score > best_score + threshold:
                    if score_plus >= score_minus:
                        best_points = pts_plus
                    else:
                        best_points = pts_minus
                    best_score = best_new_score
                    improved = True
                    step_sizes[i, j] *= 1.03
                else:
                    step_sizes[i, j] *= 0.9
        
        if not improved:
            no_improve_count += 1
            if no_improve_count > 15:
                break
        else:
            no_improve_count = 0
    
    return best_points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS - _SAFETY_MARGIN
    
    # Generate diverse initial guesses
    guesses = _generate_initial_guesses()
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Optimize each initial guess with soft surrogate (broader search)
    for i, guess in enumerate(guesses):
        if time.time() > deadline:
            break
        
        # Use soft surrogate for smoother landscape
        opt_points = _nelder_mead_optimize(guess, deadline, max_iter=3000, 
                                            use_soft=True, alpha=30.0)
        opt_score = _score(opt_points)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
    
    # Phase 2: Refine best with raw objective (sharper local search)
    if best_points is not None:
        if time.time() < deadline - 60:
            refined = _nelder_mead_optimize(best_points, deadline, max_iter=2000, 
                                             use_soft=False)
            refined_score = _score(refined)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined.copy()
        
        # Phase 3: High-precision polish
        if time.time() < deadline - 30:
            polished = _refine_with_gradient(best_points, deadline)
            polished_score = _score(polished)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished.copy()
    
    # Phase 4: Try more random starts if time permits
    if best_points is None:
        best_points = guesses[0]
        best_score = _score(best_points)
    
    rng = np.random.default_rng(789)
    extra_starts = 0
    while time.time() < deadline - 45 and extra_starts < 15:
        random_pts = np.zeros((14, 3))
        for _ in range(14):
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            r = np.sqrt(1 - z**2)
            random_pts[_] = [r * np.cos(theta), r * np.sin(theta), z]
        
        # Use soft surrogate for random starts
        opt = _nelder_mead_optimize(random_pts, deadline, max_iter=1500, 
                                     use_soft=True, alpha=30.0)
        score = _score(opt)
        
        if score > best_score:
            best_score = score
            best_points = opt.copy()
        
        extra_starts += 1
    
    # Final polish on the best solution found
    if time.time() < deadline - 15:
        final_polish = _refine_with_gradient(best_points, deadline)
        final_score = _score(final_polish)
        if final_score > best_score:
            best_score = final_score
            best_points = final_polish.copy()
    
    # Final check: ensure shape and finiteness
    best_points = np.asarray(best_points, dtype=np.float64)
    if best_points.shape != (14, 3):
        best_points = guesses[0]
    
    if not np.all(np.isfinite(best_points)):
        best_points = guesses[0]
    
    return best_points


# EVOLVE-BLOCK-END
