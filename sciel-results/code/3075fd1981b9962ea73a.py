"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 14
D = 3

# EVOLVE-BLOCK-START

def _pairwise_sq_dists(points):
    """Compute squared pairwise distances as flat array of length N*(N-1)/2."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=-1)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def _score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    sq_dists = _pairwise_sq_dists(points)
    d_min_sq = np.min(sq_dists)
    d_max_sq = np.max(sq_dists)
    if d_min_sq < 1e-24:
        return 1e10
    return d_max_sq / d_min_sq

def _soft_max_log_sq(x, beta=50.0):
    """
    Soft max of log(squared distances) using log-sum-exp.
    Returns log(sum(exp(beta * log_sq_dists))) / beta which approximates max(log_sq_dists).
    """
    # x is array of squared distances
    if np.any(x < 1e-30):
        return -np.inf
    log_x = np.log(x)
    m = np.max(log_x)
    if m < -300:
        return m
    shifted = beta * (log_x - m)
    # Clip for numerical stability
    shifted = np.clip(shifted, -700, 700)
    log_sum = m + np.log(np.sum(np.exp(shifted))) / beta
    return log_sum

def _soft_min_log_sq(x, beta=50.0):
    """
    Soft min of log(squared distances) using log-sum-exp.
    Returns -log(sum(exp(-beta * log_sq_dists))) / beta which approximates min(log_sq_dists).
    """
    if np.any(x < 1e-30):
        return -np.inf
    log_x = np.log(x)
    m = np.min(log_x)
    if m < -300:
        return m
    shifted = -beta * (log_x - m)
    shifted = np.clip(shifted, -700, 700)
    log_sum = m - np.log(np.sum(np.exp(shifted))) / beta
    return log_sum

def _objective_log(params, beta=50.0):
    """
    Smooth objective in log-space: (soft_max_log - soft_min_log)^2
    This is equivalent to log(max_sq/min_sq) which when squared gives the target.
    Actually we want (max/min)^2 = exp(2*(log_max - log_min))
    So minimize 2*(soft_max_log - soft_min_log)
    """
    points = params.reshape(N, D)
    
    # Center to remove translation invariance
    points = points - np.mean(points, axis=0, keepdims=True)
    
    sq_dists = _pairwise_sq_dists(points)
    
    if np.any(sq_dists < 1e-20):
        return 1e10
    
    # Normalize scale: set mean squared distance to 1
    mean_sq = np.mean(sq_dists)
    if mean_sq < 1e-20:
        return 1e10
    points = points / np.sqrt(mean_sq)
    sq_dists = _pairwise_sq_dists(points)
    
    if np.any(sq_dists < 1e-20):
        return 1e10
    
    log_max = _soft_max_log_sq(sq_dists, beta)
    log_min = _soft_min_log_sq(sq_dists, beta)
    
    if log_max < -300 or log_min < -300:
        return 1e10
    
    # The objective is (max/min)^2 = exp(2*(log_max - log_min))
    # But for optimization stability, we minimize the log version:
    # 2*(log_max - log_min) which is proportional to log(objective)
    # Minimizing this is equivalent to minimizing the objective
    obj = 2.0 * (log_max - log_min)
    
    if obj > 100:  # Cap to avoid overflow issues
        return 1e10
    
    return obj

def _objective_true(params):
    """True objective: (max / min)^2."""
    points = params.reshape(N, D)
    return _score(points)

def _tetrahedron_initial():
    """
    Generate initial configuration based on regular tetrahedron vertices
    plus midpoints of edges, then add additional points.
    4 vertices + 6 midpoints + 4 face centers = 14 points.
    """
    a = 1.0
    verts = np.array([
        [a, a, a],
        [a, -a, -a],
        [-a, a, -a],
        [-a, -a, a]
    ], dtype=float)
    
    verts -= np.mean(verts, axis=0)
    edge_len = np.linalg.norm(verts[0] - verts[1])
    verts /= edge_len
    
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    midpoints = []
    for i, j in edges:
        midpoints.append((verts[i] + verts[j]) / 2)
    midpoints = np.array(midpoints)
    
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    face_centers = []
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    points = np.vstack([verts, midpoints, face_centers])
    points -= np.mean(points, axis=0)
    
    return points

def _icosahedron_plus2_initial(rng):
    """Icosahedron (12 vertices) + 2 additional points."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Add 2 points at opposite poles or random positions
    p1 = rng.normal(size=D)
    p1 = p1 / np.linalg.norm(p1)
    p2 = -p1  # Opposite point for symmetry
    
    full_pts = np.vstack([ico, p1, p2])
    full_pts -= np.mean(full_pts, axis=0)
    
    # Random rotation
    rot = Rotation.from_quat(rng.normal(size=4))
    full_pts = rot.apply(full_pts)
    
    return full_pts

def _random_spherical_initial(rng):
    """Random points on sphere."""
    pts = rng.normal(size=(N, D))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pts -= np.mean(pts, axis=0)
    rot = Rotation.from_quat(rng.normal(size=4))
    pts = rot.apply(pts)
    return pts

def _fibonacci_sphere_initial(rng):
    """Fibonacci sphere distribution."""
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = []
    for i in range(N):
        z = 1 - 2 * (i + 0.5) / N
        radius = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden_ratio
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        points.append([x, y, z])
    pts = np.array(points)
    pts -= np.mean(pts, axis=0)
    rot = Rotation.from_quat(rng.normal(size=4))
    pts = rot.apply(pts)
    return pts

def _two_rings_initial(rng):
    """Two rings of 7 points each at different z-heights."""
    z_offset = rng.uniform(0.3, 0.6)
    ring1_angles = np.linspace(0, 2*np.pi, 7, endpoint=False) + rng.uniform(0, 2*np.pi)
    ring2_angles = ring1_angles + np.pi/7
    
    ring1 = np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles), np.full(7, z_offset)])
    ring2 = np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles), np.full(7, -z_offset)])
    pts = np.vstack([ring1, ring2])
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    pts -= np.mean(pts, axis=0)
    rot = Rotation.from_quat(rng.normal(size=4))
    pts = rot.apply(pts)
    return pts

def _polish(points, deadline, beta=200.0):
    """Final high-precision polish pass."""
    if time.time() > deadline - 5.0:
        return points
    
    best_points = points.copy()
    best_score = _score(best_points)
    
    # L-BFGS-B with smooth objective
    try:
        result = minimize(
            _objective_log,
            best_points.flatten(),
            args=(beta,),
            method='L-BFGS-B',
            options={
                'maxiter': 5000,
                'ftol': 1e-14,
                'gtol': 1e-12,
                'maxls': 50
            }
        )
        
        if result.fun < 50:  # Reasonable log-objective value
            candidate = result.x.reshape(N, D)
            candidate -= np.mean(candidate, axis=0)
            candidate_score = _score(candidate)
            if candidate_score < best_score:
                best_score = candidate_score
                best_points = candidate.copy()
    except Exception:
        pass
    
    # Nelder-Mead with tight tolerances
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                _objective_log,
                best_points.flatten(),
                args=(beta,),
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-11,
                    'fatol': 1e-11,
                    'adaptive': True
                }
            )
            
            if result.fun < 50:
                candidate = result.x.reshape(N, D)
                candidate -= np.mean(candidate, axis=0)
                candidate_score = _score(candidate)
                if candidate_score < best_score:
                    best_score = candidate_score
                    best_points = candidate.copy()
        except Exception:
            pass
    
    return best_points

def construct():
    """Construct 14 points in 3D to minimize squared max/min distance ratio."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = np.inf
    
    def update_best(pts):
        nonlocal best_points, best_score
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, D):
            return
        sq_dists = _pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-18:
            return
        score = _score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Tetrahedron-based (primary structural guess)
    candidates.append(_tetrahedron_initial())
    
    # Perturbed tetrahedron
    for trial in range(3):
        p = _tetrahedron_initial()
        p = p + rng.normal(size=(N, D)) * 0.05
        p -= np.mean(p, axis=0)
        candidates.append(p)
    
    # 2. Icosahedron + 2
    for trial in range(3):
        candidates.append(_icosahedron_plus2_initial(rng))
    
    # 3. Fibonacci sphere
    for trial in range(2):
        candidates.append(_fibonacci_sphere_initial(rng))
    
    # 4. Two rings
    for trial in range(3):
        candidates.append(_two_rings_initial(rng))
    
    # 5. Random spherical
    for trial in range(3):
        candidates.append(_random_spherical_initial(rng))
    
    # Evaluate all initial candidates
    for c in candidates:
        update_best(c)
    
    # Progressive beta annealing optimization
    beta_schedule = [10.0, 30.0, 80.0, 200.0, 500.0]
    
    n_candidates = len(candidates)
    
    # Phase 1: Coarse optimization with low beta
    for i in range(n_candidates):
        if time.time() > deadline - 30.0:
            break
        x0 = candidates[i].flatten()
        try:
            result = minimize(
                _objective_log,
                x0,
                args=(beta_schedule[0],),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            if result.x is not None:
                pts = result.x.reshape(N, D)
                pts -= np.mean(pts, axis=0)
                update_best(pts)
                candidates[i] = pts
        except Exception:
            pass
    
    # Phase 2: Medium optimization
    for i in range(n_candidates):
        if time.time() > deadline - 20.0:
            break
        x0 = candidates[i].flatten()
        try:
            result = minimize(
                _objective_log,
                x0,
                args=(beta_schedule[1],),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7}
            )
            if result.x is not None:
                pts = result.x.reshape(N, D)
                pts -= np.mean(pts, axis=0)
                update_best(pts)
                candidates[i] = pts
        except Exception:
            pass
    
    # Phase 3: Fine optimization
    for i in range(n_candidates):
        if time.time() > deadline - 12.0:
            break
        x0 = candidates[i].flatten()
        try:
            result = minimize(
                _objective_log,
                x0,
                args=(beta_schedule[2],),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            if result.x is not None:
                pts = result.x.reshape(N, D)
                pts -= np.mean(pts, axis=0)
                update_best(pts)
                candidates[i] = pts
        except Exception:
            pass
    
    # Phase 4: Very fine optimization on best candidates
    # Sort candidates by score and optimize top ones
    scored = []
    for i in range(n_candidates):
        pts = candidates[i]
        s = _score(pts)
        scored.append((s, i))
    scored.sort()
    
    top_k = min(5, len(scored))
    for rank in range(top_k):
        if time.time() > deadline - 8.0:
            break
        _, idx = scored[rank]
        x0 = candidates[idx].flatten()
        try:
            result = minimize(
                _objective_log,
                x0,
                args=(beta_schedule[3],),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-13, 'gtol': 1e-10}
            )
            if result.x is not None:
                pts = result.x.reshape(N, D)
                pts -= np.mean(pts, axis=0)
                update_best(pts)
                candidates[idx] = pts
        except Exception:
            pass
    
    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 3.0:
        best_points = _polish(best_points, deadline, beta=500.0)
    
    # Ensure we have a valid result
    if best_points is None:
        best_points = _tetrahedron_initial()
    
    # Final validation
    sq_dists = _pairwise_sq_dists(best_points)
    if np.min(sq_dists) < 1e-16:
        best_points = _tetrahedron_initial()
    
    return best_points

# EVOLVE-BLOCK-END
