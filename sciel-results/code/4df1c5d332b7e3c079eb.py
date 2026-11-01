import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

N = 14
D = 3
DEADLINE = 115.0  # Safety margin from 120s

def _pairwise_sq_dists(points):
    """Compute squared pairwise distances for N points in D dimensions."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def _true_score(pts):
    """Calculate the actual score: (max/min)^2."""
    sq_dists = _pairwise_sq_dists(pts)
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def _soft_log_ratio_sq(sq_dists, tau=50.0):
    """
    Smooth approximation of log(max_sq_dist) - log(min_sq_dist).
    Returns a scalar that approximates log(max) - log(min) of squared distances.
    """
    sq_dists = np.maximum(sq_dists, 1e-15)
    log_sq_dists = np.log(sq_dists)
    
    # Soft max of log_sq_dists
    m_max = np.max(log_sq_dists)
    exp_max = np.exp(tau * (log_sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max)) / tau
    
    # Soft min of log_sq_dists
    m_min = np.min(log_sq_dists)
    exp_min = np.exp(-tau * (log_sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min)) / tau
    
    return soft_max - soft_min

def _objective(params, tau=20.0):
    """
    Objective function: minimize the soft log ratio of squared distances.
    Handles translation and scale invariance by centering and normalizing.
    """
    pts = params.reshape(N, D)
    
    # Center to remove translation invariance
    pts = pts - np.mean(pts, axis=0)
    
    # Scale to fix scale invariance: make mean squared distance = 1
    sq_dists = _pairwise_sq_dists(pts)
    mean_sq_dist = np.mean(sq_dists)
    if mean_sq_dist < 1e-12:
        return 1e10
    pts = pts / np.sqrt(mean_sq_dist)
    
    # Recompute squared distances after normalization
    sq_dists = _pairwise_sq_dists(pts)
    
    if np.any(sq_dists < 1e-15):
        return 1e10
    
    return _soft_log_ratio_sq(sq_dists, tau=tau)

def _generate_fib_sphere_points():
    """Generate points on a sphere using Fibonacci distribution."""
    i = np.arange(N)
    phi = np.arccos(1 - 2 * (i + 0.5) / N)
    theta = np.pi * (1 + 5 ** 0.5) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

def _generate_icosahedron_plus_2():
    """
    Generate points based on an icosahedron (12 vertices) plus 2 additional points.
    """
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize to unit sphere
    norms = np.linalg.norm(v, axis=1)
    v = v / norms[:, np.newaxis]
    
    p1 = np.array([0, 0, 1.0])
    p2 = np.array([0, 0, -1.0])
    
    pts = np.vstack([v, p1, p2])
    return pts

def _generate_ring_config():
    """
    Generate a ring-like configuration: points distributed on two rings
    in the x-y plane at different z levels.
    """
    n_per_ring = N // 2
    remaining = N - 2 * n_per_ring
    
    pts = []
    # Ring 1 at z = 0.3
    angles1 = np.linspace(0, 2*np.pi, n_per_ring, endpoint=False)
    for a in angles1:
        pts.append([np.cos(a), np.sin(a), 0.3])
    
    # Ring 2 at z = -0.3
    angles2 = np.linspace(0, 2*np.pi, n_per_ring, endpoint=False) + np.pi / n_per_ring
    for a in angles2:
        pts.append([np.cos(a), np.sin(a), -0.3])
    
    # Remaining points
    if remaining > 0:
        pts.append([0, 0, 0.7])
    if remaining > 1:
        pts.append([0, 0, -0.7])
    
    return np.array(pts[:N])

def _generate_tetrahedron_config():
    """
    Generate a tetrahedron-based configuration: 4 vertices of a regular tetrahedron
    plus 10 points at edge midpoints and face centers.
    """
    # Regular tetrahedron vertices
    s = np.sqrt(2)
    tetra = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ]) / s
    
    # Edge midpoints (6 edges)
    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    midpoints = []
    for i, j in edges:
        mid = (tetra[i] + tetra[j]) / 2
        midpoints.append(mid)
    
    # We have 4 + 6 = 10 points, need 4 more
    # Add face centers (4 faces)
    faces = [(0,1,2), (0,1,3), (0,2,3), (1,2,3)]
    face_centers = []
    for f in faces:
        center = (tetra[f[0]] + tetra[f[1]] + tetra[f[2]]) / 3
        face_centers.append(center)
    
    pts = np.vstack([tetra, midpoints, face_centers])
    # Normalize to unit sphere
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    return pts[:N]

def _generate_spherical_shell():
    """
    Generate points on two concentric spherical shells.
    7 points on inner shell, 7 on outer shell.
    """
    n_inner = 7
    n_outer = N - n_inner
    
    # Inner shell: radius 0.6
    phi_in = np.arccos(1 - 2 * (np.arange(n_inner) + 0.5) / n_inner)
    theta_in = np.pi * (1 + 5 ** 0.5) * np.arange(n_inner)
    inner = np.column_stack([
        0.6 * np.sin(phi_in) * np.cos(theta_in),
        0.6 * np.sin(phi_in) * np.sin(theta_in),
        0.6 * np.cos(phi_in)
    ])
    
    # Outer shell: radius 1.0
    phi_out = np.arccos(1 - 2 * (np.arange(n_outer) + 0.5) / n_outer)
    theta_out = np.pi * (1 + 5 ** 0.5) * np.arange(n_outer)
    outer = np.column_stack([
        np.sin(phi_out) * np.cos(theta_out),
        np.sin(phi_out) * np.sin(theta_out),
        np.cos(phi_out)
    ])
    
    return np.vstack([inner, outer])

def _generate_random_points(rng):
    """Generate random points."""
    pts = rng.normal(size=(N, D))
    rot = Rotation.from_quat(rng.normal(size=4))
    pts = rot.apply(pts)
    return pts

def _generate_random_sphere_points(rng):
    """Generate random points on the sphere."""
    pts = rng.normal(size=(N, D))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    rot = Rotation.from_quat(rng.normal(size=4))
    pts = rot.apply(pts)
    return pts

def construct():
    t_start = time.time()
    
    best_score = np.inf
    best_pts = None
    
    rng = np.random.default_rng(42)
    
    # Initial candidates - diverse starting points
    candidates = []
    candidates.append(_generate_fib_sphere_points())
    candidates.append(_generate_icosahedron_plus_2())
    candidates.append(_generate_ring_config())
    candidates.append(_generate_tetrahedron_config())
    candidates.append(_generate_spherical_shell())
    
    # Add random starts
    for _ in range(6):
        candidates.append(_generate_random_points(rng))
    for _ in range(4):
        candidates.append(_generate_random_sphere_points(rng))
    
    # Multi-stage optimization:
    # Stage 1: Coarse optimization with L-BFGS from each candidate
    # Stage 2: Progressive tau increase for finer optimization
    # Stage 3: Fine polish with higher tau for more precise optimization
    # Stage 4: Additional local refinement with Nelder-Mead
    
    stage1_results = []
    
    for i, init_pts in enumerate(candidates):
        if time.time() - t_start > DEADLINE - 40:
            break
        
        try:
            init_flat = init_pts.flatten()
            
            # Stage 1: L-BFGS with moderate tau
            res = minimize(
                _objective,
                init_flat,
                args=(15.0,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            opt_pts = res.x.reshape(N, D)
            opt_pts = opt_pts - np.mean(opt_pts, axis=0)
            
            # Check for distinct points
            sq_dists = _pairwise_sq_dists(opt_pts)
            if np.min(sq_dists) < 1e-10:
                continue
            
            score = _true_score(opt_pts)
            stage1_results.append((score, opt_pts))
            
            if score < best_score:
                best_score = score
                best_pts = opt_pts.copy()
                
        except Exception:
            continue
    
    # Stage 2: Progressive tau increase with more iterations
    tau_schedule = [30.0, 80.0, 200.0, 500.0]
    
    if len(stage1_results) > 0:
        stage1_results.sort(key=lambda x: x[0])
        top_results = stage1_results[:min(4, len(stage1_results))]
        
        for i, (init_score, init_pts) in enumerate(top_results):
            for tau in tau_schedule:
                if time.time() - t_start > DEADLINE - 15:
                    break
                
                try:
                    res = minimize(
                        _objective,
                        init_pts.flatten(),
                        args=(tau,),
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-12}
                    )
                    
                    opt_pts = res.x.reshape(N, D)
                    opt_pts = opt_pts - np.mean(opt_pts, axis=0)
                    
                    sq_dists = _pairwise_sq_dists(opt_pts)
                    if np.min(sq_dists) < 1e-10:
                        continue
                    
                    score = _true_score(opt_pts)
                    
                    if score < best_score:
                        best_score = score
                        best_pts = opt_pts.copy()
                        
                except Exception:
                    continue
    
    # Stage 3: Fine polish with highest tau for more precise optimization
    if time.time() - t_start < DEADLINE - 8 and best_pts is not None:
        try:
            def _objective_fine(params):
                pts = params.reshape(N, D)
                pts = pts - np.mean(pts, axis=0)
                sq_dists = _pairwise_sq_dists(pts)
                mean_sq_dist = np.mean(sq_dists)
                if mean_sq_dist < 1e-12:
                    return 1e10
                pts = pts / np.sqrt(mean_sq_dist)
                sq_dists = _pairwise_sq_dists(pts)
                if np.any(sq_dists < 1e-15):
                    return 1e10
                return _soft_log_ratio_sq(sq_dists, tau=1000.0)
            
            res = minimize(
                _objective_fine,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14}
            )
            
            opt_pts = res.x.reshape(N, D)
            opt_pts = opt_pts - np.mean(opt_pts, axis=0)
            
            sq_dists = _pairwise_sq_dists(opt_pts)
            if np.min(sq_dists) < 1e-10:
                return best_pts
            
            score = _true_score(opt_pts)
            if score < best_score:
                best_score = score
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
    
    # Stage 4: Additional local refinement with Nelder-Mead
    if time.time() - t_start < DEADLINE - 3 and best_pts is not None:
        try:
            def _objective_nm(params):
                pts = params.reshape(N, D)
                pts = pts - np.mean(pts, axis=0)
                sq_dists = _pairwise_sq_dists(pts)
                mean_sq_dist = np.mean(sq_dists)
                if mean_sq_dist < 1e-12:
                    return 1e10
                pts = pts / np.sqrt(mean_sq_dist)
                sq_dists = _pairwise_sq_dists(pts)
                if np.any(sq_dists < 1e-15):
                    return 1e10
                return _soft_log_ratio_sq(sq_dists, tau=500.0)
            
            res = minimize(
                _objective_nm,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            
            opt_pts = res.x.reshape(N, D)
            opt_pts = opt_pts - np.mean(opt_pts, axis=0)
            
            sq_dists = _pairwise_sq_dists(opt_pts)
            if np.min(sq_dists) < 1e-10:
                return best_pts
            
            score = _true_score(opt_pts)
            if score < best_score:
                best_score = score
                best_pts = opt_pts.copy()
                
        except Exception:
            pass
    
    # Fallback
    if best_pts is None:
        best_pts = _generate_fib_sphere_points()
        
    return best_pts
