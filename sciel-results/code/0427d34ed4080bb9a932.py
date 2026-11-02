import time
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3
DEADLINE = 115.0  # Safety margin from 120s (or 150s if evaluator kills at 150)

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
    
    # Add 2 points at the poles
    p1 = np.array([0, 0, 1.0])
    p2 = np.array([0, 0, -1.0])
    
    pts = np.vstack([v, p1, p2])
    return pts

def _generate_ring_config():
    """
    Generate a ring-based configuration: 2 rings of 7 points each at different heights.
    This provides good angular separation.
    """
    pts = []
    # Ring 1 at z=0.5
    for i in range(7):
        angle = 2 * np.pi * i / 7
        pts.append([np.cos(angle), np.sin(angle), 0.5])
    # Ring 2 at z=-0.5, rotated by half-step
    for i in range(7):
        angle = 2 * np.pi * (i + 0.5) / 7
        pts.append([np.cos(angle), np.sin(angle), -0.5])
    return np.array(pts)

def _generate_two_rings_optimized():
    """
    Two rings with optimized radial distances.
    """
    pts = []
    r1, r2 = 1.0, 0.85
    z1, z2 = 0.6, -0.6
    for i in range(7):
        angle = 2 * np.pi * i / 7
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle), z1])
    for i in range(7):
        angle = 2 * np.pi * (i + 0.5) / 7
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle), z2])
    return np.array(pts)

def _generate_random_points(rng):
    """Generate random points."""
    pts = rng.normal(size=(N, D))
    from scipy.spatial.transform import Rotation
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
    candidates.append(_generate_two_rings_optimized())
    
    # Add more random starts with different seeds
    for seed in range(12):
        rng_local = np.random.default_rng(seed)
        candidates.append(_generate_random_points(rng_local))
    
    # Multi-stage optimization with varied tau schedules
    tau_schedules = [
        [10.0, 50.0, 200.0, 1000.0, 5000.0],
        [5.0, 30.0, 150.0, 800.0, 4000.0],
        [20.0, 100.0, 500.0, 2000.0, 10000.0],
    ]
    
    for i, init_pts in enumerate(candidates):
        if time.time() - t_start > DEADLINE - 15:
            break
        
        try:
            init_flat = init_pts.flatten()
            
            # Choose tau schedule based on candidate index for diversity
            tau_sched = tau_schedules[i % len(tau_schedules)]
            
            for tau in tau_sched:
                if time.time() - t_start > DEADLINE - 5:
                    break
                
                res = minimize(
                    _objective,
                    init_flat,
                    args=(tau,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                opt_pts = res.x.reshape(N, D)
                opt_pts = opt_pts - np.mean(opt_pts, axis=0)
                
                # Check for distinct points
                sq_dists = _pairwise_sq_dists(opt_pts)
                if np.min(sq_dists) < 1e-10:
                    continue
                
                score = _true_score(opt_pts)
                
                if score < best_score:
                    best_score = score
                    best_pts = opt_pts.copy()
                    init_flat = opt_pts.flatten()
                    
        except Exception:
            continue
    
    # Final high-precision polish on the global best using epigraph formulation
    if time.time() - t_start < DEADLINE - 3 and best_pts is not None:
        try:
            def _objective_epigraph(params):
                """
                Epigraph formulation: minimize t subject to 
                log_sq_dist_i <= log_t for all i (max constraint)
                log_sq_dist_i >= -log_t for all i (min constraint)
                This is equivalent to minimizing max_i log_sq_dist_i - min_i log_sq_dist_i
                """
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
                
                # Use very high tau for near-exact max/min
                return _soft_log_ratio_sq(sq_dists, tau=10000.0)
            
            res = minimize(
                _objective_epigraph,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
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
