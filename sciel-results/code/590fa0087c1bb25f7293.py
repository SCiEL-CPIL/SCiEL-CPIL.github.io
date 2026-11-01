import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import eigh

N = 14
D = 3
NUM_PAIRS = N * (N - 1) // 2

def get_pairwise_sq_dists(points):
    """
    Compute squared pairwise distances for N points in D dimensions.
    Returns a flat array of length N*(N-1)/2.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    """
    Calculate the actual score: (max_dist / min_dist)^2.
    """
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return np.inf
    return (np.max(sq_dists) / min_d)

def normalize_points(points):
    """
    Center and normalize points to unit mean squared norm.
    This removes translational and scale invariance.
    """
    points = points - np.mean(points, axis=0)
    mean_norm_sq = np.mean(np.sum(points**2, axis=1))
    if mean_norm_sq > 1e-12:
        points = points / np.sqrt(mean_norm_sq)
    return points

def objective_smooth(x, tau=100.0):
    """
    Smooth approximation of the log ratio of max to min squared distance.
    Uses Log-Sum-Exp for soft max and soft min.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau
    
    # Soft min
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
    
    # Minimize log(soft_max) - log(soft_min)
    obj_val = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    return obj_val

def objective_epigraph(x, tau=50.0):
    """
    Epigraph formulation: minimize t such that:
      soft_max(sq_dists) <= t * soft_min(sq_dists)
    Reformulated as: minimize log(t) + penalty * max(0, log(soft_max) - log(t) - log(soft_min))
    """
    # x layout: [points (N*D), log_t (1)]
    pts = x[:N*D].reshape(N, D)
    log_t = x[N*D]
    t = np.exp(log_t)
    
    pts = normalize_points(pts)
    sq_dists = get_pairwise_sq_dists(pts)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft max and soft min
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau
    
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
    
    log_ratio = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    log_t_val = log_t
    
    # Objective: log_t + penalty for violation
    violation = np.maximum(0.0, log_ratio - log_t_val)
    penalty_coeff = 100.0
    
    obj = log_t_val + penalty_coeff * violation**2
    
    return obj

def objective_epigraph_grad(x, tau=50.0):
    """
    Gradient for epigraph objective.
    """
    pts = x[:N*D].reshape(N, D)
    log_t = x[N*D]
    
    pts = normalize_points(pts)
    sq_dists = get_pairwise_sq_dists(pts)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - m_max))
    log_sum_exp_max = m_max + np.log(np.sum(exp_max)) / tau
    
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    log_sum_exp_min = m_min - np.log(np.sum(exp_min)) / tau
    
    log_ratio = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    log_t_val = log_t
    
    violation = np.maximum(0.0, log_ratio - log_t_val)
    penalty_coeff = 100.0
    
    # d_obj/d_log_t = 1 + 2*penalty_coeff*violation*(-1) if violation > 0 else 1
    if violation > 0:
        d_obj_d_log_t = 1.0 - 2.0 * penalty_coeff * violation
    else:
        d_obj_d_log_t = 1.0
    
    # For the points part, we'd need to compute gradients of soft_max and soft_min
    # This is complex, so we'll use numerical gradients or just L-BFGS-B with finite differences
    return None  # Let L-BFGS-B handle it with finite differences

def objective_direct(x):
    """
    Direct non-smooth objective for SLSQP or other methods that can handle it.
    Minimizes log(max) - log(min) or just ratio.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return 1e10
    return np.log(np.max(sq_dists)) - np.log(np.min(sq_dists))

def fibonacci_sphere(n, seed=42):
    """Generate n points on a sphere using Fibonacci lattice."""
    rng = np.random.default_rng(seed)
    golden_angle = np.pi * (3 - np.sqrt(5))
    points = np.zeros((n, D))
    for i in range(n):
        z = 1 - 2 * i / (n - 1)
        r = np.sqrt(max(0, 1 - z**2))
        theta = golden_angle * i + rng.uniform(0, 2*np.pi) * 0.01
        points[i, 0] = r * np.cos(theta)
        points[i, 1] = r * np.sin(theta)
        points[i, 2] = z
    return points

def generate_ring_config(n_inner, n_outer, radius_inner=0.5, radius_outer=1.0, seed=42):
    """Generate points in two concentric rings."""
    rng = np.random.default_rng(seed)
    pts = []
    # Inner ring
    for i in range(n_inner):
        theta = 2 * np.pi * i / n_inner
        pts.append([radius_inner * np.cos(theta), radius_inner * np.sin(theta), 0.0])
    # Outer ring
    for i in range(n_outer):
        theta = 2 * np.pi * i / n_outer + np.pi / n_outer  # offset
        pts.append([radius_outer * np.cos(theta), radius_outer * np.sin(theta), 0.0])
    return np.array(pts)

def generate_spherical_shell(n, r_min=0.7, r_max=1.0, seed=42):
    """Generate points in a spherical shell."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n):
        # Random direction
        v = rng.normal(size=D)
        v = v / np.linalg.norm(v)
        # Random radius
        r = rng.uniform(r_min, r_max)
        pts.append(r * v)
    return np.array(pts)

def construct():
    start_time = time.time()
    timeout = 145.0 # Safety margin for 150s limit
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        pts = normalize_points(pts)
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Fibonacci sphere
    fib_pts = fibonacci_sphere(N)
    candidates.append(fib_pts.flatten())
    
    # 2. Random points on sphere
    for _ in range(15):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
        
    # 3. Icosahedron + 2 points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Symmetric candidate: Ico + Poles
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    pts_sym = np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)])
    candidates.append(pts_sym.flatten())
    
    # 4. Tetrahedron-based initialization
    # Regular tetrahedron vertices
    tetra = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    tetra = tetra / np.linalg.norm(tetra, axis=1, keepdims=True)
    
    # Edge midpoints
    edge_midpoints = []
    for i in range(4):
        for j in range(i+1, 4):
            mid = (tetra[i] + tetra[j]) / 2
            mid = mid / np.linalg.norm(mid)
            edge_midpoints.append(mid)
    
    # Add 6 more points to reach 14: face centers
    face_centers = []
    for i in range(4):
        for j in range(i+1, 4):
            for k in range(j+1, 4):
                center = (tetra[i] + tetra[j] + tetra[k]) / 3
                center = center / np.linalg.norm(center)
                face_centers.append(center)
    
    # Combine: 4 tetra + 6 edge mids + 4 face centers = 14
    tetra_full = np.vstack([tetra, np.array(edge_midpoints), np.array(face_centers[:4])])
    candidates.append(tetra_full.flatten())
    
    # 5. Cuboctahedron vertices (12 points) + 2 random
    perms = np.array([
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
    ])
    cubo = perms / np.linalg.norm(perms, axis=1, keepdims=True)
    for _ in range(10):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([cubo, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())
    
    # 6. Fibonacci sphere perturbations
    for _ in range(5):
        perturbed = fib_pts + rng.normal(scale=0.1, size=(N, D))
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
        candidates.append(perturbed.flatten())
    
    # Optimization Loop
    n_restarts = len(candidates)
    
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 15:
            break
            
        x0 = candidates[i].copy()
        x0 = x0.reshape(N, D)
        x0 = normalize_points(x0)
        x0 = x0.flatten()
        
        try:
            # Use Nelder-Mead for non-smooth objective first
            res = minimize(
                objective_direct, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res.x is not None:
                pts_current = res.x.reshape(N, D)
                update_best(pts_current)
                
        except Exception:
            pass
        
        # Then refine with smooth objective using L-BFGS-B
        if time.time() - start_time > timeout - 10:
            break
            
        if best_points is not None:
            try:
                x_smooth = best_points.flatten()
                res2 = minimize(
                    objective_smooth, 
                    x_smooth, 
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res2.x is not None:
                    pts_smooth = res2.x.reshape(N, D)
                    update_best(pts_smooth)
            except Exception:
                pass
    
    # Final polish with smooth objective
    if best_points is not None and time.time() - start_time < timeout - 5:
        try:
            x_polish = best_points.flatten()
            res = minimize(
                objective_smooth, 
                x_polish, 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.x is not None:
                pts_polish = res.x.reshape(N, D)
                update_best(pts_polish)
        except Exception:
            pass

    if best_points is None:
        best_points = normalize_points(candidates[0].reshape(N, D))
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
