import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 14
D = 3

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

def _icosahedron():
    """Return 12 icosahedron vertices on unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    return ico

def _cuboctahedron():
    """Return 12 cuboctahedron vertices on unit sphere."""
    perms = np.array([
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
    ])
    return perms / np.linalg.norm(perms, axis=1, keepdims=True)

def _tetrahedron():
    """Return 4 regular tetrahedron vertices on unit sphere."""
    pts = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

def _octahedron():
    """Return 6 regular octahedron vertices on unit sphere."""
    pts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    return pts

def _square_antiprism():
    """Return 8 square antiprism vertices on unit sphere."""
    # Two squares rotated 45 degrees relative to each other
    n = 8
    angles_bottom = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    angles_top = angles_bottom + np.pi/4
    z = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]) * 0.577
    r = np.sqrt(1 - z**2)
    angles = np.concatenate([angles_top, angles_bottom])
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles), z])
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)

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

def objective_exact(x):
    """
    Exact objective: (max_dist / min_dist)^2.
    Used for final polishing with derivative-free methods.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return 1e6
    return np.max(sq_dists) / min_d

def objective_epigraph(x, t, tau=100.0):
    """
    Epigraph formulation: minimize t subject to all pairwise sq_dists <= t * d_min_soft
    and d_min_soft >= epsilon. Uses log-sum-exp for smoothness.
    """
    points = x.reshape(N, D)
    points = normalize_points(points)
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    # Soft min via log-sum-exp
    m_min = np.min(sq_dists)
    exp_min = np.exp(-tau * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min)) / tau
    
    # Soft max via log-sum-exp
    m_max = np.max(sq_dists)
    exp_max = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max)) / tau
    
    # Objective: maximize soft_min - soft_max (we want to minimize this negative)
    # Actually we want to minimize soft_max / soft_min
    # Use log form: log(soft_max) - log(soft_min)
    obj_val = np.log(soft_max) - np.log(soft_min)
    
    return obj_val

def objective_combined(x, tau=100.0, w_ratio=1.0, w_spread=0.1):
    """
    Combined objective: log-ratio + spread penalty.
    The spread penalty encourages all pairwise distances to be similar,
    which helps escape local minima where only a few distances are close.
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
    
    # Log ratio term
    obj_ratio = np.log(log_sum_exp_max) - np.log(log_sum_exp_min)
    
    # Spread penalty: variance of log distances
    log_dists = np.log(sq_dists)
    spread = np.var(log_dists)
    
    return w_ratio * obj_ratio + w_spread * spread

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin (evaluator kills at 150s)
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts)
        if pts.shape != (N, D):
            return
        # Check for duplicate points
        sq_dists = get_pairwise_sq_dists(pts)
        if np.min(sq_dists) < 1e-10:
            return
        pts = normalize_points(pts)
        score = raw_score(pts)
        if score < best_score:
            best_score = score
            best_points = pts.copy()

    rng = np.random.default_rng(42)
    
    # Generate initial candidates
    candidates = []
    
    # 1. Fibonacci sphere (well-distributed)
    fib_pts = fibonacci_sphere(N, seed=42)
    candidates.append(fib_pts.flatten())
    
    # Add perturbed Fibonacci variants
    for _ in range(3):
        perturbed = fib_pts + rng.normal(size=(N, D)) * 0.1
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
        candidates.append(perturbed.flatten())
    
    # 2. Random points on sphere
    for _ in range(8):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
        
    # 3. Icosahedron + 2 points
    ico = _icosahedron()
    
    # Symmetric candidate: Ico + Poles
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    pts_sym = np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)])
    candidates.append(pts_sym.flatten())
    
    for _ in range(5):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([ico, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())

    # 4. Cuboctahedron vertices (12 points) + 2 points
    cubo = _cuboctahedron()
    for _ in range(5):
        p1 = rng.normal(size=D)
        p2 = rng.normal(size=D)
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        pts = np.vstack([cubo, p1.reshape(1, D), p2.reshape(1, D)])
        candidates.append(pts.flatten())
    
    # 5. Square antiprism (8) + tetrahedron (4) + octahedron (2)
    antiprism = _square_antiprism()
    tetra = _tetrahedron()
    octa = _octahedron()
    # Combine: 8 + 4 + 2 = 14
    combo1 = np.vstack([antiprism, tetra, octa[:2]])
    candidates.append(combo1.flatten())
    
    # 6. Concentric shells
    inner = fibonacci_sphere(6, seed=100) * 0.6
    outer = fibonacci_sphere(8, seed=200) * 1.0
    shell_pts = np.vstack([inner, outer])
    shell_pts = shell_pts / np.linalg.norm(shell_pts, axis=1, keepdims=True)
    candidates.append(shell_pts.flatten())
    
    for _ in range(3):
        inner_pert = inner + rng.normal(size=(6, D)) * 0.1
        outer_pert = outer + rng.normal(size=(8, D)) * 0.1
        shell_pert = np.vstack([inner_pert, outer_pert])
        shell_pert = shell_pert / np.linalg.norm(shell_pert, axis=1, keepdims=True)
        candidates.append(shell_pert.flatten())
    
    # 7. Two concentric icosahedra-like arrangements (7+7)
    inner7 = fibonacci_sphere(7, seed=300) * 0.5
    outer7 = fibonacci_sphere(7, seed=400) * 1.0
    shell2 = np.vstack([inner7, outer7])
    shell2 = shell2 / np.linalg.norm(shell2, axis=1, keepdims=True)
    candidates.append(shell2.flatten())

    # Optimization Loop
    n_restarts = len(candidates)
    
    # Phase 1: Smooth Log-Ratio Annealing with refined tau range
    # Use a more gradual annealing schedule
    taus = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0, 1000.0, 3000.0]
    
    for i in range(n_restarts):
        if time.time() - start_time > timeout - 30:
            break
            
        x0 = candidates[i].copy()
        x0 = x0.reshape(N, D)
        x0 = normalize_points(x0)
        x0 = x0.flatten()
        
        try:
            current_x = x0
            
            # Phase 1: Standard smooth optimization with annealing
            for tau in taus:
                if time.time() - start_time > timeout - 30:
                    break
                
                res = minimize(
                    objective_smooth, 
                    current_x, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if res.x is not None and res.fun < np.inf:
                    current_x = res.x
                    pts_current = current_x.reshape(N, D)
                    update_best(pts_current)
            
            # Phase 2: Local perturbation escape from local minima
            if time.time() - start_time < timeout - 25:
                best_local = current_x.copy()
                for _ in range(2):
                    if time.time() - start_time > timeout - 25:
                        break
                    
                    # Add small random perturbation
                    perturbation = rng.normal(size=(N, D)) * 0.03
                    x_pert = best_local.reshape(N, D) + perturbation
                    x_pert = normalize_points(x_pert).flatten()
                    
                    # Optimize from perturbed point with high tau
                    for tau in [500.0, 2000.0]:
                        if time.time() - start_time > timeout - 25:
                            break
                        
                        res = minimize(
                            objective_smooth, 
                            x_pert, 
                            method='L-BFGS-B', 
                            args=(tau,),
                            options={'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                        )
                        
                        if res.x is not None and res.fun < np.inf:
                            x_pert = res.x
                            pts_current = x_pert.reshape(N, D)
                            update_best(pts_current)
                            best_local = x_pert.copy()

        except Exception:
            pass
    
    # Phase 3: Final high-precision polish using Nelder-Mead on exact objective
    if best_points is not None and time.time() - start_time < timeout - 10:
        try:
            x_polish = best_points.flatten()
            
            # Use Nelder-Mead on the exact (nonsmooth) objective for final refinement
            res_nm = minimize(
                objective_exact,
                x_polish,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res.x is not None:
                pts_nm = res.x.reshape(N, D)
                update_best(pts_nm)
        except Exception:
            pass
    
    # Phase 4: Very high-precision L-BFGS-B polish with very high tau
    if best_points is not None and time.time() - start_time < timeout - 3:
        try:
            x_polish = best_points.flatten()
            
            for tau in [10000.0, 50000.0]:
                if time.time() - start_time > timeout - 3:
                    break
                    
                res = minimize(
                    objective_smooth, 
                    x_polish, 
                    method='L-BFGS-B', 
                    args=(tau,),
                    options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
                )
                if res.x is not None and res.fun < np.inf:
                    x_polish = res.x
                    pts_polish = x_polish.reshape(N, D)
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
