import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 14
D = 3
NP = N * (N - 1) // 2  # number of pairwise distances

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def objective_log_ratio(x, tau=100.0):
    """Smooth log of max/min ratio using log-sum-exp."""
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-14)
    
    # Soft log-max: log(sum(exp(tau*d))) / tau
    m_max = np.max(sq_dists)
    log_sum_exp_max = m_max + np.log(np.sum(np.exp(tau * (sq_dists - m_max)))) / tau
    
    # Soft log-min: -log(sum(exp(-tau*d))) / tau
    m_min = np.min(sq_dists)
    log_sum_exp_min = m_min - np.log(np.sum(np.exp(-tau * (sq_dists - m_min)))) / tau
    
    return log_sum_exp_max - log_sum_exp_min

def objective_epigraph(x):
    """Epigraph formulation: minimize t subject to constraints.
    x = [points (N*D), log_max, log_min]
    Minimize log_max - log_min
    Constraints: log_max >= log(d_i) for all i, log_min <= log(d_i) for all i
    """
    points = x[:N*D].reshape(N, D)
    log_max = x[N*D]
    log_min = x[N*D + 1]
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-14)
    log_dists = np.log(sq_dists)
    
    # Objective: minimize log_max - log_min
    obj = log_max - log_min
    
    # Constraints (for SLSQP): g_i >= 0
    # log_max >= log(d_i) => log_max - log(d_i) >= 0
    # log_min <= log(d_i) => log(d_i) - log_min >= 0
    cons = []
    for i in range(NP):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N*D] - np.log(np.maximum(get_pairwise_sq_dists(x[:N*D].reshape(N, D))[i], 1e-14))})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: np.log(np.maximum(get_pairwise_sq_dists(x[:N*D].reshape(N, D))[i], 1e-14)) - x[N*D+1]})
    
    return obj, cons

def objective_epigraph_vec(x):
    """Vectorized epigraph objective for SLSQP with bounds on t."""
    points = x[:N*D].reshape(N, D)
    log_max = x[N*D]
    log_min = x[N*D + 1]
    
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-14)
    log_dists = np.log(sq_dists)
    
    # Penalty for constraint violations (smooth approximation)
    # log_max should be >= max(log_dists)
    # log_min should be <= min(log_dists)
    viol_max = np.maximum(0, np.max(log_dists) - log_max)
    viol_min = np.maximum(0, log_min - np.min(log_dists))
    
    return log_max - log_min + 1000.0 * (viol_max**2 + viol_min**2)

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-10:
        return np.inf
    return (np.max(sq_dists) / np.min(sq_dists))

def fibonacci_sphere(n, radius=1.0):
    """Generate n points evenly distributed on a sphere using Fibonacci lattice."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    points = np.zeros((n, 3))
    for i in range(n):
        z = 1.0 - 2.0 * i / (n - 1) if n > 1 else 0.0
        r = np.sqrt(max(0, 1.0 - z*z))
        theta = golden_angle * i
        points[i] = [r * np.cos(theta), r * np.sin(theta), z]
    return points * radius

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        try:
            score = raw_score(pts)
            if score < best_score:
                best_score = score
                best_points = pts.copy()
        except Exception:
            pass

    rng = np.random.default_rng(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Fibonacci sphere (evenly distributed on sphere)
    fib_pts = fibonacci_sphere(N, radius=2.0)
    candidates.append(fib_pts.flatten())
    
    # 2. Icosahedron vertices (12) + 2 additional points
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True) * 2.0
    for _ in range(3):
        p1 = rng.normal(size=D) * 0.3
        p2 = rng.normal(size=D) * 0.3
        pts = np.vstack([ico, p1, p2])
        candidates.append(pts.flatten())
    
    # 3. Random points on sphere with various radii
    for radius in [1.5, 2.0, 2.5]:
        for _ in range(5):
            pts = rng.normal(size=(N, D))
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms * radius
            candidates.append(pts.flatten())
    
    # 4. Grid-based initialization
    grid_pts = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                grid_pts.append([x, y, z])
    grid_pts = np.array(grid_pts[:14], dtype=float)
    grid_pts = grid_pts - np.mean(grid_pts, axis=0)
    grid_pts *= 1.5
    candidates.append(grid_pts.flatten())
    
    # 5. FCC lattice subset
    fcc_pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                fcc_pts.append([i, j, k])
                fcc_pts.append([i+0.5, j+0.5, k])
                fcc_pts.append([i+0.5, j, k+0.5])
                fcc_pts.append([i, j+0.5, k+0.5])
    fcc_pts = np.array(fcc_pts[:14], dtype=float)
    fcc_pts = fcc_pts - np.mean(fcc_pts, axis=0)
    fcc_pts *= 1.2
    candidates.append(fcc_pts.flatten())
    
    # Phase 1: Differential Evolution with smooth objective
    bounds = [(-4, 4)] * (N * D)
    
    de_results = []
    for seed in [42, 123, 456]:
        if time.time() - start_time > timeout - 60:
            break
        try:
            res_de = differential_evolution(
                lambda x: objective_log_ratio(x, tau=50.0),
                bounds,
                maxiter=80,
                popsize=12,
                seed=seed,
                tol=1e-6,
                polish=False
            )
            if res_de.x is not None:
                de_results.append(res_de.x)
                # Polish with increasing tau
                for tau in [100, 500, 2000]:
                    if time.time() - start_time > timeout - 30:
                        break
                    try:
                        res_polish = minimize(
                            lambda x, t=tau: objective_log_ratio(x, tau=t),
                            res_de.x,
                            method='L-BFGS-B',
                            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                        )
                        if res_polish.x is not None:
                            update_best(res_polish.x.reshape(N, D))
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Phase 2: Polish each candidate with multi-tau schedule
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout - 20:
            break
        
        current_x = init_x.copy()
        for tau in [50, 200, 1000, 5000]:
            if time.time() - start_time > timeout - 10:
                break
            try:
                res = minimize(
                    lambda x, t=tau: objective_log_ratio(x, tau=t),
                    current_x,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.x is not None and res.fun < np.inf:
                    current_x = res.x
                    update_best(res.x.reshape(N, D))
            except Exception:
                pass
    
    # Phase 3: Final high-precision polish on best solution using epigraph-like approach
    if best_points is not None and time.time() - start_time < timeout - 5:
        # Try SLSQP with explicit bounds on the ratio components
        try:
            # Start from best point, use higher tau
            for tau in [2000, 10000, 50000]:
                if time.time() - start_time > timeout - 2:
                    break
                res_final = minimize(
                    lambda x, t=tau: objective_log_ratio(x, tau=t),
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                if res_final.x is not None:
                    update_best(res_final.x.reshape(N, D))
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = candidates[0].reshape(N, D)
        best_score = raw_score(best_points)

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
