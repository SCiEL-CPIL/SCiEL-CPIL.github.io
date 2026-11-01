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

def generate_icosahedron_poles():
    """Generate Icosahedron vertices + 2 poles."""
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    pts_sym = np.vstack([ico, p_top.reshape(1, D), p_bot.reshape(1, D)])
    return pts_sym

def generate_tetrahedron_based():
    """Generate points based on tetrahedron symmetry."""
    tetra = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    tetra = tetra / np.linalg.norm(tetra, axis=1, keepdims=True)
    
    edge_midpoints = []
    for i in range(4):
        for j in range(i+1, 4):
            mid = (tetra[i] + tetra[j]) / 2
            mid = mid / np.linalg.norm(mid)
            edge_midpoints.append(mid)
    
    face_centers = []
    for i in range(4):
        for j in range(i+1, 4):
            for k in range(j+1, 4):
                center = (tetra[i] + tetra[j] + tetra[k]) / 3
                center = center / np.linalg.norm(center)
                face_centers.append(center)
    
    # 4 tetra + 6 edge mids + 4 face centers = 14
    tetra_full = np.vstack([tetra, np.array(edge_midpoints), np.array(face_centers[:4])])
    return tetra_full

def generate_cuboctahedron_based():
    """Generate points based on Cuboctahedron vertices + 2 random."""
    rng = np.random.default_rng(123)
    perms = np.array([
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0]
    ])
    cubo = perms / np.linalg.norm(perms, axis=1, keepdims=True)
    p1 = rng.normal(size=D)
    p2 = rng.normal(size=D)
    p1 /= np.linalg.norm(p1)
    p2 /= np.linalg.norm(p2)
    pts = np.vstack([cubo, p1.reshape(1, D), p2.reshape(1, D)])
    return pts

def construct():
    start_time = time.time()
    timeout = 140.0 # Safety margin for 150s limit
    
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
    for _ in range(10):
        pts = rng.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts.flatten())
        
    # 3. Icosahedron + Poles
    ico_pts = generate_icosahedron_poles()
    candidates.append(ico_pts.flatten())
    
    # 4. Tetrahedron-based
    tetra_pts = generate_tetrahedron_based()
    candidates.append(tetra_pts.flatten())
    
    # 5. Cuboctahedron-based
    cubo_pts = generate_cuboctahedron_based()
    candidates.append(cubo_pts.flatten())
    
    # 6. Fibonacci sphere perturbations
    for _ in range(5):
        perturbed = fib_pts + rng.normal(scale=0.1, size=(N, D))
        perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
        candidates.append(perturbed.flatten())

    # Optimization Loop
    # We will use a multi-stage tau annealing strategy with L-BFGS-B
    # and a few Nelder-Mead runs for robustness.
    
    tau_schedule = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    for i in range(len(candidates)):
        if time.time() - start_time > timeout - 10:
            break
            
        x0 = candidates[i].copy()
        x0 = x0.reshape(N, D)
        x0 = normalize_points(x0)
        x0 = x0.flatten()
        
        current_best_x = x0
        current_best_val = np.inf
        
        # Stage 1: Nelder-Mead on direct objective to find a good basin
        # This is expensive, so we limit iterations
        try:
            res_nm = minimize(
                objective_direct, 
                current_best_x, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res_nm.x is not None:
                val_nm = objective_direct(res_nm.x)
                if val_nm < current_best_val:
                    current_best_val = val_nm
                    current_best_x = res_nm.x
        except Exception:
            pass
        
        # Stage 2: L-BFGS-B with tau annealing
        for tau in tau_schedule:
            if time.time() - start_time > timeout - 5:
                break
                
            try:
                res_lbfgs = minimize(
                    lambda x: objective_smooth(x, tau=tau),
                    current_best_x,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if res_lbfgs.x is not None:
                    val_lbfgs = objective_smooth(res_lbfgs.x, tau=tau)
                    if val_lbfgs < current_best_val:
                        current_best_val = val_lbfgs
                        current_best_x = res_lbfgs.x
            except Exception:
                pass
        
        # Update global best
        if current_best_x is not None:
            pts_candidate = current_best_x.reshape(N, D)
            update_best(pts_candidate)
            
        # If we have a very good score, we can skip some random starts
        if best_score < 1.05:
            break

    # Final Polish
    if best_points is not None and time.time() - start_time < timeout - 2:
        try:
            x_polish = best_points.flatten()
            # Use a very high tau for final polish
            res_polish = minimize(
                lambda x: objective_smooth(x, tau=1000.0),
                x_polish,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res_polish.x is not None:
                pts_polish = res_polish.x.reshape(N, D)
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
