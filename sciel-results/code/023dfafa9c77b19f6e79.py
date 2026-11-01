import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.linalg import eigh

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """
    Calculate the squared ratio of max to minimum pairwise distance.
    """
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _gram_to_points(G):
    """
    Convert a Gram matrix to points in 3D space.
    Since G must be PSD and rank <= 3, we take the top 3 eigenvectors.
    This projects the potentially higher-rank relaxed solution onto the 3D subspace.
    """
    # Center the points (translation invariance)
    # Note: The Gram matrix formulation usually assumes centered points or includes translation.
    # For distance ratio, translation doesn't matter. Scaling doesn't matter either.
    # We can fix the scale by setting trace(G) = N (average squared norm = 1) or similar.
    
    # Ensure symmetry
    G = (G + G.T) / 2.0
    
    # Eigen decomposition
    try:
        vals, vecs = eigh(G)
    except np.linalg.LinAlgError:
        # Fallback to random if eigen fails (shouldn't happen for small N)
        return np.random.randn(N, D)
    
    # We want the top 3 eigenvalues (largest) to capture the most variance
    # eigh returns in ascending order
    idx = np.argsort(vals)[::-1][:D]
    top_vecs = vecs[:, idx]
    top_vals = vals[idx]
    
    # Points are vecs * sqrt(vals)
    # If any eigenvalue is negative (due to numerical noise), clamp to 0
    top_vals = np.maximum(top_vals, 0)
    sqrt_vals = np.sqrt(top_vals)
    
    pts = top_vecs * sqrt_vals
    
    # Center the points to remove any residual translation
    pts -= np.mean(pts, axis=0, keepdims=True)
    
    # Normalize to unit norm scale to avoid optimization drift in scale
    # The ratio is scale-invariant, so we can fix the scale.
    # Let's fix the average squared distance to the origin to 1.
    norm_sq = np.sum(pts**2)
    if norm_sq > 1e-12:
        pts /= np.sqrt(norm_sq / N)
        
    return pts

def _objective_from_params(params):
    """
    Objective function for the optimizer.
    We optimize the entries of the Gram matrix G.
    Constraints:
    1. G is PSD (enforced by projecting to PSD in the step or via penalty)
    2. diag(G) is related to norms.
    
    Actually, a simpler parameterization for SDP-like relaxation in Python without CVXPY:
    We can optimize the distance matrix D directly? No, D must satisfy triangle inequality and metric properties.
    
    Alternative Strategy:
    The current program uses gradient descent on coordinates.
    The "Different Strategy" is to use **Simulated Annealing on the Distance Matrix** constrained to be a Euclidean metric?
    That's hard.
    
    Let's stick to the Gram Matrix idea but implement it via **Coordinate Descent on the Gram Matrix** with PSD projection.
    
    Objective: Minimize (max_ij sqrt(G_ii + G_jj - 2G_ij) / min_ij sqrt(G_ii + G_jj - 2G_ij))^2
    Subject to: G is PSD, diag(G) = 1 (fixing scale/shape).
    
    Wait, fixing diag(G)=1 fixes the radius if centered at origin.
    If we fix diag(G)=1, the points lie on a sphere of radius 1 (if centered).
    This is a strong constraint. The optimal configuration for N=14 in 3D might not be on a sphere?
    Actually, for minimizing max/min ratio, the points will tend to be on the boundary of the convex hull, which is a sphere-like surface.
    So constraining to the sphere (diag G = 1) is a valid restriction that likely contains the global optimum.
    
    So, we optimize G such that:
    1. G is PSD.
    2. diag(G) = 1.
    3. We minimize the ratio of max/min distances derived from G.
    
    Since G must be PSD, we can parameterize G = V V^T where V is N x K (K>=3).
    But we are in 3D, so K=3. This brings us back to optimizing coordinates, which is what the parent does.
    
    **True Structural Change:**
    Instead of gradient descent on coordinates, we will use **Simulated Annealing** on the coordinates.
    Gradient descent gets stuck in local minima because the landscape of (max/min)^2 is non-smooth and has many local minima.
    Simulated Annealing can jump over barriers.
    
    Furthermore, we will use a **Repulsive Force Model** (like molecular dynamics) to define the energy landscape, which is different from the direct ratio objective.
    Energy = sum_{i<j} (1/r_ij^2) + lambda * sum_{i<j} (r_ij^2) ... no, that's not right.
    
    Let's use **Repulsive Force Minimization** with a specific potential:
    We want to maximize the minimum distance and minimize the maximum distance.
    This is equivalent to minimizing the variance of the pairwise distances? No.
    
    Let's try **Simulated Annealing** on the coordinates with the objective being the true score, but with a cooling schedule that allows escape from local minima.
    This is materially different from L-BFGS gradient descent.
    
    To make it even more different, we will use a **Genetic Algorithm** or **Particle Swarm Optimization**?
    PSO is a good candidate. It uses a swarm of particles that move through the search space based on their own and the swarm's best positions.
    
    Let's implement **Particle Swarm Optimization (PSO)**.
    """
    pass

def _pso_optimize(initial_swarm, n_iter, time_limit):
    """
    Run Particle Swarm Optimization.
    """
    start_time = time.time()
    n_particles = len(initial_swarm)
    
    # Initialize velocities
    velocities = np.zeros_like(initial_swarm)
    # Random initial velocities
    velocities += np.random.normal(0, 0.1, size=initial_swarm.shape)
    
    # Personal bests
    pbests = initial_swarm.copy()
    pbest_scores = np.array([_true_score(p) for p in pbests])
    
    # Global best
    global_idx = np.argmin(pbest_scores)
    gbest = pbests[global_idx].copy()
    gbest_score = pbest_scores[global_idx]
    
    # PSO Parameters
    w = 0.9  # Inertia
    c1 = 1.5 # Cognitive
    c2 = 1.5 # Social
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
            
        for k in range(n_particles):
            # Update velocity
            r1 = np.random.rand(D)
            r2 = np.random.rand(D)
            
            vel = w * velocities[k] + c1 * r1 * (pbests[k] - initial_swarm[k]) + c2 * r2 * (gbest - initial_swarm[k])
            
            # Clamp velocity to prevent explosion
            max_vel = 0.5
            vel = np.clip(vel, -max_vel, max_vel)
            
            velocities[k] = vel
            
            # Update position
            initial_swarm[k] = initial_swarm[k] + vel
            
            # Ensure distinct points (simple check, will be handled by score inf)
            # Evaluate score
            score = _true_score(initial_swarm[k])
            
            if score < pbest_scores[k]:
                pbest_scores[k] = score
                pbests[k] = initial_swarm[k].copy()
                
                if score < gbest_score:
                    gbest_score = score
                    gbest = initial_swarm[k].copy()
                    
    return gbest, gbest_score

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_score = np.inf
    
    # Strategy: Multi-start PSO
    # Generate diverse initial swarms
    
    # 1. Fibonacci Sphere Swarm
    i = np.arange(N)
    phi = np.arccos(1 - 2*(i+0.5)/N)
    theta = np.pi*(1+5**0.5)*i
    x = np.sin(phi)*np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    z = np.cos(phi)
    fib_pts = np.column_stack([x, y, z])
    
    # Create a swarm around Fibonacci
    swarm_fib = []
    for _ in range(20):
        perturbed = fib_pts + rng.normal(0, 0.1, size=(N, 3))
        # Normalize to sphere to keep them in a good region
        perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
        swarm_fib.append(perturbed)
        
    # 2. Random Sphere Swarm
    swarm_rand = []
    for _ in range(20):
        pts = rng.normal(0, 1, size=(N, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        swarm_rand.append(pts)
        
    # 3. Icosahedron-based Swarm
    phi_ratio = (1 + 5 ** 0.5) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi_ratio, 1], [0, phi_ratio, -1], [0, -phi_ratio, 1], [0, -phi_ratio, -1],
        [1, 0, phi_ratio], [1, 0, -phi_ratio], [-1, 0, phi_ratio], [-1, 0, -phi_ratio],
        [phi_ratio, 1, 0], [phi_ratio, -1, 0], [-phi_ratio, 1, 0], [-phi_ratio, -1, 0]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    icosa_base = verts[:12].copy()
    extra = rng.normal(0, 1, size=(2, 3))
    extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
    icosa_base = np.vstack([icosa_base, extra])
    
    swarm_icosa = []
    for _ in range(20):
        perturbed = icosa_base + rng.normal(0, 0.1, size=(N, 3))
        perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
        swarm_icosa.append(perturbed)
        
    # Combine swarms
    all_swarm = swarm_fib + swarm_rand + swarm_icosa
    
    # Run PSO
    # We have ~145 seconds.
    # PSO with 60 particles, N=14, D=3.
    # Each iteration: 60 evaluations.
    # Each evaluation: cdist (14x14) -> fast.
    # Let's run for as many iterations as possible within time limit.
    
    # Estimate time per iteration
    # Let's just run in a loop and break on deadline.
    
    # We can run multiple PSO runs in parallel? No, single thread.
    # We will run one big PSO with 60 particles for the remaining time.
    # Or better, run several short PSO runs from different swarms and keep the best.
    
    # Let's do 3 runs of 20 particles each, with different time allocations.
    
    # Run 1: Fibonacci-based
    if time.time() - _t0 < DEADLINE - 10:
        pts, score = _pso_optimize(np.array(swarm_fib), n_iter=1000, time_limit=time.time() - _t0 + (DEADLINE - 50))
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
            
    # Run 2: Random-based
    if time.time() - _t0 < DEADLINE - 10:
        pts, score = _pso_optimize(np.array(swarm_rand), n_iter=1000, time_limit=time.time() - _t0 + (DEADLINE - 50))
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
            
    # Run 3: Icosahedron-based
    if time.time() - _t0 < DEADLINE - 10:
        pts, score = _pso_optimize(np.array(swarm_icosa), n_iter=1000, time_limit=time.time() - _t0 + (DEADLINE - 50))
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
            
    # Final Polish: Use the best point found by PSO as a seed for a quick local refinement (Nelder-Mead)
    # This combines the global exploration of PSO with local exploitation.
    if best_pts is not None and time.time() - _t0 < DEADLINE - 5:
        def obj_flat(x):
            pts = x.reshape(N, D)
            return _true_score(pts)
            
        try:
            res = minimize(
                obj_flat, 
                best_pts.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            res_pts = res.x.reshape(N, D)
            res_score = _true_score(res_pts)
            
            if res_score < best_score:
                best_score = res_score
                best_pts = res_pts.copy()
        except Exception:
            pass
            
    # Fallback
    if best_pts is None:
        best_pts = np.array(swarm_fib[0])
        
    # Ensure distinct points
    d = cdist(best_pts, best_pts)
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-6:
        best_pts += rng.normal(0, 1e-4, size=(N, D))
        
    return best_pts
