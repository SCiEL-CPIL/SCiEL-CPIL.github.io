import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # safety margin for 630s kill

    def score(pts):
        """Compute (d_min / d_max)^2 for the point configuration."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d) or not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def logsumexp_surrogate(x, beta):
        """
        Smooth surrogate: -log(sum(exp(-beta * dists))) / beta
        This approximates min(dists) as beta -> inf, but is smooth.
        We want to maximize min/max ratio, so we minimize this surrogate
        relative to the max distance.
        """
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        
        # Avoid log of zero
        dists = np.maximum(dists, 1e-12)
        
        # Log-sum-exp of -beta * dists
        scaled = -beta * dists
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        
        # Approximate min distance
        min_approx = -log_sum_exp / beta
        
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        
        ratio = min_approx / d_max
        return -(ratio ** 2)

    def multi_stage_beta_optimize(pts, betas=None, max_iter_per_stage=200):
        """
        Progressive sharpening: start with low beta (smooth), increase beta
        to sharpen the min approximation.
        """
        if betas is None:
            betas = [20, 50, 100, 200, 500]
        
        pts = pts.copy().astype(float)
        
        for beta in betas:
            if time.time() > deadline - 30:
                break
            
            def objective(x):
                return logsumexp_surrogate(x, beta)
            
            try:
                result = minimize(
                    objective,
                    pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': max_iter_per_stage, 'ftol': 1e-14, 'gtol': 1e-10}
                )
                if result.fun < objective(pts.flatten()):
                    pts = result.x.reshape((n, d))
                    pts = pts - np.mean(pts, axis=0)
            except Exception:
                pass
        
        return pts

    def fibonacci_sphere_init(n_pts, seed=42):
        """Generate initial points using Fibonacci sphere."""
        rng = np.random.RandomState(seed)
        indices = np.arange(0, n_pts, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        
        pts = np.column_stack([x, y, z])
        
        # Add small perturbation
        pert = rng.randn(n_pts, d) * 0.05
        pts = pts + pert
        
        return pts

    def coulomb_repulsion_optimize(pts, max_iter=2000, lr_schedule=None):
        """
        Optimize using Coulomb repulsion: each pair of points repels with force
        proportional to 1/r^2. This naturally pushes points apart and tends to
        equalize distances.
        """
        pts = pts.copy().astype(float)
        
        if lr_schedule is None:
            lr_schedule = lambda it: 0.01 * (0.995 ** (it // 50))
        
        for it in range(max_iter):
            if time.time() > deadline - 30:
                break
            
            # Compute all pairwise distances and forces
            dists = pdist(pts)
            if np.any(dists < 1e-12):
                # Add small random jitter to avoid division by zero
                pts += np.random.randn(n, d) * 1e-6
                dists = pdist(pts)
            
            # Compute forces: F_ij = k * (r_i - r_j) / |r_i - r_j|^3
            forces = np.zeros((n, d))
            
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-12:
                        force_mag = 1.0 / (dist ** 3)
                        forces[i] += force_mag * diff
                        forces[j] -= force_mag * diff
            
            forces -= 0.001 * pts
            
            lr = lr_schedule(it)
            pts += lr * forces
            
            pts = pts - np.mean(pts, axis=0)
        
        return pts

    # Generate diverse initial configurations
    rng = np.random.RandomState(12345)
    candidates = []
    
    # Fibonacci sphere with different seeds
    for seed in range(8):
        candidates.append(fibonacci_sphere_init(n, seed=seed))
    
    # Random configurations on sphere
    for i in range(8):
        pts = rng.randn(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        pts = pts / norms
        candidates.append(pts)
    
    # Evaluate all candidates
    best_score = -np.inf
    best_points = None
    
    for cand in candidates:
        if time.time() > deadline - 180:
            break
        s = score(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()
    
    if best_points is None:
        best_points = candidates[0]

    # Phase 1: Differential Evolution on smooth surrogate for global exploration
    if time.time() < deadline - 120:
        def de_objective(x):
            return logsumexp_surrogate(x, beta=50)
        
        bounds = [(-10, 10)] * (n * d)
        
        try:
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=150,
                popsize=30,
                tol=1e-8,
                mutation=(0.5, 1.0),
                recombination=0.8,
                seed=42,
                polish=False
            )
            
            de_pts = de_result.x.reshape((n, d))
            de_pts = de_pts - np.mean(de_pts, axis=0)
            s_de = score(de_pts)
            
            if s_de > best_score:
                best_score = s_de
                best_points = de_pts.copy()
        except Exception:
            pass

    # Phase 2: Multi-stage beta annealing from DE result
    if time.time() < deadline - 60:
        annealed_pts = multi_stage_beta_optimize(
            best_points,
            betas=[20, 50, 100, 200, 500],
            max_iter_per_stage=300
        )
        s_annealed = score(annealed_pts)
        if s_annealed > best_score:
            best_score = s_annealed
            best_points = annealed_pts.copy()

    # Phase 3: Coulomb refinement
    if time.time() < deadline - 30:
        coulomb_pts = coulomb_repulsion_optimize(best_points, max_iter=2000)
        s_coulomb = score(coulomb_pts)
        if s_coulomb > best_score:
            best_score = s_coulomb
            best_points = coulomb_pts.copy()

    # Phase 4: Final beta sharpening
    if time.time() < deadline - 15:
        final_pts = multi_stage_beta_optimize(
            best_points,
            betas=[100, 200, 500, 1000],
            max_iter_per_stage=200
        )
        s_final = score(final_pts)
        if s_final > best_score:
            best_score = s_final
            best_points = final_pts.copy()

    # Final normalization
    best_points = best_points - np.mean(best_points, axis=0)
    norm_max = np.max(np.linalg.norm(best_points, axis=1))
    if norm_max > 0:
        best_points = best_points / norm_max * 100
    
    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))
    
    return best_points
