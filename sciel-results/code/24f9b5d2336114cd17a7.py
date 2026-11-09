import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize, differential_evolution

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 620  # Safety margin before 630s kill

    def score(pts):
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(x):
        pts = x.reshape((n, d))
        if not np.all(np.isfinite(pts)):
            return 1.0
        s = score(pts)
        return -s

    # Vectorized gradient computation for the objective
    def objective_and_grad(x):
        pts = x.reshape((n, d))
        if not np.all(np.isfinite(pts)):
            return 1.0, np.zeros_like(x)
        
        # Center the points to remove translation invariance
        pts_centered = pts - np.mean(pts, axis=0)
        
        # Compute all pairwise distances
        diff = pts_centered[:, np.newaxis, :] - pts_centered[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dists, np.inf)
        
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return 1.0, np.zeros_like(x)
        
        ratio = d_min / d_max
        s = ratio ** 2
        
        # Compute gradient numerically (simpler and more robust for this non-smooth function)
        grad = np.zeros_like(x)
        eps = 1e-7
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = objective(x_plus)
            
            x_minus = x.copy()
            x_minus[i] -= eps
            f_minus = objective(x_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        return -s, -grad

    # Coulomb repulsion energy (inverse square law) for spreading points
    def coulomb_energy(x):
        pts = x.reshape((n, d))
        if not np.all(np.isfinite(pts)):
            return 1e10
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        # Use a softened inverse square to avoid singularities
        energy = np.sum(1.0 / (dists ** 2 + 1e-6))
        return energy

    def coulomb_grad(x):
        pts = x.reshape((n, d))
        if not np.all(np.isfinite(pts)):
            return np.zeros_like(x)
        grad = np.zeros_like(x)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist_sq = np.dot(diff, diff) + 1e-6
                dist = np.sqrt(dist_sq)
                # Gradient of 1/dist^2 with respect to pts[i]
                grad[i] += 4 * diff / (dist_sq ** 2)
                grad[j] -= 4 * diff / (dist_sq ** 2)
        return grad.reshape((n, d))

    best_points = None
    best_score = -np.inf

    def update_best(pts):
        nonlocal best_points, best_score
        if pts is None or not np.all(np.isfinite(pts)):
            return
        s = score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # Strategy 1: Coulomb repulsion pre-stage to spread points
    if time.time() < deadline - 180:
        try:
            # Start from a random configuration and minimize Coulomb energy
            rng = np.random.RandomState(42)
            x_init = rng.randn(n * d) * 5
            result_coulomb = minimize(
                coulomb_energy,
                x_init,
                jac=coulomb_grad,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-10}
            )
            pts_coulomb = result_coulomb.x.reshape((n, d))
            if np.all(np.isfinite(pts_coulomb)):
                update_best(pts_coulomb)
        except Exception:
            pass

    # Strategy 2: Differential Evolution (global optimization)
    if time.time() < deadline - 120:
        try:
            bounds = [(-10, 10)] * (n * d)
            result_de = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-8,
                polish=False,
                init='sobol'
            )
            pts_de = result_de.x.reshape((n, d))
            if np.all(np.isfinite(pts_de)):
                update_best(pts_de)
        except Exception:
            pass

    # Strategy 3: Multiple Nelder-Mead restarts from diverse initializations
    rng = np.random.RandomState(123)
    
    # Create diverse starting points
    initial_configs = []
    
    # Fibonacci sphere
    def fibonacci_sphere(n_pts):
        indices = np.arange(0, n_pts, dtype=float) + 0.5
        phi_f = np.arccos(1 - 2 * indices / n_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.cos(theta) * np.sin(phi_f)
        y = np.sin(theta) * np.sin(phi_f)
        z = np.cos(phi_f)
        return np.column_stack([x, y, z])
    
    fib14 = fibonacci_sphere(14)
    initial_configs.append(fib14)
    
    # Random points on sphere
    for i in range(15):
        pts = rng.randn(n, d)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        initial_configs.append(pts)
    
    # Perturbed Fibonacci
    for i in range(5):
        pert = fib14 + rng.randn(n, d) * 0.1
        initial_configs.append(pert)
    
    # If we have a Coulomb result, use it as a starting point
    if best_points is not None:
        initial_configs.append(best_points.copy())
    
    # Run Nelder-Mead on each initial config
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        current_pts = init_pts.copy()
        
        # First polish with Nelder-Mead
        try:
            result_nm = minimize(
                objective,
                current_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            pts_nm = result_nm.x.reshape((n, d))
            if np.all(np.isfinite(pts_nm)):
                update_best(pts_nm)
                if score(pts_nm) > score(current_pts):
                    current_pts = pts_nm
        except Exception:
            pass
        
        # Try a few random kicks and re-optimize
        if time.time() < deadline - 20:
            for _ in range(3):
                if time.time() > deadline - 20:
                    break
                kick_scale = 0.05 + 0.1 * (i / len(initial_configs))
                kick = rng.randn(n, d) * kick_scale
                kicked_pts = current_pts + kick
                try:
                    result_nm2 = minimize(
                        objective,
                        kicked_pts.flatten(),
                        method='Nelder-Mead',
                        options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                    )
                    pts_kick = result_nm2.x.reshape((n, d))
                    if np.all(np.isfinite(pts_kick)):
                        update_best(pts_kick)
                        if score(pts_kick) > score(current_pts):
                            current_pts = pts_kick
                except Exception:
                    pass

    # Strategy 4: Simulated Annealing to escape local minima
    if best_points is not None and time.time() < deadline - 30:
        try:
            current = best_points.copy()
            current_score = score(current)
            best_sa = current.copy()
            best_sa_score = current_score
            temperature = 0.1
            cooling_rate = 0.999
            steps = 5000
            
            for step in range(steps):
                if time.time() > deadline - 10:
                    break
                
                # Random perturbation
                perturbation = rng.randn(n, d) * temperature
                candidate = current + perturbation
                
                cand_score = score(candidate)
                delta = cand_score - current_score
                
                # Accept if better, or with probability exp(delta / temperature)
                if delta > 0 or np.random.random() < np.exp(delta / temperature):
                    current = candidate
                    current_score = cand_score
                    if current_score > best_sa_score:
                        best_sa_score = current_score
                        best_sa = current.copy()
                
                temperature *= cooling_rate
            
            update_best(best_sa)
        except Exception:
            pass

    # Final polish on the best solution
    if best_points is not None and time.time() < deadline - 5:
        try:
            result_final = minimize(
                objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            pts_final = result_final.x.reshape((n, d))
            if np.all(np.isfinite(pts_final)):
                update_best(pts_final)
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = fibonacci_sphere(14)

    # Normalize to ensure finite and reasonable scale
    best_points = best_points - np.mean(best_points, axis=0)
    norm_max = np.max(np.linalg.norm(best_points, axis=1))
    if norm_max > 0:
        best_points = best_points / norm_max * 100

    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))

    return best_points
