import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Uses SciPy L-BFGS-B optimizer to minimize overlap energy for increasing radii.
    This approach uses a quasi-Newton method which is more robust and accurate
    than simple gradient descent for finding tight packings.
    """
    
    def energy_and_grad(points, r):
        """
        Compute the energy (sum of squared overlaps) and its gradient.
        Overlap with other circles: (2r - dist)^2 if dist < 2r
        Overlap with boundary: (1 - r - dist)^2 if dist > 1 - r
        """
        points = np.asarray(points).reshape(-1, 2)
        energy = 0.0
        grad = np.zeros_like(points)
        
        # Pairwise overlaps
        for i in range(N):
            for j in range(i + 1, N):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 2 * r:
                    # Overlap amount
                    ov = 2 * r - dist
                    # Energy term: ov^2
                    energy += ov * ov
                    # Gradient of ov^2 w.r.t points[i]
                    # d(ov^2)/d(points[i]) = 2 * ov * d(ov)/d(points[i])
                    # ov = 2r - ||points[i] - points[j]||
                    # d(ov)/d(points[i]) = - (points[i] - points[j]) / dist
                    if dist > 1e-9:
                        factor = 2 * ov * (-1.0 / dist)
                        grad[i] += factor * diff
                        grad[j] -= factor * diff
                    else:
                        # Perturb if coincident
                        angle = np.random.rand() * 2 * np.pi
                        diff = np.array([np.cos(angle), np.sin(angle)])
                        dist = 1.0
                        factor = 2 * ov * (-1.0 / dist)
                        grad[i] += factor * diff
                        grad[j] -= factor * diff

        # Boundary overlaps
        for i in range(N):
            dist = np.linalg.norm(points[i])
            if dist > 1.0 - r:
                ov = dist - (1.0 - r)
                energy += ov * ov
                # Gradient of ov^2 w.r.t points[i]
                # ov = ||points[i]|| - (1-r)
                # d(ov)/d(points[i]) = points[i] / dist
                if dist > 1e-9:
                    factor = 2 * ov * (1.0 / dist)
                    grad[i] += factor * points[i]
                else:
                    # At center, gradient is undefined, but force should be 0 if r < 1
                    pass

        return energy, grad

    def minimize_energy(points, r, max_iters=1000):
        """
        Run L-BFGS-B to minimize energy for a fixed radius r.
        """
        points_flat = points.flatten()
        
        def obj(x):
            p = x.reshape(-1, 2)
            e, _ = energy_and_grad(p, r)
            return e
            
        def grad(x):
            p = x.reshape(-1, 2)
            _, g = energy_and_grad(p, r)
            return g.flatten()

        # Use L-BFGS-B for robustness
        res = minimize(
            obj, 
            points_flat, 
            jac=grad, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        return res.x.reshape(-1, 2), res.fun

    def get_score(points):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        radii = np.linalg.norm(points, axis=1)
        min_boundary = 1.0 - np.max(radii)
        return min(min_pairwise / 2.0, min_boundary)

    # Initialization: Fermat Spiral
    points = np.zeros((N, 2))
    for i in range(N):
        r_init = np.sqrt(i / N) * 0.9
        theta = i * np.pi * (3 - np.sqrt(5))
        points[i] = [r_init * np.cos(theta), r_init * np.sin(theta)]
        
    current_points = points.copy()
    r = 0.05
    min_step = 0.001
    max_r = 0.126 # Slightly above known best to allow discovery
    
    start_time = time.time()
    time_limit = 115.0
    
    best_overall_points = points.copy()
    best_overall_r = r
    best_score = get_score(best_overall_points)
    
    while time.time() - start_time < time_limit:
        r_new = r + min_step
        
        # Run L-BFGS-B
        new_points, energy = minimize_energy(current_points, r_new, max_iters=500)
        
        # Check if packing is valid (energy close to 0)
        # Energy is sum of squared overlaps. If < 1e-8, it's effectively valid.
        if energy < 1e-8:
            r = r_new
            current_points = new_points
            score = get_score(current_points)
            if score > best_score:
                best_score = score
                best_overall_r = r
                best_overall_points = current_points.copy()
                
            # Adapt step size
            if r > 0.12:
                min_step = 0.0001
            elif r > 0.10:
                min_step = 0.0005
            else:
                min_step = 0.001
        else:
            # Failed to pack at r_new
            # Reduce step and try again
            min_step *= 0.5
            if min_step < 1e-6:
                break
                
    # Final polish: Run a few more iterations at the best radius found
    # to squeeze out any remaining slack
    final_r = best_overall_r
    for _ in range(3):
        new_points, energy = minimize_energy(best_overall_points, final_r, max_iters=1000)
        if energy < 1e-10:
            best_overall_points = new_points
        else:
            break

    return best_overall_points

# EVOLVE-BLOCK-END
