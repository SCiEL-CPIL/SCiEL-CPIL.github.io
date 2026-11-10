import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # --- 1. Initialization: Hexagonal Lattice ---
    # We use a hexagonal lattice as the base. The "superlattice" idea suggests
    # that the optimal packing might not be a perfect hex lattice, but a distorted one.
    # We will optimize the coordinates directly, but start from a good hex-like structure.
    
    def get_hex_init():
        # Estimate spacing
        # Area of hex cell = sqrt(3)/2 * s^2
        # N * Area ~ pi * 1^2
        # s ~ sqrt(2*pi / (sqrt(3)*N))
        spacing = np.sqrt(2.0 * np.pi / (np.sqrt(3.0) * N))
        
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            # Keep points strictly inside, leaving room for the circle radius
            # The radius of the small circles is approx spacing/2
            mask = radius <= 1.0 - spacing / 2.0
            points = points[mask]
            
            if len(points) >= N:
                # Select the N points closest to the center to maximize symmetry/compactness
                dists = np.linalg.norm(points, axis=1)
                idx = np.argsort(dists, kind='stable')
                return points[idx[:N]].copy()
            spacing *= 0.98

    centers = get_hex_init()
    
    # --- 2. Score Calculation ---
    def calculate_score(pts):
        # r = min(min_pair_dist / 2, min(1 - |p|))
        if len(pts) < 2:
            return 1.0
        dists_center = np.linalg.norm(pts, axis=1)
        boundary_r = 1.0 - np.max(dists_center)
        
        # Efficient min pair distance
        # For N=300, cdist is fast enough. 
        # diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        # dist_sq = np.sum(diff**2, axis=2)
        # np.fill_diagonal(dist_sq, np.inf)
        # min_pair_dist = np.sqrt(np.min(dist_sq))
        
        from scipy.spatial.distance import cdist
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair_dist = np.min(d)
        
        pair_r = min_pair_dist / 2.0
        return min(boundary_r, pair_r)

    best_centers = centers.copy()
    best_score = calculate_score(centers)

    # --- 3. Optimization: L-BFGS-B with Smooth Surrogate ---
    # The objective is to maximize r.
    # We minimize an energy function that penalizes overlaps and boundary violations.
    # E = sum_{i<j, d_ij < 2r} (2r - d_ij)^2 + sum_i (1 - r_i)^2 if r_i > 1-r? 
    # Actually, a common approach is to minimize the sum of squared violations of constraints.
    # Constraint 1: d_ij >= 2r  =>  violation = max(0, 2r - d_ij)
    # Constraint 2: r_i <= 1-r  =>  violation = max(0, r_i - (1-r))
    # But r is a variable we want to maximize.
    
    # Alternative: Fix r at a slightly lower value than current best, and try to push points apart.
    # Or, use an epigraph formulation:
    # Minimize -t + sum violations(t, pts)
    # where violations are active if d_ij < 2t or r_i > 1-t.
    
    # Let's use a simpler approach: 
    # Minimize E(pts, t) = -t + lambda * ( sum_{i<j} max(0, 2t - d_ij)^2 + sum_i max(0, r_i - (1-t))^2 )
    # This is non-smooth due to max.
    # Smooth approximation: soft-plus or log-sum-exp.
    # For simplicity and robustness, we can just use the sum of squared violations with a small smoothing if needed, 
    # or rely on L-BFGS-B handling the kinks reasonably well if we start close to optimum.
    # However, L-BFGS-B assumes smoothness.
    
    # Better: Minimize the sum of (1/d_ij)^2 + (1/(1-r_i))^2.
    # This is smooth and pushes points apart and away from boundary.
    # As points get closer, the energy goes to infinity, effectively enforcing the constraint.
    # The global minimum of this energy corresponds to the maximal packing.
    
    def energy_and_grad(flat_x):
        x = flat_x.reshape((N, 2))
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        dist_sq_safe = dist_sq + 1e-9
        np.fill_diagonal(dist_sq_safe, 1.0)
        
        # Energy term 1: Pairwise repulsion
        # E_pair = sum_{i!=j} 1/d_ij^2
        inv_dist_sq = 1.0 / dist_sq_safe
        np.fill_diagonal(inv_dist_sq, 0.0)
        E_pair = np.sum(inv_dist_sq)
        
        # Gradient of E_pair
        # d/dx_i (1/|x_i - x_j|^2) = -2 (x_i - x_j) / |x_i - x_j|^4
        grad_factor = -2.0 / (dist_sq_safe**2)
        np.fill_diagonal(grad_factor, 0.0)
        grad_pair = np.sum(grad_factor[:, :, np.newaxis] * diff, axis=1)
        
        # Energy term 2: Boundary repulsion
        r_i = np.linalg.norm(x, axis=1)
        r_safe = np.where(r_i < 1e-9, 1e-9, r_i)
        d_b = 1.0 - r_i
        d_b_safe = np.where(d_b < 1e-6, 1e-6, d_b)
        
        E_bound = np.sum(1.0 / d_b_safe**2)
        
        # Gradient of E_bound
        # d/dx_i (1/d_b^2) = 2 * (x_i / r_i) / d_b^3
        grad_bound = 2.0 * (x / r_safe[:, np.newaxis]) / d_b_safe[:, np.newaxis]**3
        
        total_energy = E_pair + E_bound
        total_grad = grad_pair + grad_bound
        
        return total_energy, total_grad

    # Run L-BFGS-B
    # We might need to run this multiple times or with different initializations if it gets stuck.
    # But for now, let's try one robust run.
    
    try:
        res = minimize(
            energy_and_grad, 
            best_centers.flatten(), 
            jac=True, 
            method='L-BFGS-B', 
            bounds=[(-1.0, 1.0)] * (2 * N),
            options={'maxiter': 5000, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        optimized_points = res.x.reshape((N, 2))
        
        # Check validity
        if np.all(np.isfinite(optimized_points)):
            r_i = np.linalg.norm(optimized_points, axis=1)
            if np.all(r_i <= 1.0):
                score = calculate_score(optimized_points)
                if score > best_score:
                    best_score = score
                    best_centers = optimized_points
    except Exception:
        pass

    # --- 4. Local Refinement (Polish) ---
    # After the global optimization, we might be stuck in a local minimum or have numerical noise.
    # We can perform a few steps of "steepest ascent" on the actual score function, or just
    # re-run L-BFGS-B with tighter tolerances.
    
    if time.time() < deadline - 5.0:
        # Try to polish with a smaller step size or different method
        # Or simply re-run L-BFGS-B from the current best point
        try:
            res2 = minimize(
                energy_and_grad, 
                best_centers.flatten(), 
                jac=True, 
                method='L-BFGS-B', 
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 2000, 'disp': False, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            polished_points = res2.x.reshape((N, 2))
            
            if np.all(np.isfinite(polished_points)):
                r_i = np.linalg.norm(polished_points, axis=1)
                if np.all(r_i <= 1.0):
                    score = calculate_score(polished_points)
                    if score > best_score:
                        best_score = score
                        best_centers = polished_points
        except Exception:
            pass

    # --- 5. Final Symmetry Breaking / Perturbation (Optional) ---
    # The "superlattice" idea suggests that breaking symmetry might help.
    # If we have time, we can try perturbing the points slightly and re-optimizing.
    # But given the time budget, we might skip this if the previous steps were fast.
    
    if time.time() < deadline - 10.0:
        # Try a few random perturbations and re-optimize
        for attempt in range(3):
            if time.time() > deadline - 5.0:
                break
            
            perturbed = best_centers + np.random.normal(0, 0.01, size=(N, 2))
            # Clamp to circle
            norms = np.linalg.norm(perturbed, axis=1)
            mask = norms > 0.99
            perturbed[mask] *= 0.99 / norms[mask, np.newaxis]
            
            try:
                res3 = minimize(
                    energy_and_grad, 
                    perturbed.flatten(), 
                    jac=True, 
                    method='L-BFGS-B', 
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 1000, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                new_points = res3.x.reshape((N, 2))
                
                if np.all(np.isfinite(new_points)):
                    r_i = np.linalg.norm(new_points, axis=1)
                    if np.all(r_i <= 1.0):
                        score = calculate_score(new_points)
                        if score > best_score:
                            best_score = score
                            best_centers = new_points
            except Exception:
                pass

    return best_centers
