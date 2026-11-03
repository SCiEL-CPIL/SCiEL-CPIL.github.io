"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    N = 50
    
    # --- Energy-based Optimization ---
    
    def compute_score(centers):
        """Compute the packing radius from centers."""
        if len(centers) != N:
            return -1.0
        dists = np.hypot(centers[:, 0], centers[:, 1])
        if np.any(dists >= 1.0):
            return -1.0
        # Compute minimum pairwise distance
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                if d < min_dist:
                    min_dist = d
        r_pair = min_dist / 2
        r_boundary = 1.0 - np.max(dists)
        return min(r_pair, r_boundary)

    def energy_and_grad(x):
        """
        Compute energy and gradient for circle packing.
        x: shape (N, 2)
        Energy: sum_{i<j} 1/d_ij^2 + sum_i 1/(1-|c_i|)^2
        """
        x = x.reshape(N, 2)
        E = 0.0
        grad = np.zeros_like(x)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i, 0] - x[j, 0]
                dy = x[i, 1] - x[j, 1]
                d2 = dx*dx + dy*dy
                if d2 < 1e-12:
                    # Avoid singularity
                    d2 = 1e-12
                    dx = 1e-6
                    dy = 0.0
                # term = 1/d2
                # d(1/d2)/dx_i = -2 dx / d2^2
                coeff = -2.0 / (d2 * d2)
                grad[i, 0] += coeff * dx
                grad[i, 1] += coeff * dy
                grad[j, 0] -= coeff * dx
                grad[j, 1] -= coeff * dy
                E += 1.0 / d2
        
        # Boundary repulsion
        for i in range(N):
            xi = x[i, 0]
            yi = x[i, 1]
            r = np.sqrt(xi*xi + yi*yi)
            if r >= 1.0:
                # Push back hard
                r = 0.999999
                # Gradient points inward
                grad[i, 0] += 1000.0 * (xi / r)
                grad[i, 1] += 1000.0 * (yi / r)
                E += 1e6
            else:
                # term = 1/(1-r)^2
                # d(1/(1-r)^2)/dx = -2(1-r)^-3 * (-dx/r) = 2 dx / (r (1-r)^3)
                factor = 2.0 / (r * (1.0 - r)**3)
                grad[i, 0] += factor * xi
                grad[i, 1] += factor * yi
                E += 1.0 / ((1.0 - r)**2)
        
        return E, grad

    def get_ring_initialization():
        """Generate initial centers based on ring structure."""
        ring_sizes = [1, 6, 12, 12, 19]
        # Radii approx for r=0.1258
        ring_radii = [0.0, 0.252, 0.504, 0.756, 0.944]
        # Angles offsets
        ring_angles = [0.0, 0.0, np.pi/6, 0.0, np.pi/19]
        
        centers = []
        for i, (n, r, theta_off) in enumerate(zip(ring_sizes, ring_radii, ring_angles)):
            if i == 0:
                centers.append([0.0, 0.0])
            else:
                angles = np.linspace(0, 2*np.pi, n, endpoint=False) + theta_off
                for a in angles:
                    centers.append([r * np.cos(a), r * np.sin(a)])
        return np.array(centers)

    best_centers = get_ring_initialization()
    best_score = compute_score(best_centers)
    
    # Optimization loop
    n_restarts = 10
    for trial in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        if trial == 0:
            x0 = best_centers.copy()
        else:
            x0 = best_centers + np.random.randn(N, 2) * 0.01
        
        # Ensure x0 is valid (inside circle)
        dists = np.hypot(x0[:, 0], x0[:, 1])
        mask = dists >= 0.99
        if np.any(mask):
            x0[mask] *= 0.98 / dists[mask]
        
        try:
            res = minimize(energy_and_grad, x0, method='L-BFGS-B', jac=True,
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            x_opt = res.x.reshape(N, 2)
            
            # Check if valid
            score = compute_score(x_opt)
            if score > best_score:
                best_score = score
                best_centers = x_opt
        except Exception:
            pass

    # Final polish with smaller perturbations if time remains
    if time.time() < deadline - 5:
        try:
            x0 = best_centers
            res = minimize(energy_and_grad, x0, method='L-BFGS-B', jac=True,
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            x_opt = res.x.reshape(N, 2)
            score = compute_score(x_opt)
            if score > best_score:
                best_score = score
                best_centers = x_opt
        except Exception:
            pass

    return best_centers
# EVOLVE-BLOCK-END
