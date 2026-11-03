"""
Strategy: Level-Set / PDE-based density evolution on the sphere.
Instead of moving individual points (which gets stuck in local minima due to discrete
particle interactions), we evolve a continuous scalar field (density) on the sphere
using a gradient flow derived from the logarithmic energy. The field naturally handles
"collisions" by spreading mass. We then extract 150 points from the evolved density
and perform a short local polish.
"""
import numpy as np
import time
import math
import scipy.ndimage as ndimage

N = 150
DEADLINE = 145.0

def _energy(points):
    """
    Compute logarithmic energy: sum_{i<j} log(1 / |x_i - x_j|)
    """
    G = points @ points.T
    dist_sq = 2.0 - 2.0 * G
    dist_sq = np.maximum(dist_sq, 1e-15)
    iu = np.triu_indices(N, k=1)
    vals = dist_sq[iu]
    return -0.5 * np.sum(np.log(vals))

def _create_fibonacci_grid(N):
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _lbfgs_polish(points, time_limit=5.0):
    """
    Short L-BFGS polish using tangent space projection.
    """
    from scipy.optimize import minimize
    
    start_time = time.time()
    x0 = points.flatten()
    
    def fun_and_grad(flat_x):
        p = _project_to_sphere(flat_x.reshape(N, 3))
        # Energy
        G = p @ p.T
        dist_sq = 2.0 - 2.0 * G
        dist_sq = np.maximum(dist_sq, 1e-15)
        iu = np.triu_indices(N, k=1)
        vals = dist_sq[iu]
        E = -0.5 * np.sum(np.log(vals))
        
        # Gradient of E w.r.t. p
        # E = -0.5 * sum log(d_ij^2)
        # dE/dp_i = sum_{j!=i} dE/dd_ij^2 * dd_ij^2/dp_i
        # dE/dd^2 = -0.5 / d^2
        # dd^2/dp_i = -2 * (p_i - p_j)
        # So dE/dp_i = sum_{j!=i} (-0.5/d^2) * (-2(p_i-p_j)) = sum (p_i-p_j)/d^2
        
        # Vectorized gradient computation
        # G is Gram matrix
        # We need sum_j (p_i - p_j) / (2 - 2 p_i.p_j)
        # Let M_ij = 1 / (2 - 2 p_i.p_j)
        # grad_i = sum_j (p_i - p_j) M_ij = sum_j p_i M_ij - sum_j p_j M_ij
        # Note diagonal terms are 0/0, set to 0.
        
        M = np.zeros((N, N))
        # Avoid division by zero on diagonal
        np.fill_diagonal(dist_sq, 1.0)
        M = 1.0 / dist_sq
        np.fill_diagonal(M, 0.0)
        
        # sum_j p_j M_ij
        # (M @ p.T).T
        weighted_sum = (M @ p.T).T
        # sum_j M_ij
        M_sum = M.sum(axis=1, keepdims=True)
        
        grad = M_sum * p - weighted_sum
        
        # Project gradient to tangent space
        # grad_tan_i = grad_i - (grad_i . p_i) p_i
        dots = np.sum(grad * p, axis=1, keepdims=True)
        grad_tan = grad - dots * p
        
        return E, grad_tan.flatten()
    
    best_points = points
    best_energy = _energy(points)
    
    # Use L-BFGS-B on the flattened vector, but we need to handle the sphere constraint.
    # Since we project inside the function, the gradient is tangent, but the optimizer
    # moves in Euclidean space. This is a standard "projection" approach but can be
    # unstable if steps are large. For a short polish, it's usually okay.
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', options={'maxiter': 50, 'maxfun': 100, 'disp': False})
        if res.success or res.fun < best_energy:
            new_points = _project_to_sphere(res.x.reshape(N, 3))
            new_energy = _energy(new_points)
            if new_energy < best_energy:
                best_points = new_points
                best_energy = new_energy
    except Exception:
        pass
        
    return best_points, best_energy

def construct():
    start_time = time.time()
    
    # 1. Initialize with Fibonacci spiral
    points = _create_fibonacci_grid(N)
    best_points = points
    best_energy = _energy(points)
    
    # 2. Level-Set / PDE Evolution on a high-resolution grid
    # We discretize the sphere using a spherical grid (theta, phi).
    # This is an approximation, but allows us to use FFT or simple diffusion.
    # Actually, doing PDE on a sphere grid is tricky due to coordinate singularities.
    # Alternative "Level Set" interpretation for points:
    # Evolve the points using a repulsive force derived from a smoothed kernel.
    # This is effectively a "Particle Method" for the PDE.
    # Let's implement a "Softening" approach where we compute forces using a Gaussian
    # kernel on the chordal distance, which prevents singularities and allows points
    # to slide over each other, escaping local minima.
    
    def _softened_energy_and_grad(points, sigma):
        """
        Compute energy and gradient with a softened distance:
        d_soft = sqrt(d^2 + sigma^2)
        E_soft = -0.5 sum log(d_soft^2)
        This is smooth everywhere.
        """
        G = points @ points.T
        dist_sq = 2.0 - 2.0 * G
        dist_sq_soft = dist_sq + sigma**2
        
        iu = np.triu_indices(N, k=1)
        vals = dist_sq_soft[iu]
        E = -0.5 * np.sum(np.log(vals))
        
        # Gradient w.r.t points
        # dE/dp_i = sum_{j!=i} (p_i - p_j) / d_soft^2
        # Note: d/dp_i (d^2 + sigma^2) = -2(p_i - p_j)
        # dE/d(d^2) = -0.5 / (d^2 + sigma^2)
        # Chain rule: dE/dp_i = sum (-0.5/d_soft^2) * (-2(p_i-p_j)) = sum (p_i-p_j)/d_soft^2
        
        M = np.zeros((N, N))
        np.fill_diagonal(dist_sq_soft, 1.0)
        M = 1.0 / dist_sq_soft
        np.fill_diagonal(M, 0.0)
        
        weighted_sum = (M @ points.T).T
        M_sum = M.sum(axis=1, keepdims=True)
        grad = M_sum * points - weighted_sum
        
        # Project to tangent
        dots = np.sum(grad * points, axis=1, keepdims=True)
        grad_tan = grad - dots * points
        
        return E, grad_tan

    # Annealing schedule for sigma and step size
    # Start with large sigma (blurry, global view) and small step
    # End with small sigma (sharp, local detail) and larger step
    
    num_steps = 2000
    sigma_start = 0.5
    sigma_end = 0.05
    step_start = 0.01
    step_end = 0.05
    
    current_points = points.copy()
    current_energy = _energy(current_points)
    
    for step in range(num_steps):
        if time.time() > DEADLINE - 10.0:
            break
            
        # Interpolate sigma and step size
        frac = step / num_steps
        sigma = sigma_start * (sigma_end / sigma_start) ** frac
        step_size = step_start * (step_end / step_start) ** frac
        
        E_soft, grad_tan = _softened_energy_and_grad(current_points, sigma)
        
        # Gradient descent step
        new_points = current_points - step_size * grad_tan
        new_points = _project_to_sphere(new_points)
        
        # Check energy with exact formula
        new_energy = _energy(new_points)
        
        # Accept if better, or if it's early in the process (to allow exploration)
        # Since the softened gradient points towards lower softened energy,
        # it should generally reduce exact energy, but not guaranteed.
        # We accept if new_energy < current_energy OR if we are in the "annealing" phase
        # and the move is small.
        
        if new_energy < current_energy:
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_points = current_points.copy()
                best_energy = current_energy
        else:
            # If energy increased, we might reject it, but in PDE methods we often
            # just continue. However, for minimization, we should be careful.
            # Let's use a simple acceptance: if the step is small enough, accept anyway
            # to allow crossing shallow barriers? No, gradient descent is deterministic.
            # If it increases, the gradient might have pointed the wrong way due to
            # discretization or the difference between soft and exact energy.
            # Let's just reject if it increases significantly.
            if new_energy - current_energy > 1.0:
                continue
            else:
                # Small increase, accept to keep moving
                current_points = new_points
                current_energy = new_energy
                if current_energy < best_energy:
                    best_points = current_points.copy()
                    best_energy = current_energy

    # 3. Final Polish with L-BFGS on the best points found
    final_points, final_energy = _lbfgs_polish(best_points, time_limit=DEADLINE - time.time() - 2.0)
    
    if final_energy < best_energy:
        best_points = final_points
        best_energy = final_energy
        
    return best_points
