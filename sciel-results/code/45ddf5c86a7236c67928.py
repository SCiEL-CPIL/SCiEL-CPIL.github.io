import numpy as np
import time
from scipy.optimize import minimize

N = 38

def lj_energy_and_forces(x):
    """
    Compute LJ energy and forces (gradient) for a configuration x of shape (N, 3).
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    r2[r2 == 0] = 1e-10
    
    inv_r6 = 1.0 / (r2**3)
    inv_r12 = inv_r6**2
    
    # Energy
    e_pair = 4.0 * (inv_r12 - inv_r6)
    e_pair[np.diag_indices_from(e_pair)] = 0.0
    energy = 0.5 * np.sum(e_pair)
    
    # Gradient
    inv_r2 = 1.0 / r2
    # dE/dx_i = sum_j 4 * (x_i - x_j) * (6 r^-8 - 12 r^-14)
    # coeff = 24 * (inv_r6 - 2*inv_r12) / r2
    f_scalar = 24.0 * (inv_r6 - 2.0 * inv_r12) * inv_r2
    f_scalar[np.diag_indices_from(f_scalar)] = 0.0
    
    grad = np.sum(diff * f_scalar[..., np.newaxis], axis=1)
    
    return energy, grad

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    rng = np.random.default_rng(42)
    
    # Initial guess: Random points in a sphere
    # Scale to ensure reasonable density
    # Typical cluster diameter is ~5-6 sigma
    x = rng.uniform(-2.5, 2.5, size=(N, 3))
    
    best_x = None
    best_e = float('inf')
    
    def local_minimize(x0):
        # Use L-BFGS-B for robustness
        res = minimize(
            lambda pos: lj_energy_and_forces(pos)[0],
            x0,
            jac=lambda pos: lj_energy_and_forces(pos)[1],
            method='L-BFGS-B',
            options={'gtol': 1e-5, 'maxiter': 100}
        )
        return res.x, res.fun
    
    # Initial relaxation
    x, e = local_minimize(x)
    best_x = x.copy()
    best_e = e
    
    # Basin Hopping parameters
    T = 0.5
    sigma = 0.3
    
    while time.time() < deadline:
        # Perturb
        x_pert = x + rng.normal(0, sigma, size=(N, 3))
        
        # Relax
        x_new, e_new = local_minimize(x_pert)
        
        # Metropolis
        delta_e = e_new - e
        if delta_e < 0 or (delta_e < 10.0 and rng.random() < np.exp(-delta_e / T)):
            x = x_new
            e = e_new
            
            if e < best_e:
                best_e = e
                best_x = x.copy()
        
        # Optional: Adaptive sigma
        # If stuck, increase sigma?
        # For now, keep fixed.
        
    return best_x
