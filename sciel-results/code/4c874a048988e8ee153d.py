"""New approach: parameterized coefficients with Nelder-Mead global optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

# ---------------------------------------------------------------------------
# Precomputation for fast evaluation
# ---------------------------------------------------------------------------

# Hermite polynomials via recurrence
def _hermite_values(n, x):
    """Evaluate physicists' Hermite polynomial H_n at points x."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    
    H_nm1 = np.ones_like(x)
    H_n = 2.0 * x
    
    for k in range(1, n):
        H_new = 2.0 * x * H_n - 2.0 * k * H_nm1
        H_nm1 = H_n
        H_n = H_new
        
    return H_n

# Gauss-Hermite quadrature for L1
N_GH = 150
u_nodes, u_weights = np.polynomial.hermite.hermgauss(N_GH)
x_nodes = np.sqrt(2.0) * u_nodes
w_nodes = np.sqrt(2.0) * u_weights

H0_nodes = _hermite_values(0, x_nodes)
H4_nodes = _hermite_values(4, x_nodes)
H8_nodes = _hermite_values(8, x_nodes)
H12_nodes = _hermite_values(12, x_nodes)

# Dense grid for L_inf
N_GRID = 3000
x_grid = np.linspace(-12.0, 12.0, N_GRID)
exp_grid = np.exp(-0.5 * x_grid**2)

H0_grid = _hermite_values(0, x_grid)
H4_grid = _hermite_values(4, x_grid)
H8_grid = _hermite_values(8, x_grid)
H12_grid = _hermite_values(12, x_grid)

# Constants
C0 = 2**0 * math.factorial(0)
C4 = 2**4 * math.factorial(4)
C8 = 2**8 * math.factorial(8)
C12 = 2**12 * math.factorial(12)
SQRT_PI = math.sqrt(math.pi)

# Values at 0 for constraint g(0)=0: a0*1 + a4*(-3) + a8*105 + a12*(-10395) = 0
H0_0 = 1.0
H4_0 = -3.0
H8_0 = 105.0
H12_0 = -10395.0

def _objective_from_params(params):
    """
    New parameterization: use log-scale and ratio parameters to explore wider range.
    params = [log_scale, ratio1, ratio2]
    a0 = sign0 * exp(log_scale)
    a4 = ratio1 * a0
    a8 = ratio2 * a0
    """
    log_scale, ratio1, ratio2 = params
    a0 = math.exp(log_scale)
    a4 = ratio1 * a0
    a8 = ratio2 * a0
    
    # Calculate a12 from constraint g(0)=0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # L2 norm squared
    norm2_sq = SQRT_PI * (a0**2 * C0 + a4**2 * C4 + a8**2 * C8 + a12**2 * C12)
    
    if norm2_sq < 1e-12:
        return 1e100
        
    # L1 norm
    P_nodes = a0 * H0_nodes + a4 * H4_nodes + a8 * H8_nodes + a12 * H12_nodes
    l1_val = np.sum(w_nodes * np.abs(P_nodes))
    
    # L_inf norm
    P_grid = a0 * H0_grid + a4 * H4_grid + a8 * H8_grid + a12 * H12_grid
    g_grid = P_grid * exp_grid
    max_val = np.max(np.abs(g_grid))
    
    if max_val < 1e-100:
        return 1e100
        
    bound = (l1_val * max_val) / norm2_sq
    return bound

def _objective_direct(a):
    """Direct objective for final polish."""
    a0, a4, a8 = a
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    norm2_sq = SQRT_PI * (a0**2 * C0 + a4**2 * C4 + a8**2 * C8 + a12**2 * C12)
    if norm2_sq < 1e-12:
        return 1e100
        
    P_nodes = a0 * H0_nodes + a4 * H4_nodes + a8 * H8_nodes + a12 * H12_nodes
    l1_val = np.sum(w_nodes * np.abs(P_nodes))
    
    P_grid = a0 * H0_grid + a4 * H4_grid + a8 * H8_grid + a12 * H12_grid
    g_grid = P_grid * exp_grid
    max_val = np.max(np.abs(g_grid))
    
    if max_val < 1e-100:
        return 1e100
        
    bound = (l1_val * max_val) / norm2_sq
    return bound

def construct():
    """
    Return the three free Hermite coefficients (degrees 0, 4, 8).
    Uses Nelder-Mead on parameterized space, then polish with L-BFGS-B.
    """
    # Parameter bounds: log_scale in [-3, 3], ratios in [-2, 2]
    param_bounds = [(-3.0, 3.0), (-2.0, 2.0), (-2.0, 2.0)]
    
    best_val = 1e100
    best_params = None
    best_a = np.array([0.0, 0.0, 0.0])
    
    # Multi-start Nelder-Mead in parameter space
    rng = np.random.default_rng(123)
    
    # Deterministic starts
    starts = [
        np.array([0.0, 0.0, 0.0]),
        np.array([-0.5, 0.1, 0.01]),
        np.array([-1.0, -0.2, 0.05]),
        np.array([0.5, 0.3, -0.02]),
        np.array([-0.3, 0.15, -0.03]),
    ]
    
    # Random starts
    for _ in range(30):
        x0 = np.array([
            rng.uniform(*param_bounds[0]),
            rng.uniform(*param_bounds[1]),
            rng.uniform(*param_bounds[2])
        ])
        starts.append(x0)
        
    for x0 in starts:
        try:
            res = minimize(
                _objective_from_params,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
                # Convert back to a
                a0 = math.exp(res.x[0])
                a4 = res.x[1] * a0
                a8 = res.x[2] * a0
                best_a = np.array([a0, a4, a8])
        except Exception:
            pass
    
    # Final polish with L-BFGS-B on direct coefficients
    if best_a is not None and best_val < 1e100:
        try:
            res_final = minimize(
                _objective_direct,
                best_a,
                method='L-BFGS-B',
                bounds=[(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)],
                options={
                    'maxiter': 2000,
                    'ftol': 1e-15,
                    'gtol': 1e-10,
                }
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_a = res_final.x
        except Exception:
            pass
            
    # Additional local search around best solution
    if best_a is not None:
        for _ in range(10):
            perturbation = rng.normal(0, 0.01, size=3)
            x_try = best_a + perturbation
            try:
                res_local = minimize(
                    _objective_direct,
                    x_try,
                    method='L-BFGS-B',
                    bounds=[(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)],
                    options={
                        'maxiter': 1000,
                        'ftol': 1e-14,
                        'gtol': 1e-10,
                    }
                )
                if res_local.fun < best_val:
                    best_val = res_local.fun
                    best_a = res_local.x
            except Exception:
                pass
                
    return best_a
# EVOLVE-BLOCK-END
