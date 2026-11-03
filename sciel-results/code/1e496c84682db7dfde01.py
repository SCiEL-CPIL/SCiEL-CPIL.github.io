"""New approach: Use variable transformation and differential evolution for global optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import math

# ---------------------------------------------------------------------------
# Precomputation
# ---------------------------------------------------------------------------

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

# Dense grid for all norms - use enough points for accuracy
N_GRID = 5000
x_grid = np.linspace(-8.0, 8.0, N_GRID)
dx = x_grid[1] - x_grid[0]
exp_grid = np.exp(-0.5 * x_grid**2)

H0_grid = _hermite_values(0, x_grid)
H4_grid = _hermite_values(4, x_grid)
H8_grid = _hermite_values(8, x_grid)
H12_grid = _hermite_values(12, x_grid)

# Values at 0 for constraint g(0)=0
H0_0 = 1.0
H4_0 = -3.0
H8_0 = 105.0
H12_0 = -10395.0

# Constants for L2 norm
C0 = 2**0 * math.factorial(0)
C4 = 2**4 * math.factorial(4)
C8 = 2**8 * math.factorial(8)
C12 = 2**12 * math.factorial(12)
SQRT_PI = math.sqrt(math.pi)

def _compute_objective(a):
    """
    Compute C4 = ||g||_1 * ||g||_inf / ||g||_2^2
    """
    a0, a4, a8 = a
    
    # Calculate a12 from constraint g(0) = 0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # L2 norm squared (analytic)
    norm2_sq = SQRT_PI * (a0**2 * C0 + a4**2 * C4 + a8**2 * C8 + a12**2 * C12)
    
    if norm2_sq < 1e-15:
        return 1e100
        
    # Evaluate P(x) on grid
    P_grid = a0 * H0_grid + a4 * H4_grid + a8 * H8_grid + a12 * H12_grid
    g_grid = P_grid * exp_grid
    
    # L_inf norm
    l_inf = np.max(np.abs(g_grid))
    if l_inf < 1e-100:
        return 1e100
        
    # L1 norm using trapezoidal rule
    l1_val = np.trapz(np.abs(g_grid), x_grid)
    
    if l1_val < 1e-100:
        return 1e100
        
    return (l1_val * l_inf) / norm2_sq

def construct():
    """
    Return the three free Hermite coefficients (degrees 0, 4, 8).
    Uses differential evolution for global search, then local refinement.
    """
    # Use a clever parameterization: normalize by a0 (if nonzero)
    # Let a4 = a0 * t1, a8 = a0 * t2
    # Then a12 = -a0 * (1 - 3*t1 + 105*t2) / (-10395) = a0 * (1 - 3*t1 + 105*t2) / 10395
    
    def _objective_normalized(t):
        """Objective in terms of ratios t1 = a4/a0, t2 = a8/a0"""
        t1, t2 = t
        a0 = 1.0  # Scale doesn't matter for the ratio
        a4 = a0 * t1
        a8 = a0 * t2
        return _compute_objective(np.array([a0, a4, a8]))
    
    # Global search using differential evolution
    bounds_de = [
        (-10.0, 10.0),   # t1 = a4/a0
        (-5.0, 5.0),     # t2 = a8/a0
    ]
    
    best_val = 1e100
    best_t = None
    
    try:
        de_result = differential_evolution(
            _objective_normalized,
            bounds_de,
            seed=42,
            maxiter=200,
            tol=1e-12,
            popsize=30,
            mutation=(0.3, 1.0),
            recombination=0.7,
            polish=True,
            init='latinhypercube'
        )
        
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_t = de_result.x
    except Exception:
        pass
    
    # Local refinement from multiple starting points
    rng = np.random.default_rng(123)
    
    # Generate diverse starting points
    starts = []
    if best_t is not None:
        starts.append(best_t)
    
    # Add some heuristic starting points
    for _ in range(50):
        t1 = rng.uniform(-5.0, 5.0)
        t2 = rng.uniform(-2.0, 2.0)
        starts.append(np.array([t1, t2]))
    
    # Also try some structured points
    for t1 in np.linspace(-3, 3, 10):
        for t2 in np.linspace(-1, 1, 5):
            starts.append(np.array([t1, t2]))
    
    best_t_local = None
    best_val_local = 1e100
    
    for x0 in starts:
        try:
            res = minimize(
                _objective_normalized,
                x0,
                method='L-BFGS-B',
                bounds=bounds_de,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_val_local:
                best_val_local = res.fun
                best_t_local = res.x
        except Exception:
            pass
    
    # Final polish
    if best_t_local is not None:
        try:
            res_final = minimize(
                _objective_normalized,
                best_t_local,
                method='L-BFGS-B',
                bounds=bounds_de,
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            if res_final.fun < best_val_local:
                best_val_local = res_final.fun
                best_t_local = res_final.x
        except Exception:
            pass
    
    # Use the best result
    if best_t_local is not None and best_val_local < best_val:
        best_t = best_t_local
        best_val = best_val_local
    
    if best_t is None:
        # Fallback
        return np.array([1.0, 0.0, 0.0])
    
    # Convert back to original parameters
    a0 = 1.0
    a4 = a0 * best_t[0]
    a8 = a0 * best_t[1]
    
    return np.array([a0, a4, a8])
# EVOLVE-BLOCK-END
