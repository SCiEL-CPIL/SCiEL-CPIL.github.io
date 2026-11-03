"""Rewritten program: uses fixed Gauss-Hermite quadrature and dense grid sampling for fast, stable evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

# ---------------------------------------------------------------------------
# Precomputation for fast evaluation
# ---------------------------------------------------------------------------

# We need to evaluate H_k(x) for k in {0, 4, 8, 12}
# We will use Gauss-Hermite quadrature for L1 and L2 (L2 is analytic anyway)
# and a dense grid for L_inf.

# Number of GH nodes
N_GH = 200
# GH nodes and weights for integral \int f(x) e^{-x^2} dx
# Note: Standard GH is for e^{-x^2}.
# Our L1 integral is \int |P(x)| e^{-x^2/2} dx.
# Let x = sqrt(2) u. dx = sqrt(2) du.
# L1 = sqrt(2) \int |P(sqrt(2) u)| e^{-u^2} du.
# So we evaluate P at x_i = sqrt(2) u_i where u_i are standard GH nodes.
# And weight is sqrt(2) w_i.

u_nodes, u_weights = np.polynomial.hermite.hermgauss(N_GH)
x_nodes = np.sqrt(2.0) * u_nodes
w_nodes = np.sqrt(2.0) * u_weights

# Precompute H_k(x_nodes) for k=0,4,8,12
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

H0_nodes = _hermite_values(0, x_nodes)
H4_nodes = _hermite_values(4, x_nodes)
H8_nodes = _hermite_values(8, x_nodes)
H12_nodes = _hermite_values(12, x_nodes)

# Dense grid for L_inf
# The function decays as e^{-x^2/2}. Range [-10, 10] is sufficient.
N_GRID = 2000
x_grid = np.linspace(-10.0, 10.0, N_GRID)
exp_grid = np.exp(-0.5 * x_grid**2)

H0_grid = _hermite_values(0, x_grid)
H4_grid = _hermite_values(4, x_grid)
H8_grid = _hermite_values(8, x_grid)
H12_grid = _hermite_values(12, x_grid)

# Constants for L2 norm
# ||g||_2^2 = sqrt(pi) * sum a_k^2 2^k k!
C0 = 2**0 * math.factorial(0)
C4 = 2**4 * math.factorial(4)
C8 = 2**8 * math.factorial(8)
C12 = 2**12 * math.factorial(12)
SQRT_PI = math.sqrt(math.pi)

# Values at 0 for constraint g(0)=0
H0_0 = 1.0
H4_0 = -3.0
H8_0 = 105.0
H12_0 = -10395.0

def _objective(a):
    """
    Compute the upper bound C4 = ||g||_1 * ||g||_inf / ||g||_2^2
    """
    a0, a4, a8 = a
    
    # Calculate a12
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # 1. Analytic L2 norm squared
    norm2_sq = SQRT_PI * (a0**2 * C0 + a4**2 * C4 + a8**2 * C8 + a12**2 * C12)
    
    if norm2_sq < 1e-12:
        return 1e100
        
    # 2. L1 norm using GH quadrature
    # P(x) = a0 H0 + a4 H4 + a8 H8 + a12 H12
    # Evaluate at x_nodes
    P_nodes = a0 * H0_nodes + a4 * H4_nodes + a8 * H8_nodes + a12 * H12_nodes
    
    # L1 = sum w_i |P(x_i)|
    l1_val = np.sum(w_nodes * np.abs(P_nodes))
    
    # 3. L_inf norm using dense grid
    P_grid = a0 * H0_grid + a4 * H4_grid + a8 * H8_grid + a12 * H12_grid
    g_grid = P_grid * exp_grid
    max_val = np.max(np.abs(g_grid))
    
    if max_val < 1e-100:
        return 1e100
        
    # Calculate bound
    bound = (l1_val * max_val) / norm2_sq
    return bound

def construct():
    """
    Return the three free Hermite coefficients (degrees 0, 4, 8).
    Uses Multi-start L-BFGS-B optimization.
    """
    # Bounds for a0, a4, a8
    # The optimal solution is likely small.
    bounds = [
        (-5.0, 5.0),
        (-1.0, 1.0),
        (-0.1, 0.1)
    ]
    
    best_val = 1e100
    best_x = np.array([0.0, 0.0, 0.0])
    
    # Multi-start strategy
    rng = np.random.default_rng(42)
    
    # Try a few deterministic starts + random
    starts = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.01]),
    ]
    
    for _ in range(20):
        x0 = np.array([
            rng.uniform(bounds[0][0], bounds[0][1]),
            rng.uniform(bounds[1][0], bounds[1][1]),
            rng.uniform(bounds[2][0], bounds[2][1])
        ])
        starts.append(x0)
        
    for x0 in starts:
        try:
            res = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                }
            )
            
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
            
    # Final polish
    try:
        res_final = minimize(
            _objective,
            best_x,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 2000,
                'ftol': 1e-15,
                'gtol': 1e-10,
            }
        )
        if res_final.fun < best_val:
            best_val = res_final.fun
            best_x = res_final.x
    except Exception:
        pass
        
    return best_x
# EVOLVE-BLOCK-END
