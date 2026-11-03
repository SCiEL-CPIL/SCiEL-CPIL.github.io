"""Rewritten program: uses Differential Evolution for global optimization of the uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
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
    Uses Differential Evolution for global optimization.
    """
    # Bounds for a0, a4, a8
    # Based on the nature of the problem, coefficients are likely small.
    # We use wider bounds than the local optimizer to ensure global search.
    bounds = [
        (-10.0, 10.0),
        (-10.0, 10.0),
        (-10.0, 10.0)
    ]
    
    # Differential Evolution settings
    # popsize: 15 * dim = 45
    # maxiter: 100 (each iteration evaluates popsize * (1 + CR) points roughly)
    # tol: 1e-8
    # seed: 42 for reproducibility
    
    try:
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=200,
            popsize=20,
            tol=1e-8,
            seed=42,
            polish=True,  # Use L-BFGS-B for final polish
            init='latinhypercube'
        )
        
        if result.success:
            return result.x
        else:
            # If DE fails, try a few random starts with L-BFGS-B as fallback
            # or just return the best found by DE
            return result.x
            
    except Exception as e:
        # Fallback to a simple heuristic if DE fails completely
        # This should not happen, but safety first.
        return np.array([0.0, 0.0, 0.0])

# EVOLVE-BLOCK-END
