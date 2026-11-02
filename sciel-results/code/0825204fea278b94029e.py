"""Program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import eval_herm
from scipy.optimize import minimize, differential_evolution

# Physicists' Hermite polynomials: H_0, H_4, H_8, H_12 at 0
H0_0 = 1.0
H4_0 = -12.0
H8_0 = 1680.0
H12_0 = -103950.0

# Pre-compute the Hermite values on a dense grid for fast evaluation
x_grid = np.linspace(-12.0, 12.0, 200001)
w = np.exp(-x_grid**2)  # Gaussian weight for the inner product

# Hermite polynomials on the grid (physicists' convention)
H0 = np.ones_like(x_grid)
H4 = eval_herm(4, x_grid)
H8 = eval_herm(8, x_grid)
H12 = eval_herm(12, x_grid)

# Pre-compute weighted basis functions and their derivatives
# g(x) = P(x) * exp(-x^2/2), where P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
# We need M0 = ∫|g|^2 dx = ∫ P^2 * exp(-x^2) dx
#        M2 = ∫ x^2 |g|^2 dx = ∫ x^2 P^2 * exp(-x^2) dx
#        A(g) = sqrt(M2 / M0)
#        A(Fourier(g)) = sqrt( ∫(g')^2 dx / ∫|g|^2 dx )
#        ∫(g')^2 dx = ∫ (P' - xP)^2 * exp(-x^2) dx

# For efficiency, pre-compute the basis for P and P'
# P = c0*H0 + c4*H4 + c8*H8 + c12*H12
# P' = c0*0 + c4*8*H3 + c8*16*H7 + c12*24*H11
H3 = eval_herm(3, x_grid)
H7 = eval_herm(7, x_grid)
H11 = eval_herm(11, x_grid)

# Weighted inner product components
# For M0: ∫ P^2 * w dx
# We can express this as a quadratic form: P = B @ c, where B is 4 x N
# M0 = (B @ c)^T * w * (B @ c) = c^T (B^T * diag(w) * B) c
# Let's pre-compute the matrix A0 = B^T * diag(w) * B
B = np.column_stack([H0, H4, H8, H12])  # shape (N, 4)
dw = w * np.ones_like(x_grid)  # weight
A0 = B.T @ (B * dw[:, None])  # (4,4) matrix for M0

# For M2: ∫ x^2 P^2 * w dx
A2 = B.T @ (B * (x_grid**2 * dw)[:, None])  # (4,4) matrix for M2

# For ∫(g')^2 dx = ∫ (P' - xP)^2 * w dx
# Let Q = P' - xP
# P' = Bp @ c, where Bp columns are [0, 8*H3, 16*H7, 24*H11]
Bp = np.column_stack([np.zeros_like(x_grid), 8*H3, 16*H7, 24*H11])
# xP = x * (B @ c)
# Q = (Bp - x[:,None]*B) @ c
Bq = Bp - x_grid[:, None] * B  # shape (N, 4)
Aq = Bq.T @ (Bq * dw[:, None])  # (4,4) matrix for ∫(g')^2 dx

# The uncertainty product squared is:
# (A(g) * A(Fourier(g)))^2 = (M2/M0) * (Aq_val / M0) = (M2 * Aq_val) / M0^2
# where M0 = c^T A0 c, M2 = c^T A2 c, Aq_val = c^T Aq c

# Constraint: g(0) = 0 => P(0) = 0 => c0*H0_0 + c4*H4_0 + c8*H8_0 + c12*H12_0 = 0
# => c12 = -(c0*1.0 + c4*(-12.0) + c8*1680.0) / (-103950.0)
# => c12 = (c0 - 12*c4 + 1680*c8) / 103950.0

def get_full_coefficients(c0, c4, c8):
    """Return the full 4-vector of coefficients given the 3 free ones."""
    c12 = (c0 - 12.0*c4 + 1680.0*c8) / 103950.0
    return np.array([c0, c4, c8, c12])

def objective(c_free):
    """Compute the uncertainty product A(g)*A(Fourier(g)) for the given 3 free coefficients."""
    c0, c4, c8 = c_free
    c = get_full_coefficients(c0, c4, c8)
    
    M0 = c @ A0 @ c
    if M0 <= 1e-30:
        return 1e30
    M2 = c @ A2 @ c
    Aq_val = c @ Aq @ c
    
    # Ensure non-negative due to numerical issues
    M2 = max(M2, 0.0)
    Aq_val = max(Aq_val, 0.0)
    
    prod_sq = (M2 * Aq_val) / (M0 * M0)
    if prod_sq < 0:
        return 1e30
    return np.sqrt(prod_sq)

def construct():
    # Multi-start optimization
    best_val = np.inf
    best_c = None
    
    # Start from the seed
    seed_c = np.array([-113/100, 1/25, 1/3240])
    val = objective(seed_c)
    if val < best_val:
        best_val = val
        best_c = seed_c.copy()
    
    # Try differential evolution for global search
    bounds = [(-50, 50), (-50, 50), (-50, 50)]
    try:
        result = differential_evolution(objective, bounds, seed=42, maxiter=50, popsize=15, tol=1e-12)
        if result.fun < best_val:
            best_val = result.fun
            best_c = result.x.copy()
    except Exception:
        pass
    
    # Refine with Nelder-Mead
    if best_c is not None:
        try:
            result = minimize(objective, best_c, method='Nelder-Mead', options={'xatol': 1e-14, 'fatol': 1e-14, 'maxiter': 10000, 'maxfev': 10000})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    # Try a few random starts with L-BFGS-B
    np.random.seed(123)
    for _ in range(20):
        c_start = np.random.uniform(-10, 10, size=3)
        try:
            result = minimize(objective, c_start, method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-14})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except Exception:
            pass
    
    if best_c is None:
        return seed_c
    
    return best_c
# EVOLVE-BLOCK-END
