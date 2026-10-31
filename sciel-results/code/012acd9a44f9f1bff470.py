"""Optimized program for uncertainty inequality test function tuning."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Precompute Hermite polynomials values and integration matrices
# We need H0, H4, H8, H12
# Indices: 0, 1, 2, 3 corresponding to degrees 0, 4, 8, 12
DEGREES = [0, 4, 8, 12]
N_BASIS = len(DEGREES)

def _hermite_prob(x, n):
    """Compute probabilists' Hermite polynomial Hn(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return x
    elif n == 2:
        return x**2 - 1
    elif n == 3:
        return x**3 - 3*x
    elif n == 4:
        return x**4 - 6*x**2 + 3
    elif n == 8:
        return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105
    elif n == 12:
        return x**12 - 66*x**10 + 1540*x**8 - 15400*x**6 + 69300*x**4 - 118800*x**2 + 945
    else:
        raise ValueError(f"Unsupported Hermite degree: {n}")

# Precompute Position Space Matrices
# Weight w_pos(x) = exp(-x^2/2)
# A[i,j] = integral H(DEGREES[i]) H(DEGREES[j]) w_pos(x) dx
# B[i,j] = integral x^2 H(DEGREES[i]) H(DEGREES[j]) w_pos(x) dx

def _precompute_pos_matrices():
    # Use Gauss-Hermite quadrature for high precision
    # np.polynomial.hermite.hermgauss(n) gives nodes and weights for exp(-x^2)
    # We need exp(-x^2/2). Let u = x/sqrt(2). x = u*sqrt(2). dx = sqrt(2) du.
    # w_pos(x) dx = exp(-u^2) sqrt(2) du.
    # So we can use standard hermgauss nodes/weights scaled.
    n_quad = 100
    nodes_u, weights_u = np.polynomial.hermite.hermgauss(n_quad)
    nodes_x = nodes_u * np.sqrt(2)
    weights_x = weights_u * np.sqrt(2)
    
    # Evaluate Hermite polynomials at nodes
    H_vals = np.zeros((N_BASIS, n_quad))
    for i, deg in enumerate(DEGREES):
        H_vals[i] = _hermite_prob(nodes_x, deg)
    
    # A[i,j] = sum_k weights_x[k] * H_vals[i,k] * H_vals[j,k]
    A = np.zeros((N_BASIS, N_BASIS))
    for i in range(N_BASIS):
        for j in range(N_BASIS):
            A[i,j] = np.sum(weights_x * H_vals[i] * H_vals[j])
            
    # B[i,j] = sum_k weights_x[k] * x_k^2 * H_vals[i,k] * H_vals[j,k]
    B = np.zeros((N_BASIS, N_BASIS))
    x_sq = nodes_x**2
    for i in range(N_BASIS):
        for j in range(N_BASIS):
            B[i,j] = np.sum(weights_x * x_sq * H_vals[i] * H_vals[j])
            
    return A, B

# Precompute Frequency Space Matrices
# Weight w_freq(xi) = exp(-2*xi^2)
# Basis V_k(xi) = H_k(xi/sqrt(2))
# M[i,j] = integral V_i(xi) V_j(xi) w_freq(xi) dxi
# N[i,j] = integral xi^2 V_i(xi) V_j(xi) w_freq(xi) dxi

def _precompute_freq_matrices():
    n_quad = 100
    # For exp(-2*xi^2), let u = xi*sqrt(2). xi = u/sqrt(2). dxi = du/sqrt(2).
    # w_freq(xi) dxi = exp(-u^2) du/sqrt(2).
    nodes_u, weights_u = np.polynomial.hermite.hermgauss(n_quad)
    nodes_xi = nodes_u / np.sqrt(2)
    weights_xi = weights_u / np.sqrt(2)
    
    # Evaluate V_k(xi) = H_k(xi/sqrt(2)) = H_k(u/2) ?? No.
    # xi = u/sqrt(2). xi/sqrt(2) = u/2.
    # So we need H_k(u/2).
    V_vals = np.zeros((N_BASIS, n_quad))
    for i, deg in enumerate(DEGREES):
        V_vals[i] = _hermite_prob(nodes_u / 2.0, deg)
        
    M = np.zeros((N_BASIS, N_BASIS))
    for i in range(N_BASIS):
        for j in range(N_BASIS):
            M[i,j] = np.sum(weights_xi * V_vals[i] * V_vals[j])
            
    N_mat = np.zeros((N_BASIS, N_BASIS))
    xi_sq = nodes_xi**2
    for i in range(N_BASIS):
        for j in range(N_BASIS):
            N_mat[i,j] = np.sum(weights_xi * xi_sq * V_vals[i] * V_vals[j])
            
    return M, N_mat

# Global precomputed matrices
A_POS, B_POS = _precompute_pos_matrices()
M_FREQ, N_FREQ = _precompute_freq_matrices()

# Coefficients for frequency space transformation
# c_k = a_k * (-i)^k * (sqrt(2))^k
# We need to handle complex arithmetic in the objective.
# Let's precompute the complex scaling factors for each basis index
COMPLEX_SCALES = []
for deg in DEGREES:
    # (-i)^deg * (sqrt(2))^deg
    # (-i)^0 = 1, (-i)^1 = -i, (-i)^2 = -1, (-i)^3 = i
    # Powers of -i cycle: 1, -i, -1, i
    phase = [1, -1j, -1, 1j][deg % 4]
    scale = phase * (np.sqrt(2) ** deg)
    COMPLEX_SCALES.append(scale)
COMPLEX_SCALES = np.array(COMPLEX_SCALES)

def _compute_bound_analytic(a0, a4, a8):
    """Compute the uncertainty bound using analytic matrices."""
    # Derive a12
    a12 = (a0 + 3*a4 + 105*a8) / 945.0
    a = np.array([a0, a4, a8, a12])
    
    # Position moments
    I0 = np.dot(a, A_POS @ a)
    I2 = np.dot(a, B_POS @ a)
    
    if I0 < 1e-15 or I2 < 1e-15:
        return 1e10
        
    # Frequency coefficients
    c = a * COMPLEX_SCALES
    
    # Frequency moments
    # J0 = sum_{i,j} c_i c_j^* M[i,j]
    # Since M is real symmetric, J0 = real(c^T M c^*) = real( (c^* M c) )?
    # Actually, J0 = c^H M c where c^H is conjugate transpose.
    J0 = np.vdot(c, M_FREQ @ c).real
    J2 = np.vdot(c, N_FREQ @ c).real
    
    if J0 < 1e-15 or J2 < 1e-15:
        return 1e10
        
    A_g = np.sqrt(I2 / I0)
    A_Fg = np.sqrt(J2 / J0)
    
    return A_g * A_Fg

def construct():
    """Find optimal Hermite coefficients to minimize the uncertainty bound."""
    deadline = time.time() + 170.0  # Safety margin
    
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_score = _compute_bound_analytic(*seed)
    best_params = seed.copy()
    
    def objective(params):
        if time.time() > deadline:
            return best_score
        return _compute_bound_analytic(*params)
        
    # Use L-BFGS-B for efficient local optimization
    try:
        if time.time() < deadline - 10.0:
            res = minimize(objective, seed, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
    except Exception:
        pass
        
    # Try a few random restarts if time permits
    np.random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 20.0:
            break
        # Random perturbations
        perturb = np.random.normal(0, 1, 3) * 0.1
        start = seed + perturb
        try:
            res = minimize(objective, start, method='Nelder-Mead', options={'maxiter': 500})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    return best_params

# EVOLVE-BLOCK-END
