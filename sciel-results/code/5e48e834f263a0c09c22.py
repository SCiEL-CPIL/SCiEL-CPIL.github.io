"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import factorial, sqrt, pi
from scipy.linalg import eigh

# Pre-compute analytic matrices for the objective function
# We want to minimize Delta_x^2 = <x^2> / <1> for f = g * exp(-x^2/2)
# where g = c0 H0 + c4 H4 + c8 H8 + c12 H12
# <1> = sum c_i c_j M_ij
# <x^2> = sum c_i c_j N_ij
# M_ij = int H_i H_j exp(-x^2) dx
# N_ij = int x^2 H_i H_j exp(-x^2) dx

def _hermite_val(n, x):
    """Compute physicists' Hermite polynomial H_n(x) at scalar x."""
    if n == 0:
        return 1.0
    elif n == 1:
        return 2.0 * x
    H_prev = 1.0
    H_curr = 2.0 * x
    for k in range(2, n + 1):
        H_next = 2.0 * x * H_curr - 2.0 * (k - 1) * H_prev
        H_prev = H_curr
        H_curr = H_next
    return H_curr

# Basis indices: 0, 4, 8, 12
indices = [0, 4, 8, 12]

# Orthogonality: int H_n H_m exp(-x^2) dx = sqrt(pi) 2^n n! delta_nm
# So M is diagonal.
M_diag = np.array([sqrt(pi) * (2**n) * factorial(n) for n in indices])

# N_ij = int x^2 H_i H_j exp(-x^2) dx
# For this basis, N is diagonal: N_ii = i * M_ii
N_diag = np.array([n * m for n, m in zip(indices, M_diag)])

# c12 is determined by c0, c4, c8 to satisfy g(0)=0
H0_0 = _hermite_val(0, 0)   # 1
H4_0 = _hermite_val(4, 0)   # 12
H8_0 = _hermite_val(8, 0)   # 168
H12_0 = _hermite_val(12, 0) # 20901120

def _compute_bound_analytic(c0, c4, c8):
    """Compute the uncertainty bound for given coefficients."""
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    coeffs = np.array([c0, c4, c8, c12])
    
    # Denominator: sum c_i^2 M_ii
    norm_sq = np.sum(coeffs**2 * M_diag)
    if norm_sq <= 1e-12:
        return 1e10
        
    # Numerator: sum c_i^2 N_ii
    x2_expectation = np.sum(coeffs**2 * N_diag)
    
    # Variance Delta_x^2 = <x^2> - <x>^2. <x>=0 due to symmetry.
    # Product Delta_x Delta_xi = Delta_x^2 because Delta_xi = Delta_x.
    return x2_expectation / norm_sq

def construct():
    """Find optimal coefficients for H0, H4, H8 using generalized eigenvalue problem."""
    start_time = time.time()
    deadline = start_time + 150.0  # safety margin
    
    # The constraint g(0) = 0 gives:
    # c0 * H0(0) + c4 * H4(0) + c8 * H8(0) + c12 * H12(0) = 0
    # c12 = -(c0 + 12*c4 + 168*c8) / 20901120
    #
    # Let's define the free variables as u = [c0, c4, c8]
    # Then c12 = a·u where a = [-1/H12_0, -12/H12_0, -168/H12_0]
    #
    # The full coefficient vector is c = [u, a·u] = [I, a; 0, 1] · u
    # More precisely: c = P · u where P is a 4x3 matrix:
    # P = [[1, 0, 0],
    #      [0, 1, 0],
    #      [0, 0, 1],
    #      [-1/H12_0, -12/H12_0, -168/H12_0]]
    
    a_vec = np.array([-1.0/H12_0, -H4_0/H12_0, -H8_0/H12_0])
    
    # Build P matrix (4x3)
    P = np.zeros((4, 3))
    P[0, 0] = 1.0
    P[1, 1] = 1.0
    P[2, 2] = 1.0
    P[3, :] = a_vec
    
    # M_full is diagonal with entries M_diag
    M_full = np.diag(M_diag)
    # N_full is diagonal with entries N_diag
    N_full = np.diag(N_diag)
    
    # Project to 3x3: M_proj = P^T M_full P, N_proj = P^T N_full P
    M_proj = P.T @ M_full @ P
    N_proj = P.T @ N_full @ P
    
    # Solve generalized eigenvalue problem: N_proj v = lambda M_proj v
    # The smallest eigenvalue gives the minimum of the Rayleigh quotient
    try:
        eigvals, eigvecs = eigh(N_proj, M_proj)
        # Sort by eigenvalue (eigh already returns sorted)
        idx = np.argmin(eigvals)
        best_val = eigvals[idx]
        best_u = eigvecs[:, idx]
        
        # Ensure the vector has reasonable scale (the objective is scale-invariant,
        # but we want a specific representative)
        # Normalize so that the largest component has magnitude ~1
        max_comp = np.max(np.abs(best_u))
        if max_comp > 1e-15:
            best_u = best_u / max_comp
            # Make the first non-zero component positive
            for i in range(3):
                if abs(best_u[i]) > 1e-10:
                    if best_u[i] < 0:
                        best_u = -best_u
                    break
    except Exception:
        # Fallback to the seed value
        best_u = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
        best_val = _compute_bound_analytic(*best_u)
    
    # Verify the result
    actual_val = _compute_bound_analytic(*best_u)
    if actual_val < best_val:
        best_val = actual_val
    
    # Local refinement using Nelder-Mead for extra precision
    from scipy.optimize import minimize
    if time.time() < deadline:
        try:
            res = minimize(
                lambda x: _compute_bound_analytic(x[0], x[1], x[2]),
                best_u,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_u = res.x
        except Exception:
            pass
    
    return best_u
# EVOLVE-BLOCK-END
