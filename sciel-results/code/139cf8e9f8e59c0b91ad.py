"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import factorial, sqrt, pi
from scipy.optimize import minimize

# Pre-compute analytic quantities for the objective function
# We minimize Delta_x^2 = <x^2> / <1> for f = g * exp(-x^2/2)
# where g = c0*H0 + c4*H4 + c8*H8 + c12*H12
# with c12 determined by g(0)=0

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

# Orthogonality: int H_n H_m exp(-x^2) dx = sqrt(pi) * 2^n * n! * delta_nm
M_diag = np.array([sqrt(pi) * (2**n) * factorial(n) for n in indices])

# N_ii = int x^2 H_n^2 exp(-x^2) dx = n * M_ii
N_diag = np.array([n * m for n, m in zip(indices, M_diag)])

# Values at x=0 for the constraint g(0)=0
H0_0 = _hermite_val(0, 0)    # 1
H4_0 = _hermite_val(4, 0)    # 12
H8_0 = _hermite_val(8, 0)    # 168
H12_0 = _hermite_val(12, 0)  # 20901120


def _compute_bound_analytic(c0, c4, c8):
    """Compute the uncertainty bound for given coefficients."""
    c12 = -(c0 * H0_0 + c4 * H4_0 + c8 * H8_0) / H12_0
    coeffs = np.array([c0, c4, c8, c12])
    
    norm_sq = np.sum(coeffs**2 * M_diag)
    if norm_sq <= 1e-12:
        return 1e10
        
    x2_expectation = np.sum(coeffs**2 * N_diag)
    return x2_expectation / norm_sq


def _objective(params):
    """Objective function for optimization."""
    return _compute_bound_analytic(params[0], params[1], params[2])


def construct():
    """Find optimal coefficients for H0, H4, H8."""
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin
    
    # The constraint g(0) = 0 gives:
    # c0*H0(0) + c4*H4(0) + c8*H8(0) + c12*H12(0) = 0
    # c12 = -(c0 + 12*c4 + 168*c8) / 20901120
    
    # We can use a different approach: parameterize using two free variables
    # since the objective is scale-invariant. Let's use ratios.
    # Let a = c4/c0, b = c8/c0 (assuming c0 != 0)
    # Then c12/c0 = -(1 + 12a + 168b) / H12_0
    
    # Actually, let's just do a multi-start optimization with the full 3 variables
    # The objective is scale-invariant, so we can normalize somehow
    
    # Let's use the fact that the Rayleigh quotient is homogeneous of degree 0
    # We can fix one coefficient and optimize the other two, or use ratios
    
    # Strategy: Use multiple random starting points with different scales
    # and run Nelder-Mead from each
    
    best_val = 1e10
    best_u = None
    
    # Try the analytic solution first (generalized eigenvalue approach)
    try:
        a_vec = np.array([-1.0/H12_0, -H4_0/H12_0, -H8_0/H12_0])
        P = np.zeros((4, 3))
        P[0, 0] = 1.0
        P[1, 1] = 1.0
        P[2, 2] = 1.0
        P[3, :] = a_vec
        
        M_full = np.diag(M_diag)
        N_full = np.diag(N_diag)
        
        M_proj = P.T @ M_full @ P
        N_proj = P.T @ N_full @ P
        
        from scipy.linalg import eigh
        eigvals, eigvecs = eigh(N_proj, M_proj)
        idx = np.argmin(eigvals)
        best_val_eig = eigvals[idx]
        best_u_eig = eigvecs[:, idx]
        
        # Normalize
        max_comp = np.max(np.abs(best_u_eig))
        if max_comp > 1e-15:
            best_u_eig = best_u_eig / max_comp
            for i in range(3):
                if abs(best_u_eig[i]) > 1e-10:
                    if best_u_eig[i] < 0:
                        best_u_eig = -best_u_eig
                    break
        
        actual_val = _compute_bound_analytic(*best_u_eig)
        if actual_val < best_val:
            best_val = actual_val
            best_u = best_u_eig.copy()
    except Exception:
        pass
    
    # Now do multi-start local optimization to see if we can do better
    # or to confirm the analytic result
    
    if best_u is None:
        # Fallback
        best_u = np.array([-1.0, 0.0, 0.0])
        best_val = _compute_bound_analytic(*best_u)
    
    # Generate diverse starting points
    np.random.seed(42)
    n_starts = 50
    starts = []
    
    # Include the current best
    starts.append(best_u.copy())
    
    # Random starts with various scales
    for _ in range(n_starts):
        # Random direction in 3D
        v = np.random.randn(3)
        scale = 10**np.random.uniform(-3, 3)
        starts.append(v * scale)
    
    # Also try some structured starts
    structured = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 1.0]),
        np.array([1.0, 0.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, -1.0]),
    ]
    starts.extend(structured)
    
    for start in starts:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _objective,
                start,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-16, 'fatol': 1e-18, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_u = res.x.copy()
        except Exception:
            continue
    
    # Final verification
    final_val = _compute_bound_analytic(*best_u)
    if final_val < best_val:
        best_val = final_val
    
    return best_u
# EVOLVE-BLOCK-END
