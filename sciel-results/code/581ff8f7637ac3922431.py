# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_hermite, factorial


def _compute_hermite_moments():
    """
    Precompute the matrix C_{k,l} = integral x^2 H_k(x) H_l(x) exp(-x^2) dx
    for k, l in {0, 4, 8, 12}.
    """
    indices = [0, 4, 8, 12]
    n = len(indices)
    C = np.zeros((n, n))
    
    # Formula for C_{k,l}:
    # Non-zero if l == k, l == k+2, or l == k-2.
    # C_{k, k+2} = 0.25 * 2^(k+2) * (k+2)! * sqrt(pi)
    # C_{k, k}   = -(k + 0.5) * 2^k * k! * sqrt(pi)
    # C_{k, k-2} = k*(k-1) * 2^(k-2) * (k-2)! * sqrt(pi)
    
    sqrt_pi = np.sqrt(np.pi)
    
    for i, k in enumerate(indices):
        for j, l in enumerate(indices):
            if l == k + 2:
                C[i, j] = 0.25 * (2 ** (k + 2)) * factorial(k + 2) * sqrt_pi
            elif l == k:
                C[i, j] = -(k + 0.5) * (2 ** k) * factorial(k) * sqrt_pi
            elif l == k - 2:
                C[i, j] = k * (k - 1) * (2 ** (k - 2)) * factorial(k - 2) * sqrt_pi
    
    return C, indices


# Precompute constants
C_MAT, INDICES = _compute_hermite_moments()
# Indices map: 0->0, 1->4, 2->8, 3->12
# We only need to optimize a0, a4, a8. a12 is determined by g(0)=0.
# g(0) = a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
# H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=4790016
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 4790016.0

# Weights for N = sum a_k^2 * 2^k * k! * sqrt(pi)
# Let W_k = 2^k * k! * sqrt(pi)
W = np.array([
    (2 ** 0) * factorial(0) * np.sqrt(np.pi),
    (2 ** 4) * factorial(4) * np.sqrt(np.pi),
    (2 ** 8) * factorial(8) * np.sqrt(np.pi),
    (2 ** 12) * factorial(12) * np.sqrt(np.pi)
])


def _compute_bound_analytic(a0, a4, a8):
    """
    Compute the uncertainty product A(g)*A(hat{g}) analytically.
    """
    # Calculate a12
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    a = np.array([a0, a4, a8, a12])
    
    # N = ||g||^2 = sum a_k^2 * W_k
    N = np.sum(a**2 * W)
    if N < 1e-15:
        return np.inf
    
    # M_x = sum a_k a_l C_{k,l}
    M_x = a @ C_MAT @ a
    
    # M_xi = sum a_k a_l i^(k-l) C_{k,l}
    # Since C_{k,l} is 0 unless k=l or |k-l|=2.
    # If k=l, i^0 = 1.
    # If |k-l|=2, i^+/-2 = -1.
    # So M_xi = sum a_k^2 C_{k,k} - sum_{|k-l|=2} a_k a_l C_{k,l}
    
    M_xi = 0.0
    for i in range(4):
        for j in range(4):
            if i == j:
                M_xi += a[i] * a[j] * C_MAT[i, j]
            elif abs(INDICES[i] - INDICES[j]) == 2:
                M_xi -= a[i] * a[j] * C_MAT[i, j]
            # else 0
    
    # Variance in x: A(g) = M_x / N (since E[x]=0 by symmetry)
    # Variance in xi: A(hat{g}) = M_xi / N (since E[xi]=0 by symmetry)
    # Product: (M_x / N) * (M_xi / N) = M_x * M_xi / N^2
    
    if M_x < 0 or M_xi < 0:
        # Should not happen for valid functions, but guard against numerical issues
        return np.inf
        
    return (M_x * M_xi) / (N * N)


def _objective(coeffs):
    """Objective function for optimization."""
    a0, a4, a8 = coeffs
    try:
        val = _compute_bound_analytic(a0, a4, a8)
        if np.isnan(val) or np.isinf(val):
            return np.inf
        return val
    except Exception:
        return np.inf


def construct():
    """
    Find optimal coefficients a0, a4, a8 to minimize the uncertainty bound.
    Returns np.array([a0, a4, a8]).
    """
    # Define bounds for the coefficients.
    # The seed solution is around [-1.13, 0.04, 0.0003].
    # We allow a wider range to ensure global search capability.
    # a0 is O(1), a4 is O(0.1), a8 is O(0.001).
    # Let's use bounds that are generous but not infinite to help DE.
    bounds = [(-10.0, 10.0), (-1.0, 1.0), (-0.1, 0.1)]
    
    # 1. Global Optimization using Differential Evolution
    # DE is robust and does not require gradients.
    # 'polys' strategy is often good for smooth functions.
    try:
        de_result = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            tol=1e-8, 
            polish=False, # We will polish manually with L-BFGS-B
            init='sobol' # Sobol sequence for better initial distribution
        )
        best_x = de_result.x
        best_val = de_result.fun
    except Exception:
        # Fallback to seed if DE fails
        best_x = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        best_val = _objective(best_x)
        
    # 2. Local Refinement using L-BFGS-B
    # L-BFGS-B is efficient for bound-constrained problems and uses limited memory.
    try:
        lbfgs_result = minimize(
            _objective, 
            best_x, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if lbfgs_result.fun < best_val:
            best_val = lbfgs_result.fun
            best_x = lbfgs_result.x
    except Exception:
        pass
        
    # 3. Additional local refinement with Nelder-Mead if L-BFGS-B didn't improve much
    # Sometimes N-M can find slightly better spots if L-BFGS-B gets stuck in a shallow local min
    try:
        nm_result = minimize(
            _objective, 
            best_x, 
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if nm_result.fun < best_val:
            best_val = nm_result.fun
            best_x = nm_result.x
    except Exception:
        pass

    return np.array(best_x)
# EVOLVE-BLOCK-END
