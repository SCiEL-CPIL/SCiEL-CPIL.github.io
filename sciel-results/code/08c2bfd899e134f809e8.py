"""
Search for optimal Hermite polynomial coefficients (degrees 0, 4, 8)
to minimize the upper bound on A(f)*A(Fourier(f)) for the uncertainty inequality.

The test function is g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
where a12 is chosen so that g(0) = 0.

We use scipy's optimization to minimize the upper bound.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
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
    # Start from the seed solution
    x0 = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    best_x = x0.copy()
    best_val = _objective(best_x)
    
    # Generate diverse starting points
    np.random.seed(42)
    starts = [x0]
    
    # Random perturbations
    for _ in range(50):
        scale = 10 ** np.random.uniform(-3, 1)
        x_trial = x0 + np.random.randn(3) * scale
        starts.append(x_trial)
        
    # Some structured starts
    starts.append([1.0, -0.1, 0.01])
    starts.append([1.0, 0.1, -0.01])
    starts.append([0.0, 1.0, 0.0])
    starts.append([0.0, 0.0, 1.0])
    
    for start in starts:
        try:
            val = _objective(start)
            if val < best_val:
                best_val = val
                best_x = start.copy()
        except Exception:
            pass
            
    # Use Nelder-Mead for local optimization from best start
    try:
        result = minimize(_objective, best_x, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
        
    # Try BFGS if gradients are well-behaved (analytic function is smooth)
    try:
        result2 = minimize(_objective, best_x, method='BFGS',
                           options={'maxiter': 500, 'gtol': 1e-8})
        if result2.fun < best_val:
            best_val = result2.fun
            best_x = result2.x
    except Exception:
        pass
        
    return np.array(best_x)
# EVOLVE-BLOCK-END
