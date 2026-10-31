import numpy as np
from scipy.optimize import minimize
import time

def _compute_variance_exact(a0, a4, a8, a12):
    """
    Compute the variance of g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2).
    Variance = <x^2> - <x>^2.
    Due to symmetry (only even Hermite polynomials), <x> = 0.
    So Variance = <x^2> = (a^T M a) / (a^T N a).
    """
    coeffs = np.array([a0, a4, a8, a12])
    degrees = np.array([0, 4, 8, 12])
    
    # Norm matrix N: N_kk = sqrt(pi) * 2^k * k!
    # We only need the diagonal for the denominator.
    # N_kk = sqrt(pi) * 2^k * k!
    
    # Second moment matrix M: M_kl = <x^2 H_k H_l>
    # M_kl = sqrt(pi) * 2^l * l! * [ 0.25 * delta_{l, k+2} + (2k+1)/2 * delta_{l, k} + k(k-1)/2 * delta_{l, k-2} ]
    
    # Precompute factorials and powers of 2 for k in {0, 4, 8, 12}
    # k! for k=0,4,8,12
    # 0! = 1
    # 4! = 24
    # 8! = 40320
    # 12! = 479001600
    
    fact = np.array([1, 24, 40320, 479001600], dtype=float)
    pow2 = np.array([1, 16, 256, 4096], dtype=float)
    sqrt_pi = np.sqrt(np.pi)
    
    # N_diag = sqrt(pi) * 2^k * k!
    N_diag = sqrt_pi * pow2 * fact
    
    # Denominator: sum a_k^2 * N_kk
    denom = np.sum(coeffs**2 * N_diag)
    
    if denom < 1e-15:
        return np.inf
        
    # Numerator: sum_{k,l} a_k a_l M_kl
    # M_kl is non-zero only if l = k, k+2, or k-2.
    # Our degrees are 0, 4, 8, 12.
    # Differences between any two degrees in {0,4,8,12} are multiples of 4.
    # So l = k+2 or l = k-2 is NEVER satisfied for any pair in our set.
    # Thus, M_kl = 0 for k != l.
    # The matrix M is diagonal for this specific set of degrees!
    
    # M_kk = sqrt(pi) * 2^k * k! * (2k+1)/2
    M_diag = N_diag * (2 * degrees + 1) / 2.0
    
    numer = np.sum(coeffs**2 * M_diag)
    
    return numer / denom

def _objective_from_angles(theta, phi):
    """
    Map spherical coordinates to coefficients, enforce constraint, compute variance.
    a0 = sin(theta) * cos(phi)
    a4 = sin(theta) * sin(phi)
    a8 = cos(theta)
    
    Constraint: a0*H0(0) + a4*H4(0) + a8*H8(0) + a12*H12(0) = 0
    H0(0) = 1
    H4(0) = -12
    H8(0) = 1680
    H12(0) = -103950
    
    a0 - 12*a4 + 1680*a8 - 103950*a12 = 0
    a12 = (a0 - 12*a4 + 1680*a8) / 103950
    """
    a0 = np.sin(theta) * np.cos(phi)
    a4 = np.sin(theta) * np.sin(phi)
    a8 = np.cos(theta)
    
    a12 = (a0 - 12.0*a4 + 1680.0*a8) / 103950.0
    
    return _compute_variance_exact(a0, a4, a8, a12)

def construct():
    """
    Find the coefficients (a0, a4, a8) that minimize the uncertainty product (variance).
    """
    start_time = time.time()
    deadline = start_time + 180.0 # Leave 30s buffer for the 210s limit
    
    best_val = np.inf
    best_params = None
    
    # The domain is theta in [0, pi], phi in [0, 2*pi).
    # We can use a grid search to find a good starting point, then polish.
    
    # Grid search
    # theta: 0 to pi
    # phi: 0 to 2*pi
    # Let's try a moderately fine grid first.
    n_theta = 200
    n_phi = 200
    
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    # Vectorize the objective calculation for the grid
    # Create meshgrid
    T, P = np.meshgrid(thetas, phis, indexing='ij')
    
    # Calculate a0, a4, a8 for all points
    a0_grid = np.sin(T) * np.cos(P)
    a4_grid = np.sin(T) * np.sin(P)
    a8_grid = np.cos(T)
    
    a12_grid = (a0_grid - 12.0*a4_grid + 1680.0*a8_grid) / 103950.0
    
    # Calculate variance for all points
    # coeffs shape: (n_theta, n_phi, 4)
    coeffs_grid = np.stack([a0_grid, a4_grid, a8_grid, a12_grid], axis=-1)
    
    # Degrees and precomputed values
    degrees = np.array([0, 4, 8, 12])
    fact = np.array([1, 24, 40320, 479001600], dtype=float)
    pow2 = np.array([1, 16, 256, 4096], dtype=float)
    sqrt_pi = np.sqrt(np.pi)
    N_diag = sqrt_pi * pow2 * fact
    M_diag = N_diag * (2 * degrees + 1) / 2.0
    
    # denom = sum(coeffs^2 * N_diag, axis=-1)
    denom_grid = np.sum(coeffs_grid**2 * N_diag, axis=-1)
    numer_grid = np.sum(coeffs_grid**2 * M_diag, axis=-1)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        var_grid = numer_grid / denom_grid
        var_grid[denom_grid < 1e-15] = np.inf
    
    # Find the minimum in the grid
    min_idx = np.unravel_index(np.argmin(var_grid), var_grid.shape)
    best_theta = thetas[min_idx[0]]
    best_phi = phis[min_idx[1]]
    best_val = var_grid[min_idx]
    
    # Refine using local optimization (Nelder-Mead or L-BFGS-B)
    # Since the function is smooth, we can use L-BFGS-B with bounds.
    # Bounds: theta in [0, pi], phi in [0, 2*pi]
    
    def obj_wrapper(x):
        t, p = x
        return _objective_from_angles(t, p)
    
    bounds = [(0, np.pi), (0, 2*np.pi)]
    
    # Perform a local optimization starting from the best grid point
    try:
        res = minimize(obj_wrapper, x0=[best_theta, best_phi], method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
        if res.success and res.fun < best_val:
            best_val = res.fun
            best_theta, best_phi = res.x
    except Exception:
        pass
    
    # Also try a few random starting points to ensure we didn't miss a global minimum
    # The landscape might have multiple local minima.
    np.random.seed(42)
    n_random_starts = 50
    
    for i in range(n_random_starts):
        if time.time() > deadline - 10:
            break
            
        t_rand = np.random.uniform(0, np.pi)
        p_rand = np.random.uniform(0, 2*np.pi)
        
        try:
            res = minimize(obj_wrapper, x0=[t_rand, p_rand], method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_theta, best_phi = res.x
        except Exception:
            continue
    
    # Extract the best coefficients
    a0 = np.sin(best_theta) * np.cos(best_phi)
    a4 = np.sin(best_theta) * np.sin(best_phi)
    a8 = np.cos(best_theta)
    
    return np.array([a0, a4, a8])

# EVOLVE-BLOCK-END
