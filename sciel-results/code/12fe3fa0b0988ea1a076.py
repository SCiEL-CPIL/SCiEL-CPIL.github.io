"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _laguerre_poly_eval(x, m, n):
    """Evaluate the generalized Laguerre polynomial L_m^(n)(x) at points x."""
    from scipy.special import eval_genlaguerre
    return eval_genlaguerre(m, n, x)

def _compute_c4_bound(roots, m=12, n=1):
    """
    Compute the C4 upper bound from the Laguerre polynomial construction.
    The bound is related to 1/sqrt(||P||^2) where P is the polynomial
    with double roots at the given locations.
    """
    x = np.array(roots)
    
    # Build polynomial with double roots: P(t) = prod_i (t - x_i)^2 * L_m^(n)(t)
    # The norm involves an integral against the Laguerre weight
    # For the Fourier sign uncertainty, we need the L2 norm in a specific sense
    
    # Using the Laguerre polynomial framework:
    # The bound involves the ratio of norms
    # Simplified: minimize the condition number of the evaluation matrix
    
    # More direct approach: compute the actual bound via quadrature
    # The C4 bound is approximately 1 / sqrt( <P, P> ) where <.,.> is the inner product
    
    # For practical computation, use the Vandermonde-based proxy which correlates
    # with the actual bound, but weighted by the Laguerre weight function
    
    # Weight function for Laguerre: w(t) = t^n * exp(-t)
    # We want to minimize the condition number in this weighted space
    
    # Construct weighted Vandermonde
    t = x
    w = np.sqrt(t**n * np.exp(-t))
    
    # V[i,j] = t[i]^j * w[i]
    V = np.vander(t, N=m+1, increasing=True) * w[:, np.newaxis]
    
    # Condition number
    cond = np.linalg.cond(V)
    
    # The bound is inversely related to the minimum singular value
    # Lower condition number -> better (lower) bound
    return cond

def _dyson_equilibrium_roots(m=12, n=1, num_iter=1000, tol=1e-10):
    """
    Solve for the equilibrium of a Dyson gas with repulsion exponent 2
    and linear confining potential, then map to Laguerre zero positions.
    
    The mean-field equation for Laguerre zeros in the asymptotic limit:
    For n zeros of L_m^(alpha)(x), the density satisfies:
    ∫ ρ(y) / (x-y) dy + alpha/(2x) - 1/2 = 0 (for x in support)
    
    For finite m, we use the discrete version:
    The zeros x_1 < x_2 < ... < x_m satisfy:
    sum_{j != i} 1/(x_i - x_j) + alpha/(2*x_i) - 1/2 = 0
    
    We solve this system via Newton's method on the log-scale.
    """
    # Initial guess: use the known asymptotic distribution
    # For Laguerre L_m^(n), zeros are approximately distributed as:
    # x_i ≈ 4*m*(i/m)^2 * (1 + O(1/m)) for i near m
    # More precisely, the support is [0, 4m + 2alpha + 2]
    
    # Better initial guess: use the actual Laguerre zeros as starting point
    from scipy.special import roots_genlaguerre
    laguerre_zeros, _ = roots_genlaguerre(m, n)
    
    # Scale: our roots should be positive and well-separated
    # The Laguerre zeros are already positive, use them as initial guess
    x = laguerre_zeros.copy()
    
    # Newton iteration on the discrete Dyson equation
    for it in range(num_iter):
        # Compute forces
        forces = np.zeros(m)
        for i in range(m):
            # Repulsion from other particles
            repulsion = 0.0
            for j in range(m):
                if j != i:
                    repulsion += 1.0 / (x[i] - x[j])
            
            # Confining potential: alpha/(2*x) - 1/2
            # For Laguerre, the potential is -1/2 + n/(2x)
            confinement = n / (2.0 * x[i]) - 0.5
            
            forces[i] = repulsion + confinement
        
        # Newton step: x_new = x - F / F'
        # F' = -sum_{j!=i} 1/(x_i-x_j)^2 - n/(2*x_i^2)
        F_prime = np.zeros(m)
        for i in range(m):
            fp = 0.0
            for j in range(m):
                if j != i:
                    fp -= 1.0 / (x[i] - x[j])**2
            fp -= n / (2.0 * x[i]**2)
            F_prime[i] = fp
        
        # Update
        step = forces / F_prime
        x_new = x - step
        
        # Ensure positivity and ordering
        x_new = np.maximum(x_new, 1e-6)
        x_new = np.sort(x_new)
        
        # Check convergence
        if np.max(np.abs(x_new - x)) < tol:
            x = x_new
            break
        
        x = x_new
    
    return x

def _geometric_parameterization(log_r0, log_q, m=12):
    """Convert geometric parameters to root positions: r_i = r0 * q^i"""
    r0 = np.exp(log_r0)
    q = np.exp(log_q)
    roots = r0 * (q ** np.arange(m))
    return roots

def _c4_bound_geometric(params, m=12, n=1):
    """Objective function for geometric parameterization."""
    log_r0, log_q = params
    roots = _geometric_parameterization(log_r0, log_q, m)
    
    # Ensure all roots are positive
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return 1e10
    
    # Compute the bound
    return _compute_c4_bound(roots, m, n)

def construct():
    m = 12
    n = 1
    
    # Deadline guard
    start_time = time.time()
    deadline = 630.0  # seconds
    safety_margin = 30.0
    
    # Strategy 1: Dyson gas equilibrium
    try:
        dyson_roots = _dyson_equilibrium_roots(m, n, num_iter=500, tol=1e-12)
        if np.all(dyson_roots > 0):
            dyson_score = _compute_c4_bound(dyson_roots, m, n)
        else:
            dyson_score = 1e10
    except Exception:
        dyson_roots = None
        dyson_score = 1e10
    
    # Strategy 2: Geometric parameterization with optimization
    # Initial guesses based on known good configurations
    best_geometric_roots = None
    best_geometric_score = 1e10
    
    # Multiple starting points
    initial_params = [
        (np.log(4.0), np.log(1.0)),    # Uniform-like
        (np.log(2.0), np.log(1.2)),    # Growing geometric
        (np.log(6.0), np.log(1.1)),    # Moderate growth
        (np.log(3.0), np.log(1.15)),   # Another variant
        (np.log(5.0), np.log(1.05)),   # Slow growth
    ]
    
    for x0 in initial_params:
        if time.time() - start_time > deadline - safety_margin:
            break
        
        try:
            res = minimize(_c4_bound_geometric, x0, method='L-BFGS-B',
                          bounds=[(np.log(0.1), np.log(100.0)), (np.log(0.5), np.log(3.0))],
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < best_geometric_score:
                best_geometric_score = res.fun
                best_geometric_roots = _geometric_parameterization(res.x[0], res.x[1], m)
        except Exception:
            continue
    
    # Strategy 3: Free parameter optimization with sorting
    best_free_roots = None
    best_free_score = 1e10
    
    if time.time() - start_time < deadline - safety_margin:
        # Use Dyson roots or geometric roots as initial guess
        if dyson_roots is not None and np.all(dyson_roots > 0):
            x0_free = dyson_roots.copy()
        elif best_geometric_roots is not None:
            x0_free = best_geometric_roots.copy()
        else:
            x0_free = np.array([4.0 * (i + 1) for i in range(m)])
        
        def objective_free(params):
            roots = np.sort(np.abs(params))
            if np.any(roots < 1e-6):
                return 1e10
            return _compute_c4_bound(roots, m, n)
        
        try:
            res = minimize(objective_free, x0_free, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_free_score:
                best_free_score = res.fun
                best_free_roots = np.sort(np.abs(res.x))
        except Exception:
            pass
    
    # Select the best result
    candidates = []
    if dyson_roots is not None and np.all(dyson_roots > 0):
        candidates.append((dyson_score, dyson_roots))
    if best_geometric_roots is not None:
        candidates.append((best_geometric_score, best_geometric_roots))
    if best_free_roots is not None:
        candidates.append((best_free_score, best_free_roots))
    
    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_score, best_roots = candidates[0]
    else:
        # Fallback to seed
        best_roots = np.array([4.0 * (i + 1) for i in range(m)])
    
    # Ensure all roots are positive
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = np.sort(best_roots)
    
    return list(best_roots)
# EVOLVE-BLOCK-END
