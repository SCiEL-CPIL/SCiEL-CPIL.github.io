import numpy as np
from scipy.special import genlaguerre
from scipy.optimize import minimize
import time
import math

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    
    Strategy:
    1. Parameterize z_i as cumulative products of ratios: z_i = s_1 * s_2 * ... * s_i * base
       where s_i > 0 are ratios and base scales the whole configuration.
       This gives 12 DOF (11 ratios + 1 scale) vs 1 DOF in the old approach.
    2. Use a fast floating-point estimator for B.
    3. Multi-start optimization with Nelder-Mead.
    4. Return the optimized z_i.
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 550.0  # 550s limit, leaving 80s margin
    
    # Pre-compute basis polynomials L_{2k}^{(-0.5)}(x)
    # Store coefficients in lowest-to-highest order for easier evaluation
    basis_polys = []
    basis_derivs = []
    for k in range(26):
        n = 2 * k
        L_n = genlaguerre(n, -0.5)
        coeffs_h2l = L_n.coeffs  # highest to lowest
        # Convert to lowest to highest
        coeffs_l2h = coeffs_h2l[::-1]
        basis_polys.append(coeffs_l2h)
        # Derivative
        deriv_h2l = np.polyder(coeffs_h2l)
        deriv_l2h = deriv_h2l[::-1]
        basis_derivs.append(deriv_l2h)
    
    # Pre-compute constant terms and linear terms for x=0 conditions
    L_at_0 = np.array([p[-1] if len(p) > 0 else 0.0 for p in basis_polys])
    Lp_at_0 = np.array([p[-1] if len(p) > 1 else 0.0 for p in basis_derivs])
    
    def estimate_B_fast(z_list):
        """Fast floating-point estimate of B."""
        z_arr = np.array(z_list)
        
        # Build A matrix (26x26)
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Row 0: g(0) = 0
        A[0, :] = L_at_0
        b[0] = 0.0
        
        # Row 1: g'(0) = 1
        A[1, :] = Lp_at_0
        b[1] = 1.0
        
        # Rows 2..27: g(z_i) = 0, g'(z_i) = 0
        for i in range(12):
            z = z_arr[i]
            row_g = 2 + 2*i
            row_gp = 3 + 2*i
            
            # Evaluate all basis polynomials at z
            for k in range(26):
                A[row_g, k] = np.polyval(basis_polys[k], z)
                A[row_gp, k] = np.polyval(basis_derivs[k], z)
            
            b[row_g] = 0.0
            b[row_gp] = 0.0
        
        # Solve
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 1e100
        
        # Construct g(x) in lowest-to-highest coefficient form
        g_coeffs = np.zeros(51)
        for k in range(26):
            deg = len(basis_polys[k]) - 1
            # basis_polys[k] is lowest to highest, length deg+1
            g_coeffs[:deg+1] += c[k] * basis_polys[k]
        
        # Construct D(x) = x * prod (x-z_i)^2 in lowest-to-highest form
        # Start with x = [0, 1] (lowest to highest: 0 + 1*x)
        d_coeffs = np.array([0.0, 1.0])
        for z in z_list:
            # (x-z)^2 = z^2 - 2zx + x^2 => [z^2, -2z, 1] in lowest-to-highest
            sq = np.array([z*z, -2*z, 1.0])
            d_coeffs = np.convolve(d_coeffs, sq)
        
        # d_coeffs should have length 25 (degree 24)
        # g_coeffs has length 51 (degree 50)
        # Quotient degree: 50 - 24 = 26
        
        # Convert to highest-to-lowest for np.polydiv
        g_h2l = g_coeffs[::-1]
        d_h2l = d_coeffs[::-1]
        
        try:
            q_h2l, r_h2l = np.polydiv(g_h2l, d_h2l)
        except Exception:
            return 1e100
        
        # Find real roots
        roots = np.roots(q_h2l)
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        real_roots = real_roots[real_roots > 0]
        
        if len(real_roots) == 0:
            return 0.0
            
        return float(np.max(real_roots))
    
    # Parameterization: z_i = base * s_1 * s_2 * ... * s_i
    # where s_i are ratios. We optimize log(s_i) for numerical stability.
    # z_1 = base * s_1
    # z_2 = base * s_1 * s_2
    # ...
    # z_12 = base * s_1 * ... * s_12
    
    def params_to_z(params):
        """Convert parameter vector to z list.
        params = [log_base, log_s1, log_s2, ..., log_s12]
        """
        if len(params) != 13:
            return None
        log_base = params[0]
        log_s = params[1:]
        
        # Compute cumulative products
        cum_log = np.cumsum(log_s)
        z = np.exp(log_base + cum_log)
        
        # Check validity
        if not np.all(np.isfinite(z)):
            return None
        if not np.all(z > 0):
            return None
        # Check distinctness
        if np.any(np.diff(np.sort(z)) < 1e-10):
            return None
            
        return z.tolist()
    
    def objective(params):
        z_list = params_to_z(params)
        if z_list is None:
            return 1e100
        return estimate_B_fast(z_list)
    
    # Initial guess: Bessel zeros z_i = (k*pi)^2
    # In our parameterization: z_i = base * prod_{j=1}^i s_j
    # z_1 = base * s_1 = pi^2
    # z_2 = base * s_1 * s_2 = (2*pi)^2 = 4*pi^2 => s_2 = 4
    # z_3 = base * s_1 * s_2 * s_3 = (3*pi)^2 = 9*pi^2 => s_3 = 9/4
    # In general: s_i = i^2/(i-1)^2 for i >= 2, s_1 = pi^2/base
    
    # Choose base = pi^2, then s_1 = 1, s_i = i^2/(i-1)^2
    base_init = math.pi ** 2
    s_init = [1.0] + [float(i*i)/((i-1)*(i-1)) for i in range(2, 13)]
    
    params_init = [math.log(base_init)] + [math.log(s) for s in s_init]
    
    best_params = params_init.copy()
    best_val = objective(params_init)
    best_z = params_to_z(params_init)
    
    # Multi-start optimization
    np.random.seed(42)
    n_starts = 20
    max_iter_per_start = 100
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
            
        if start_idx == 0:
            p0 = params_init.copy()
        else:
            # Perturb the initial guess
            perturbation = np.random.normal(0, 0.1, size=13)
            p0 = params_init + perturbation
        
        try:
            res = minimize(objective, p0, method='Nelder-Mead',
                           options={'maxiter': max_iter_per_start, 
                                   'xatol': 1e-6, 
                                   'fatol': 1e-6,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
                best_z = params_to_z(best_params)
        except Exception:
            pass
    
    # Final local refinement with tighter tolerances
    if time.time() < deadline - 60:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 200, 
                                   'xatol': 1e-8, 
                                   'fatol': 1e-8,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
                best_z = params_to_z(best_params)
        except Exception:
            pass
    
    # If optimization failed, use initial guess
    if best_z is None:
        best_z = [(k * math.pi) ** 2 for k in range(1, 13)]
    
    # Ensure validity
    if any(not np.isfinite(z) or z <= 0 for z in best_z):
        best_z = [(k * math.pi) ** 2 for k in range(1, 13)]
    
    # Check distinctness
    sorted_z = sorted(best_z)
    if any(sorted_z[i+1] - sorted_z[i] < 1e-10 for i in range(11)):
        best_z = [(k * math.pi) ** 2 for k in range(1, 13)]
    
    return [float(z) for z in best_z]

# EVOLVE-BLOCK-END
