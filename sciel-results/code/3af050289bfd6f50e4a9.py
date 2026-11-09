import numpy as np
from scipy.special import genlaguerre
from scipy.optimize import minimize, differential_evolution
import time
import math
import sys

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    
    Strategy:
    1. Use a fast vectorized B estimator.
    2. Multi-start optimization with Powell method (no gradient needed).
    3. Better initial guesses based on asymptotic analysis.
    """
    
    start_time = time.time()
    deadline = start_time + 560.0  # Safety margin
    
    # Pre-compute basis polynomials L_{2k}^{(-0.5)}(x)
    basis_polys = []
    basis_derivs = []
    for k in range(26):
        n = 2 * k
        L_n = genlaguerre(n, -0.5)
        coeffs_h2l = L_n.coeffs
        coeffs_l2h = coeffs_h2l[::-1]
        basis_polys.append(coeffs_l2h)
        deriv_h2l = np.polyder(coeffs_h2l)
        deriv_l2h = deriv_h2l[::-1]
        basis_derivs.append(deriv_l2h)
    
    L_at_0 = np.array([p[-1] if len(p) > 0 else 0.0 for p in basis_polys])
    Lp_at_0 = np.array([p[-1] if len(p) > 1 else 0.0 for p in basis_derivs])
    
    def estimate_B(z_list):
        """Fast estimate of B using floating point."""
        z_arr = np.array(z_list, dtype=np.float64)
        
        # Build A matrix (26x26)
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        A[0, :] = L_at_0
        A[1, :] = Lp_at_0
        b[1] = 1.0
        
        # Evaluate basis at each z_i
        for i in range(12):
            z = z_arr[i]
            row_g = 2 + 2*i
            row_gp = 3 + 2*i
            
            for k in range(26):
                A[row_g, k] = np.polyval(basis_polys[k], z)
                A[row_gp, k] = np.polyval(basis_derivs[k], z)
        
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 1e100
        
        # Construct g(x)
        g_coeffs = np.zeros(51)
        for k in range(26):
            deg = len(basis_polys[k]) - 1
            g_coeffs[:deg+1] += c[k] * basis_polys[k]
        
        # Construct D(x) = x * prod (x-z_i)^2
        d_coeffs = np.array([0.0, 1.0])
        for z in z_arr:
            sq = np.array([z*z, -2*z, 1.0])
            d_coeffs = np.convolve(d_coeffs, sq)
        
        g_h2l = g_coeffs[::-1]
        d_h2l = d_coeffs[::-1]
        
        try:
            q_h2l, r_h2l = np.polydiv(g_h2l, d_h2l)
        except Exception:
            return 1e100
        
        # Find real roots of q
        roots = np.roots(q_h2l)
        real_roots = roots[np.abs(roots.imag) < 1e-8].real
        real_roots = real_roots[real_roots > 0]
        
        if len(real_roots) == 0:
            return 0.0
            
        return float(np.max(real_roots))
    
    def objective(params):
        """Objective function: minimize B."""
        z_list = list(params)
        if any(z <= 0 for z in z_list):
            return 1e100
        if any(not np.isfinite(z) for z in z_list):
            return 1e100
        # Check distinctness
        sorted_z = sorted(z_list)
        for i in range(11):
            if sorted_z[i+1] - sorted_z[i] < 1e-12:
                return 1e100
        return estimate_B(z_list)
    
    # Initial guesses: use different strategies
    initial_guesses = []
    
    # Guess 1: Bessel-like spacing z_i = (c*i)^2
    for c in [0.8, 0.9, 1.0, 1.1, 1.2]:
        z_guess = [(c * i) ** 2 for i in range(1, 13)]
        initial_guesses.append(z_guess)
    
    # Guess 2: Linear spacing z_i = a + b*i
    for a, b in [(1, 1), (1, 2), (2, 1), (0.5, 1.5)]:
        z_guess = [a + b * i for i in range(1, 13)]
        initial_guesses.append(z_guess)
    
    # Guess 3: Geometric spacing
    for ratio in [1.1, 1.2, 1.3, 1.5]:
        z_guess = [1.0 * (ratio ** i) for i in range(12)]
        initial_guesses.append(z_guess)
    
    # Evaluate initial guesses
    best_z = None
    best_val = float('inf')
    
    for z_guess in initial_guesses:
        if time.time() > deadline:
            break
        val = objective(z_guess)
        if val < best_val:
            best_val = val
            best_z = z_guess.copy()
    
    # Local optimization from best initial guess
    if best_z is not None and time.time() < deadline - 30:
        try:
            res = minimize(objective, best_z, method='Powell',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_z = res.x.tolist()
        except Exception:
            pass
    
    # Try a few more local optimizations from perturbed starts
    np.random.seed(42)
    for _ in range(5):
        if time.time() > deadline - 20:
            break
        if best_z is None:
            break
        perturbed = [z * (1 + np.random.normal(0, 0.05)) for z in best_z]
        if any(z <= 0 for z in perturbed):
            continue
        val = objective(perturbed)
        if val < best_val:
            best_val = val
            best_z = perturbed
        # Local refine
        if time.time() < deadline - 15:
            try:
                res = minimize(objective, perturbed, method='Powell',
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < best_val:
                    best_val = res.fun
                    best_z = res.x.tolist()
            except Exception:
                pass
    
    # Final validation
    if best_z is None:
        best_z = [(i) ** 2 for i in range(1, 13)]
    
    # Ensure all positive and distinct
    best_z = [float(z) for z in best_z]
    for i in range(12):
        if best_z[i] <= 0 or not np.isfinite(best_z[i]):
            best_z[i] = (i+1) ** 2
    
    sorted_z = sorted(best_z)
    for i in range(11):
        if sorted_z[i+1] - sorted_z[i] < 1e-10:
            sorted_z[i+1] = sorted_z[i] + 1e-8
    best_z = sorted_z
    
    return best_z

# EVOLVE-BLOCK-END
