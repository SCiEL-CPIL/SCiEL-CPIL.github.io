import numpy as np
from scipy.special import genlaguerre
from scipy.optimize import minimize
import time
import math

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    
    Strategy:
    1. Base structure: Bessel zeros z_i = (k * beta)^2 for k=1..12.
    2. Optimize beta around pi using a floating-point approximation of the task's 
       objective function to find a configuration that minimizes the largest root B 
       of the quotient polynomial q(x).
    3. Return the optimized z_i.
    """
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 500.0  # 500s limit, leaving 100s margin
    
    # Function to estimate B using floating point arithmetic
    # This is a proxy for the exact rational evaluation
    def estimate_B(z_list):
        # z_list is a list of 12 floats
        # 1. Construct g(x) = sum c_k L_{2k}(x)
        # g(0)=0, g'(0)=1, g(z_i)=0, g'(z_i)=0
        
        # Degrees of L are 0, 2, ..., 50. Total 26 terms.
        # Unknowns: c_0, ..., c_25
        
        # We need L_d(x) and L_d'(x) for d=0,2,...,50 at x=0 and x=z_i
        # L_d(0) = binomial(d, 0) * (-1)^0 ? No, L_d^(alpha)(x).
        # L_n^(alpha)(x) = sum_{k=0}^n (-1)^k * binom(n+alpha, n-k) * x^k / k!
        # For alpha = -1/2.
        
        # We can use numpy.polynomial.laguerre or scipy.special.eval_genlaguerre
        # But we need derivatives.
        
        # Let's build the linear system A * c = b
        # A is 26x26.
        # Rows 0, 1: conditions at x=0
        # Rows 2..27: conditions at z_1..z_12 (2 rows per z)
        
        # Note: The task says g(x) = sum_{k=0}^{25} c_k L_{2k}(x).
        # So the basis functions are L_0, L_2, ..., L_50.
        
        # We need L_{2k}(x) and (L_{2k})'(x).
        
        # Use scipy.special.eval_genlaguerre(n, alpha, x)
        # Derivative can be computed numerically or analytically.
        # For speed in optimization, numerical derivative or pre-computed polys.
        
        # Let's pre-compute the polynomial coefficients for L_{2k}(-1/2, x)
        # This is expensive to do every time. 
        # However, the basis is fixed. Only z_i change.
        # We can pre-compute the coefficients of the basis polynomials once.
        
        # L_n(x) coefficients:
        # We'll represent L_{2k}(x) as a numpy polynomial array.
        
        # Pre-computation of basis polynomials L_{2k}^{(-0.5)}(x)
        # n = 0, 2, ..., 50
        
        # This part is heavy. We do it once outside the optimization loop?
        # No, construct() is called once. But estimate_B is called many times.
        # So we should pre-compute the basis polynomials and their derivatives 
        # at the specific z_i? No, z_i changes.
        # We need to evaluate the polynomials at variable points.
        
        # Better approach:
        # Build the matrix A using the basis values.
        
        # Let's use a simpler proxy if possible?
        # No, the structure of q(x) depends on the exact solution.
        
        # Optimization:
        # 1. Pre-compute the coefficients of L_{2k}^{(-0.5)}(x) for k=0..25.
        # 2. In estimate_B, evaluate these polynomials and their derivatives at 0 and z_i.
        # 3. Solve the linear system.
        # 4. Construct g(x) coefficients.
        # 5. Construct denominator D(x) = x * prod (x-z_i)^2.
        # 6. Perform polynomial division g(x) / D(x).
        # 7. Find roots of the quotient.
        # 8. Return max real root.
        
        # This is still heavy for an inner loop of an optimizer.
        # We will limit the number of evaluations.
        
        # Pre-computed basis (stored in closure or global? No, local to construct)
        # We'll compute them on the fly in the first call or pre-compute here.
        
        # Let's pre-compute the basis polynomials as numpy arrays (coeffs from highest power to lowest)
        # scipy.special.genlaguerre(n, alpha) returns a poly1d object.
        
        # We do this once before the optimization loop.
        pass # Placeholder, logic moved below
        
    # Pre-compute basis polynomials
    # L_{2k}^{(-0.5)}(x)
    basis_polys = []
    for k in range(26):
        n = 2 * k
        # genlaguerre returns a poly1d
        # We need to be careful with alpha = -0.5
        L_n = genlaguerre(n, -0.5)
        # Convert to numpy array of coefficients (highest power first)
        coeffs = L_n.coeffs
        # Derivative
        deriv_coeffs = np.polyder(coeffs)
        basis_polys.append((coeffs, deriv_coeffs))
        
    def estimate_B_impl(z_list):
        # z_list: list of 12 floats
        # Build A and b
        # A is 26x26
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Condition 1: g(0) = 0
        # g(0) = sum c_k L_{2k}(0)
        for k in range(26):
            coeffs, _ = basis_polys[k]
            # L(0) is the constant term, which is coeffs[-1]
            A[0, k] = coeffs[-1]
        b[0] = 0.0
        
        # Condition 2: g'(0) = 1
        # g'(0) = sum c_k (L_{2k})'(0)
        for k in range(26):
            _, deriv_coeffs = basis_polys[k]
            # Derivative at 0 is the coefficient of x^1, which is deriv_coeffs[-1] if len>1 else 0
            if len(deriv_coeffs) > 1:
                A[1, k] = deriv_coeffs[-1]
            else:
                A[1, k] = 0.0
        b[1] = 1.0
        
        # Conditions 3..27: g(z_i) = 0, g'(z_i) = 0 for i=1..12
        for i, z in enumerate(z_list):
            row_g = 2 + 2*i
            row_gp = 3 + 2*i
            
            for k in range(26):
                coeffs, deriv_coeffs = basis_polys[k]
                # Evaluate L_{2k}(z)
                # Using np.polyval
                A[row_g, k] = np.polyval(coeffs, z)
                A[row_gp, k] = np.polyval(deriv_coeffs, z)
            
            b[row_g] = 0.0
            b[row_gp] = 0.0
            
        # Solve A c = b
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 1e100 # Singular, bad
            
        # Construct g(x) coefficients
        # g(x) = sum c_k L_{2k}(x)
        # We need to sum the polynomials weighted by c_k.
        # Find max degree
        max_deg = 50
        g_coeffs = np.zeros(max_deg + 1)
        
        for k in range(26):
            coeffs, _ = basis_polys[k]
            # coeffs is from highest to lowest.
            # Pad to length 51
            # If coeffs is length L, it corresponds to degrees L-1 ... 0
            # We need to align them.
            # Easier: use np.polynomial.Polynomial? No, standard power basis.
            
            # Let's convert everything to standard power basis coefficients (lowest to highest) for easier addition?
            # Or just pad and add.
            
            # coeffs is [a_n, ..., a_0]
            # We want to add to g_coeffs which is [a_50, ..., a_0]
            
            # Find the degree of this polynomial
            deg = len(coeffs) - 1
            # The coefficients correspond to x^deg ... x^0
            # In g_coeffs (index 0 is x^50), the term x^deg is at index 50-deg
            start_idx = 50 - deg
            for j, coeff_val in enumerate(coeffs):
                g_coeffs[start_idx + j] += c[k] * coeff_val
                
        # Construct D(x) = x * prod (x-z_i)^2
        # Start with x
        d_coeffs = [1, 0] # x
        for z in z_list:
            # (x-z)^2 = x^2 - 2zx + z^2
            sq_coeffs = [1, -2*z, z*z]
            d_coeffs = np.convolve(d_coeffs, sq_coeffs)
            
        # Divide g(x) by D(x)
        # g_coeffs is length 51 (deg 50)
        # d_coeffs is length 1 + 24 = 25 (deg 24)
        # Quotient degree 50 - 24 = 26
        
        try:
            q_coeffs, r_coeffs = np.polydiv(g_coeffs, d_coeffs)
        except Exception:
            return 1e100
            
        # Find real roots of q(x)
        roots = np.roots(q_coeffs)
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        real_roots = real_roots[real_roots > 0] # Task says B = max(0, largest counted r)
        
        if len(real_roots) == 0:
            return 0.0
            
        return np.max(real_roots)
        
    # Initial guess: Bessel zeros
    beta_init = math.pi
    z_init = [(k * beta_init) ** 2 for k in range(1, 13)]
    
    # Optimization
    # We optimize beta. z_i = (k * beta)^2
    # We can also optimize the shape, but let's stick to scaling first.
    
    def objective(beta_vec):
        beta = beta_vec[0]
        if beta <= 0:
            return 1e100
        z_list = [(k * beta) ** 2 for k in range(1, 13)]
        return estimate_B_impl(z_list)
        
    # Use Nelder-Mead
    # We will run a limited number of iterations
    best_beta = beta_init
    best_val = objective([beta_init])
    
    # Try a few local optimizations with different starting points
    start_points = [math.pi, math.pi * 0.9, math.pi * 1.1, math.pi * 0.8, math.pi * 1.2]
    
    for sp in start_points:
        if time.time() > deadline:
            break
        try:
            res = minimize(objective, [sp], method='Nelder-Mead', 
                           options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_val:
                best_val = res.fun
                best_beta = res.x[0]
        except Exception:
            pass
            
    # Final check on a few discrete values around the best found
    if best_beta > 0:
        neighbors = [best_beta * (1 + d) for d in [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]]
        for nb in neighbors:
            if time.time() > deadline:
                break
            val = objective([nb])
            if val < best_val:
                best_val = val
                best_beta = nb
                
    # Generate final z list
    z_final = [(k * best_beta) ** 2 for k in range(1, 13)]
    
    # Ensure validity
    if any(not np.isfinite(z) or z <= 0 for z in z_final):
        # Fallback to Bessel
        z_final = [(k * math.pi) ** 2 for k in range(1, 13)]
        
    return [float(z) for z in z_final]

# EVOLVE-BLOCK-END
