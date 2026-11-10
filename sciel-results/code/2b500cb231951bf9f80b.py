"""
Combined program: Uses the derivative of L_{13}^{(-1/2)} roots as initial guesses,
then refines them using a fast float-based approximation of the score to minimize
the largest real root of the quotient polynomial q(x).
"""
import time
import math
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_genlaguerre
import numpy.polynomial.polynomial as P

# EVOLVE-BLOCK-START

def _get_initial_z():
    """
    Get initial z_i as roots of the derivative of L_{13}^{(-1/2)}(x).
    """
    try:
        # Roots of L_{13}^{(-1/2)}
        roots_13 = roots_genlaguerre(13, -0.5)
        roots_13 = np.real(roots_13)
        
        # Construct polynomial from roots
        # np.poly returns coefficients in descending order
        p = np.poly(roots_13)
        
        # Derivative
        p_prime = np.polyder(p)
        
        # Roots of derivative
        roots_prime = np.roots(p_prime)
        
        # Filter real positive roots
        z_candidates = []
        for r in roots_prime:
            if np.isreal(r) and np.real(r) > 1e-10:
                z_candidates.append(np.real(r))
                
        if len(z_candidates) == 12:
            return sorted(z_candidates)
        else:
            # Fallback: roots of L_{12}^{(-1/2)}
            roots_12 = roots_genlaguerre(12, -0.5)
            return sorted(np.real(roots_12).tolist())
    except Exception:
        return [4.0 * (i + 1) for i in range(12)]

def _approx_score(z_list):
    """
    Fast float-based approximation of the score B / (2*pi).
    Returns B (not divided by 2*pi) for optimization, as scaling is constant.
    """
    try:
        z = np.array(z_list, dtype=float)
        if np.any(z <= 0) or np.any(np.diff(np.sort(z)) < 1e-10):
            return 1e10 # Penalty for invalid
        
        # Construct Laguerre polynomials L_{2k}^{(-1/2)}(x) for k=0..25
        # We need their values and derivatives at x=0 and x=z_i
        # L_n^{(alpha)}(x) can be computed using scipy.special.eval_genlaguerre
        # But we need to build the linear system for coefficients c_k
        
        # Matrix A (26x26)
        # Rows:
        # 0: g(0) = 0
        # 1: g'(0) = 1
        # 2..25: g(z_i)=0, g'(z_i)=0 for i=0..11
        
        # We need L_{2k}^{(-1/2)}(x) and its derivative.
        # scipy.special.eval_genlaguerre(n, alpha, x) gives value.
        # Derivative: L_n^{(\alpha)\prime}(x) = -L_{n-1}^{(\alpha+1)}(x)
        
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        # Precompute Laguerre values and derivatives at 0 and z_i
        # For speed, we can just compute them on the fly in the loop, 
        # but 26*26*13 evaluations is small.
        
        for k in range(26):
            n = 2 * k
            alpha = -0.5
            
            # Value at 0
            # L_n^{(alpha)}(0) = binomial(n+alpha, n)
            # Using scipy
            L_0 = scipy_special_eval_genlaguerre(n, alpha, 0)
            # Derivative at 0
            # L_n^{(alpha)\prime}(x) = -L_{n-1}^{(alpha+1)}(x)
            if n > 0:
                Ld_0 = -scipy_special_eval_genlaguerre(n-1, alpha+1, 0)
            else:
                Ld_0 = 0.0
                
            A[0, k] = L_0
            A[1, k] = Ld_0
            
            for i in range(12):
                zi = z[i]
                L_zi = scipy_special_eval_genlaguerre(n, alpha, zi)
                if n > 0:
                    Ld_zi = -scipy_special_eval_genlaguerre(n-1, alpha+1, zi)
                else:
                    Ld_zi = 0.0
                
                A[2 + 2*i, k] = L_zi
                A[3 + 2*i, k] = Ld_zi
        
        b[1] = 1.0
        
        # Solve linear system
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 1e10 # Singular system penalty
            
        # Construct g(x)
        # g(x) = sum c_k L_{2k}^{(-1/2)}(x)
        # We need the polynomial coefficients of g(x) to divide by denominator.
        # It is easier to work with roots or values, but division requires coefficients.
        
        # Get coefficients of each L_{2k}^{(-1/2)}(x)
        # L_n^{(alpha)}(x) = sum_{j=0}^n (-1)^j binomial(n+alpha, n-j) x^j / j!
        # We can compute coefficients for each n and sum them up weighted by c_k.
        
        g_coeffs = np.zeros(51) # Degree up to 50
        
        for k in range(26):
            n = 2 * k
            alpha = -0.5
            # Coefficients of L_n^{(alpha)}(x)
            # L_n^{(alpha)}(x) = \sum_{j=0}^n \binom{n+\alpha}{n-j} \frac{(-x)^j}{j!}
            coeffs_L = np.zeros(n+1)
            for j in range(n+1):
                # binom(n+alpha, n-j)
                # scipy.special.binom
                binom_val = scipy_special_binom(n + alpha, n - j)
                coeffs_L[j] = binom_val * ((-1)**j) / math.factorial(j)
            
            g_coeffs[:n+1] += c[k] * coeffs_L
        
        # Denominator: x * prod_{i=1}^{12} (x - z_i)^2
        # Degree 1 + 24 = 25
        # Construct polynomial with roots 0, z_1, z_1, ..., z_12, z_12
        roots_denom = np.concatenate(([0.0], np.repeat(z, 2)))
        denom_coeffs = np.poly(roots_denom) # Descending order
        
        # Convert g_coeffs to descending order
        g_desc = g_coeffs[::-1]
        
        # Polynomial division
        # q(x) = g(x) / denom(x)
        # Use np.polydiv
        q_desc, rem = np.polydiv(g_desc, denom_coeffs)
        
        if np.any(np.abs(rem) > 1e-8):
            # Not exact division? Should be exact theoretically.
            # If remainder is large, the system might be ill-conditioned or we made a mistake.
            # For now, proceed with q_desc.
            pass
        
        # Find roots of q(x)
        if len(q_desc) == 0:
            return 0.0
        if len(q_desc) == 1:
            return 0.0
            
        q_roots = np.roots(q_desc)
        
        # Filter real roots
        real_roots = []
        for r in q_roots:
            if np.isreal(r) and np.real(r) > 1e-10:
                # Check sign change? The task says "counts if q has strictly opposite signs at r-eps and r+eps".
                # For simple real roots, this is true.
                # We assume roots from np.roots are simple if they are real.
                real_roots.append(np.real(r))
                
        if not real_roots:
            return 0.0
            
        return max(real_roots)
        
    except Exception as e:
        # print(f"Error in approx score: {e}", file=sys.stderr)
        return 1e10

# Need to import scipy.special functions
from scipy.special import eval_genlaguerre as scipy_special_eval_genlaguerre
from scipy.special import binom as scipy_special_binom

def construct():
    """
    Returns 12 double-root locations z_1..z_12.
    """
    start_time = time.time()
    deadline = start_time + 580.0 # 20s margin
    
    # 1. Get initial guesses
    z_init = _get_initial_z()
    best_z = z_init
    best_score = _approx_score(z_init)
    
    # 2. Optimize
    # Use Nelder-Mead
    # We minimize the largest root B.
    # The score is B / (2*pi). Minimizing B is equivalent.
    
    def objective(z):
        return _approx_score(z)
    
    # Options for Nelder-Mead
    options = {
        'maxiter': 200,
        'xatol': 1e-8,
        'fatol': 1e-8,
        'adaptive': True
    }
    
    try:
        # Run optimization
        # We might want to restrict z to be positive and distinct.
        # Nelder-Mead doesn't handle bounds well.
        # We can use a transformation or just rely on the penalty in objective.
        # Since initial z are positive and distinct, and steps are small, it should be fine.
        
        result = minimize(objective, z_init, method='Nelder-Mead', options=options)
        
        if result.success or (time.time() < deadline - 60):
            if result.fun < best_score:
                best_z = result.x
                best_score = result.fun
                
    except Exception as e:
        print(f"Optimization failed: {e}", file=sys.stderr)
        
    # 3. Final Validation and Return
    # Ensure the returned values are valid (finite, >0, distinct)
    final_z = list(best_z)
    for i in range(12):
        if not np.isfinite(final_z[i]) or final_z[i] <= 0:
            final_z = [4.0 * (i + 1) for i in range(12)]
            break
    for i in range(12):
        for j in range(i+1, 12):
            if abs(final_z[i] - final_z[j]) < 1e-9:
                final_z = [4.0 * (i + 1) for i in range(12)]
                break
        else:
            continue
        break
        
    return final_z

# EVOLVE-BLOCK-END
