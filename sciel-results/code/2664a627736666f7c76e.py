"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    
    # Deadline guard: 600s limit, keep 30s margin
    start_time = time.time()
    deadline = start_time + 570.0
    
    m = 12
    n = 1
    alpha = 2 * n  # alpha = 2
    
    def compute_c4_bound(roots):
        """
        Compute the upper bound on C4 for the given roots.
        The bound is typically related to the integral of the squared polynomial
        weighted by the Laguerre weight, or a similar functional in the Cohn-Goncalves setup.
        
        Based on the task description "upper bound on C4", and common uncertainty principles,
        C4 is likely a constant in an inequality like |hat f|_1^2 <= C4 ||f||_2^2 or similar.
        
        A common construction involves a polynomial P(x) = prod (x - r_i)^2.
        The bound might be proportional to the L2 norm of P with respect to the Laguerre weight,
        or related to the moments.
        
        Let's assume the bound is proportional to the integral of [P(x)]^2 * x^alpha * e^{-x} dx,
        normalized by some factor. Since we want to MINIMIZE the bound, we minimize this integral.
        
        However, the "C4" in Fourier sign uncertainty often refers to a specific constant.
        Let's look at the structure:
        The roots are double roots.
        Let P(x) = prod_{i=1}^{12} (x - r_i)^2.
        The weight is w(x) = x^2 e^{-x}.
        The integral I = ∫_0^∞ P(x)^2 x^2 e^{-x} dx.
        
        Alternatively, the bound might be related to the ratio of integrals.
        Given the lack of exact formula, I will use the L2 norm of the polynomial
        in the Laguerre space as a proxy, which is a standard measure of "size" 
        that affects uncertainty bounds.
        
        Actually, a more likely candidate for "C4" in this context is the constant in:
        (∫ |f|^2) (∫ |f'|^2) >= C (∫ |f|)^4 ? No, that's Heisenberg.
        
        Let's stick to the most direct interpretation: The "construction" yields a bound.
        Usually, these bounds are minimized when the polynomial is orthogonal or has specific properties.
        The roots of the Laguerre polynomial L_m^(alpha) are the ones that make the polynomial orthogonal.
        However, the seed program used L_12^(0) and got a low score (0.48).
        The "inspiration" programs scored high (33.3) by "model-and-optimize".
        
        Let's try to minimize the value of the polynomial at the roots? No, roots are 0.
        
        Let's assume the score is inversely related to the "tightness" of the bound.
        If the bound is B(roots), and score = seed/B(roots), then minimizing B improves score.
        
        What is B(roots)?
        In many uncertainty problems, the constant is derived from the ratio:
        B = (∫ P^2 w) / (∫ P w)^2 ? Or similar.
        
        Let's use a robust proxy: The variance of the roots, or the spread.
        But the "Vandermonde condition number" proxy worked well in the knowledge base (0.48 -> 1.02).
        This suggests that *balanced spread* is good.
        
        However, the inspirations with score 33.3 used "polynomial moments".
        Let's compute the L2 norm of the polynomial P(x) = prod (x-r_i)^2 with weight x^alpha e^{-x}.
        If we minimize this norm, we are finding the "smallest" polynomial in that space.
        
        Let's implement the computation of I = ∫_0^∞ [prod (x-r_i)^2]^2 * x^2 * e^{-x} dx.
        This is a moment calculation.
        P(x) = sum c_k x^k.
        P(x)^2 = sum d_k x^k.
        ∫ x^k x^2 e^{-x} dx = Gamma(k+3).
        
        So we need to compute the coefficients of P(x)^2.
        """
        # Compute coefficients of P(x) = prod (x - r_i)^2
        # This can be expensive if done naively for every step in optimization.
        # But for 12 roots, it's manageable.
        
        # Use numpy to build the polynomial
        # Start with [1]
        coeffs = np.array([1.0])
        for r in roots:
            # Multiply by (x - r)^2 = x^2 - 2rx + r^2
            # coeffs = coeffs * (x^2 - 2rx + r^2)
            # coeffs * x^2 -> shift by 2
            # coeffs * (-2r) x -> shift by 1, scale
            # coeffs * r^2 -> scale
            
            new_coeffs = np.zeros(len(coeffs) + 2)
            new_coeffs[2:] += coeffs
            new_coeffs[1:-1] += coeffs * (-2 * r)
            new_coeffs[:-2] += coeffs * (r * r)
            coeffs = new_coeffs
        
        # Now coeffs are for P(x) = c_0 + c_1 x + ... + c_{24} x^{24}
        # We need Q(x) = P(x)^2
        # Q(x) = sum_{k} q_k x^k
        # q_k = sum_{i+j=k} c_i c_j
        
        # Convolution
        q_coeffs = np.convolve(coeffs, coeffs)
        
        # Integral I = sum_{k} q_k * Gamma(k + alpha + 1)
        # alpha = 2
        # Gamma(k+3) = (k+2)!
        
        # Compute factorials up to len(q_coeffs) + 2
        max_k = len(q_coeffs) - 1
        # Use log-gamma for stability if numbers get large, but here 24*2=48 max power.
        # 50! is ~ 3e64, which is fine for float64.
        
        # Precompute factorials
        fact = np.ones(max_k + 3)
        for i in range(2, max_k + 3):
            fact[i] = fact[i-1] * i
        
        # I = sum q_k * fact[k+3]
        # Note: fact array index k corresponds to k!. We need (k+2)! which is fact[k+2] if 1-indexed?
        # My fact array: fact[0]=1, fact[1]=1, fact[2]=2... fact[i]=i!
        # We need Gamma(k+3) = (k+2)!.
        # So for term k, we multiply by fact[k+2].
        
        weights = fact[2 : 2 + len(q_coeffs)]
        
        I = np.dot(q_coeffs, weights)
        
        # The bound C4 might be proportional to I, or I / (something).
        # If we minimize I, we minimize the "energy" of the polynomial.
        # Let's return I as the objective.
        
        # Ensure positivity for optimization stability if it goes negative due to numerical issues
        if I < 0:
            I = 1e-10
            
        return I

    def objective(params):
        """
        Objective function for the optimizer.
        params: 12 values, representing log(roots) to ensure positivity.
        """
        roots = np.exp(params)
        return compute_c4_bound(roots)

    # Initial guess
    # Start with uniform spacing or Laguerre roots
    # Uniform: 1, 2, ..., 12
    x0_uniform = np.log(np.arange(1, 13, dtype=float))
    
    # Laguerre roots L_12^(0)
    try:
        from scipy.special import roots_laguerre
        r_lag, _ = roots_laguerre(12)
        x0_lag = np.log(r_lag)
    except:
        x0_lag = x0_uniform

    # Try both initial guesses and pick the best one, or just start with uniform as it seemed robust
    best_obj = float('inf')
    best_roots = None
    
    # Optimization 1: Start from uniform
    try:
        res1 = minimize(objective, x0_uniform, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-9})
        if res1.success and res1.fun < best_obj:
            best_obj = res1.fun
            best_roots = np.exp(res1.x)
    except Exception as e:
        pass

    # Optimization 2: Start from Laguerre
    try:
        res2 = minimize(objective, x0_lag, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-9})
        if res2.success and res2.fun < best_obj:
            best_obj = res2.fun
            best_roots = np.exp(res2.x)
    except Exception as e:
        pass

    # Optimization 3: Start from optimized Vandermonde (uniform-ish but slightly perturbed)
    # Just in case
    try:
        # Random restarts if time permits
        if time.time() < deadline - 120:
            for i in range(5):
                x0_rand = x0_uniform + np.random.normal(0, 0.1, 12)
                res3 = minimize(objective, x0_rand, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-9})
                if res3.success and res3.fun < best_obj:
                    best_obj = res3.fun
                    best_roots = np.exp(res3.x)
    except Exception as e:
        pass

    if best_roots is None:
        # Fallback to uniform
        best_roots = np.arange(1, 13, dtype=float)
    else:
        # Ensure they are positive and sorted
        best_roots = sorted(best_roots)

    # Check time
    if time.time() > deadline:
        return [float(x) for x in best_roots]

    # Final refinement with higher precision if time allows
    try:
        x0_final = np.log(best_roots)
        res_final = minimize(objective, x0_final, method='L-BFGS-B', options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-12})
        if res_final.success:
            best_roots = np.exp(res_final.x)
            best_roots = sorted(best_roots)
    except:
        pass

    return [float(x) for x in best_roots]

# EVOLVE-BLOCK-END
