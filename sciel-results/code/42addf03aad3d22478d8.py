import numpy as np
import time
from scipy.integrate import quad
from scipy.optimize import differential_evolution
from scipy.special import eval_hermite

_start_time = time.time()
_TIME_LIMIT = 190.0  # Safety margin before 210s kill

def _compute_bound_analytic(a):
    """
    Computes the uncertainty ratio R = (int x^2 g^2) / (int g^2)
    for g(x) = P(x) exp(-x^2/2).
    a = [a0, a4, a8]
    a12 is derived from P(0)=0.
    """
    a0, a4, a8 = a
    
    # Coefficients for H0, H4, H8, H12
    # P(x) = a0 H0 + a4 H4 + a8 H8 + a12 H12
    # P(0) = a0*1 + a4*3 + a8*105 + a12*10395 = 0
    a12 = -(a0 * 1.0 + a4 * 3.0 + a8 * 105.0) / 10395.0
    
    coeffs = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    # Denominator: int P(x)^2 exp(-x^2) dx
    # = sum_i coeffs[i]^2 * sqrt(pi) * 2^deg_i * deg_i!
    denom = 0.0
    for c, n in zip(coeffs, degrees):
        denom += c**2 * np.sqrt(np.pi) * (2**n) * np.math.factorial(n)
    
    if denom < 1e-10:
        return 1e10
        
    # Numerator: int x^2 P(x)^2 exp(-x^2) dx
    # x^2 P(x) = sum_i c_i x^2 H_{n_i}(x)
    # We need to express x^2 H_n(x) in terms of Hermite polynomials.
    # x^2 H_n(x) = 0.25 H_{n+2}(x) + (n + 0.5) H_n(x) + n(n-1) H_{n-2}(x) is not quite right for all n.
    # Let's use the recurrence: x^2 H_n = 0.25 H_{n+2} + (n+1) H_n + n(n-1) H_{n-2}
    # Check n=0: 0.25 H_2 + 1 H_0. 
    #   0.25(4x^2-2) + 1 = x^2 - 0.5 + 1 = x^2 + 0.5. 
    #   But x^2 H_0 = x^2. 
    #   So the formula is wrong.
    # Correct expansion:
    # x^2 H_n = 1/4 H_{n+2} + (n + 1/2) H_n + n(n-1) H_{n-2}
    # Check n=0: 1/4 H_2 + 1/2 H_0 = 1/4(4x^2-2) + 1/2 = x^2 - 1/2 + 1/2 = x^2. Correct.
    # Check n=1: 1/4 H_3 + 3/2 H_1 = 1/4(8x^3-12x) + 3/2(2x) = 2x^3 - 3x + 3x = 2x^3.
    #   But x^2 H_1 = x^2(2x) = 2x^3. Correct.
    # Check n=2: 1/4 H_4 + 5/2 H_2 + 2 H_0.
    #   1/4(16x^4-48x^2+12) + 2.5(4x^2-2) + 2 = 4x^4 - 12x^2 + 3 + 10x^2 - 5 + 2 = 4x^4 - 2x^2.
    #   x^2 H_2 = x^2(4x^2-2) = 4x^4 - 2x^2. Correct.
    
    # So the expansion is:
    # x^2 H_n = 1/4 H_{n+2} + (n + 0.5) H_n + n(n-1) H_{n-2}
    
    # We need to sum the squares of the coefficients of each H_k in the expansion of x^2 P(x).
    # Let Q(x) = x^2 P(x) = sum_k d_k H_k(x).
    # Then int x^2 P(x)^2 e^{-x^2} dx = int (x P(x)) (x P(x)) e^{-x^2} dx ? No.
    # Numerator is int x^2 P(x)^2 e^{-x^2} dx.
    # Let S(x) = x P(x). Then x^2 P(x)^2 = S(x)^2.
    # So Numerator = int S(x)^2 e^{-x^2} dx.
    # S(x) = x sum_i c_i H_{n_i}(x) = sum_i c_i x H_{n_i}(x).
    # x H_n(x) = 1/2 H_{n+1}(x) + n H_{n-1}(x).
    # So S(x) = sum_i c_i [ 1/2 H_{n_i+1}(x) + n_i H_{n_i-1}(x) ].
    # We can collect coefficients for each H_k in S(x).
    
    max_deg = max(degrees) + 1
    # d_k is the coefficient of H_k in S(x)
    d = np.zeros(max_deg + 2)
    
    for c, n in zip(coeffs, degrees):
        if c == 0: continue
        # Term 1: 1/2 H_{n+1}
        d[n+1] += 0.5 * c
        # Term 2: n H_{n-1}
        if n > 0:
            d[n-1] += n * c
        # if n == 0, n H_{-1} is 0.
            
    # Now Numerator = int S(x)^2 e^{-x^2} dx = sum_k d_k^2 sqrt(pi) 2^k k!
    numer = 0.0
    for k in range(len(d)):
        if d[k] != 0:
            numer += d[k]**2 * np.sqrt(np.pi) * (2**k) * np.math.factorial(k)
            
    return numer / denom

def _objective(a):
    return _compute_bound_analytic(a)

def construct():
    """
    Returns three coefficients [a0, a4, a8].
    """
    elapsed = time.time() - _start_time
    if elapsed > _TIME_LIMIT:
        return np.array([-1.0, 0.0, 0.0])
    
    # Bounds for differential evolution
    # The coefficients can be scaled, so we can fix one, e.g. a0=1, or search in a box.
    # Since the ratio is scale invariant, we can search in a bounded box.
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    # Initial best guess
    best_a = np.array([1.0, 0.0, 0.0])
    best_val = _objective(best_a)
    
    # Use Differential Evolution for global search
    # maxiter should be small enough to finish in time
    # popsize=15, tol=1e-8
    try:
        result = differential_evolution(_objective, bounds, seed=42, maxiter=100, tol=1e-8, popsize=10, polish=True)
        if result.fun < best_val:
            best_val = result.fun
            best_a = result.x
    except Exception:
        pass
    
    # Refine with Nelder-Mead if time permits
    elapsed = time.time() - _start_time
    if elapsed < _TIME_LIMIT - 20:
        try:
            from scipy.optimize import minimize
            res = minimize(_objective, best_a, method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000})
            if res.fun < best_val:
                best_val = res.fun
                best_a = res.x
        except Exception:
            pass
            
    return best_a
