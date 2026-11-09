import numpy as np
import scipy.special
from scipy.optimize import minimize

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination.
    Uses numerical optimization to minimize the uncertainty bound.
    """
    m = 12
    
    # Initial guess: roots of L_m(x)
    roots_init, _ = scipy.special.roots_laguerre(m)
    
    # Gauss-Laguerre quadrature nodes and weights
    # N=100 is sufficient for degree ~50 polynomials
    nodes, weights = scipy.special.roots_laguerre(100)
    
    def cost(params):
        x_roots = np.sort(params)
        
        if np.any(x_roots <= 0):
            return 1e10
            
        # Compute P(x) = prod (x - x_i)^2 and P'(x)
        # Using Q(x) = prod (x - x_i)
        # P = Q^2, P' = 2 Q Q'
        
        # Q_vals at nodes
        diff = nodes[:, np.newaxis] - x_roots[np.newaxis, :]
        
        # Calculate Q and Q'
        # Q = prod(diff)
        # To avoid overflow, we can't just multiply directly if values are large, 
        # but nodes are < 200 (for N=100), roots < 50. Diff < 200.
        # 200^12 is 4e27. Within float64 range (1e308).
        Q_vals = np.prod(diff, axis=1)
        
        # Q' = Q * sum(1/diff)
        # Handle diff == 0? Nodes and roots are unlikely to match exactly.
        # If they do, Q=0, Q' is non-zero.
        # 1/diff will be inf. Q * inf = nan.
        # We can compute Q' as sum of products of (diff without one factor).
        # For M=12, this is 12 * 100 * 11! operations? No.
        # Efficient way:
        # Q'(x) = sum_{j=1}^M prod_{k \neq j} (x - x_k)
        
        # Let's use the logarithmic derivative method carefully.
        # If any diff is 0, we handle it separately.
        # Given random roots and fixed nodes, probability of exact match is 0.
        
        inv_diff = 1.0 / diff
        sum_inv = np.sum(inv_diff, axis=1)
        Q_prime_vals = Q_vals * sum_inv
        P_prime_vals = 2.0 * Q_vals * Q_prime_vals
        P_vals = Q_vals**2
        
        # Integrand components
        P_sq = P_vals**2 # Wait, P(x) is already squared in the definition of the polynomial?
        # Task: "12 positive double-root locations".
        # This implies the polynomial is P(x) = prod (x-x_i)^2.
        # So P(x) in my code above is the full polynomial.
        # My code: Q = prod(x-x_i). P = Q^2.
        # So P_vals is the polynomial value.
        # The weight function is e^{-x}.
        # The norm is \int |f|^2 = \int P(x)^2 e^{-x} dx?
        # No, f(x) = P(x) e^{-x/2}.
        # |f(x)|^2 = P(x)^2 e^{-x}.
        # So the integrand for I0 is P(x)^2.
        
        # In my code:
        # P_vals = Q^2.
        # So |f|^2 integrand (before e^-x) is P_vals^2 = Q^4.
        
        # Let's re-verify.
        # f(x) = P(x) e^{-x/2}
        # If P(x) has double roots, P(x) = \prod (x-x_i)^2.
        # Then |f(x)|^2 = (\prod (x-x_i)^2)^2 e^{-x} = (\prod (x-x_i)^4) e^{-x}.
        
        # My code:
        # Q = \prod (x-x_i)
        # P = Q^2
        # P_sq = P^2 = Q^4.
        # This is correct.
        
        integrand_I0 = P_sq
        integrand_I1 = P_sq * nodes
        integrand_I2 = P_sq * nodes**2
        
        # Derivative term for frequency variance
        # f' = (P' - 0.5 P) e^{-x/2}
        # |f'|^2 = (P' - 0.5 P)^2 e^{-x}
        # Integrand = P'^2 - P P' + 0.25 P^2
        # Note: P here is the polynomial P(x), not |f|^2.
        
        integrand_K = P_prime_vals**2 - P_vals * P_prime_vals + 0.25 * P_sq
        
        I0 = np.sum(weights * integrand_I0)
        I1 = np.sum(weights * integrand_I1)
        I2 = np.sum(weights * integrand_I2)
        K_int = np.sum(weights * integrand_K)
        
        if I0 == 0:
            return 1e10
            
        mean_x = I1 / I0
        var_x = I2 / I0 - mean_x**2
        var_xi = K_int / I0
        
        if var_x < 0 or var_xi < 0:
            return 1e10
            
        return var_x * var_xi

    def cost_log(params_log):
        x_roots = np.exp(params_log)
        return cost(x_roots)
    
    init_log = np.log(roots_init)
    
    # Bounds for log(x): x > 1e-6
    bounds = [(np.log(1e-6), np.log(1000))] * m
    
    result = minimize(cost_log, init_log, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15})
    
    if result.success:
        optimal_roots = np.exp(result.x)
    else:
        optimal_roots = roots_init
        
    return [float(r) for r in optimal_roots]
