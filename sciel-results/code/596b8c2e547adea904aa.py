# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import quad

# Fixed constants for Hermite-Gaussian framework (physicist's convention assumed, consistent with seed)
_H = [eval_hermite(n) for n in [0, 4, 8, 12]]
_HNORM = [1.0, np.sqrt(4*np.pi), np.sqrt(32*np.pi), np.sqrt(256*np.pi)]
_C12_DEG4 = 2.0 * np.sqrt(2.0) / (100.0 * np.sqrt(np.pi))

def _eval_poly(coeffs, x):
    c0, c4, c8, c12 = coeffs
    return (c0 * _H[0](x) / _HNORM[0] +
            c4 * _H[1](x) / _HNORM[1] +
            c8 * _H[2](x) / _HNORM[2] +
            c12 * _H[3](x) / _HNORM[3])

def _compute_bound(free_coeffs):
    """
    Computes the uncertainty product bound for the Hermite combination.
    Returns:
        bound: The value A(f)*A(hat{f})
        success: Boolean indicating if the function is valid (g(0)=0, non-zero energy)
    """
    try:
        c0, c4, c8 = free_coeffs
        
        # Constraint: g(0) = 0
        # P(0) = c0 * 1 - c4 * 3 - c8 * 105 - c12 * 945 = 0
        # c12 = (c0 - 3*c4 - 105*c8) / 945
        c12 = (c0 - 3.0*c4 - 105.0*c8) / 945.0
        coeffs = np.array([c0, c4, c8, c12])
        
        # Helper for integration of |P(x) exp(-x^2/2)|
        def integrand_l1(x):
            p = _eval_poly(coeffs, x)
            return np.abs(p * np.exp(-0.5 * x * x))
            
        # Helper for integration of P(x)^2 exp(-x^2) (L2 norm squared)
        def integrand_l2(x):
            p = _eval_poly(coeffs, x)
            return (p * p) * np.exp(-x * x)
            
        # Helper for integration of x P(x) exp(-x^2/2) (moment)
        # Note: x * P(x) * exp(-x^2/2) is odd if P is even.
        # However, the Fourier transform of this involves derivatives or shifts.
        # The prompt specifies: "score is the upper bound on C4".
        # The uncertainty product involves the spread in position and frequency.
        # Spread in position: <x^2> - <x>^2
        # Spread in frequency: related to <k^2> - <k>^2
        
        # For a real, even function g(x):
        # 1. L2 norm: E = \int g^2 dx
        # 2. Moment <x> = 0 (by symmetry)
        # 3. Moment <x^2> = \int x^2 g^2 dx / E
        # So Var(x) = <x^2>
        
        # Frequency domain:
        # \hat{g}(k) = \mathcal{F}[g(k)]
        # Uncertainty in frequency: \int k^2 |\hat{g}(k)|^2 dk / \int |\hat{g}(k)|^2 dk
        # By Plancherel, \int |\hat{g}|^2 dk = \int g^2 dx = E (if Fourier def is unitary? or 1/sqrt(2pi)?)
        # Standard convention: \hat{g}(k) = \int g(x) e^{-ikx} dx.
        # Then \int |\hat{g}|^2 dk = 2\pi \int g^2 dx.
        # \int k^2 |\hat{g}|^2 dk = 2\pi \int (g')^2 dx.
        # So Var(k) = \frac{2\pi \int (g')^2 dx}{2\pi \int g^2 dx} = \frac{\int (g')^2 dx}{\int g^2 dx}.
        
        # So the product is: \sqrt{\frac{\int x^2 g^2}{\int g^2}} \cdot \sqrt{\frac{\int (g')^2}{\int g^2}}
        
        # We need to integrate:
        # 1. I0 = \int P(x)^2 e^{-x^2} dx
        # 2. I2 = \int x^2 P(x)^2 e^{-x^2} dx
        # 3. Ig2 = \int (g'(x))^2 dx = \int (P' e^{-x^2/2} - x P e^{-x^2/2})^2 dx
        #          = \int (P' - x P)^2 e^{-x^2} dx
        
        # Note: (P' - xP) is the derivative of g w.r.t x.
        
        def deriv_p(x):
            # Numerical or analytical derivative?
            # Analytical is safer. P(x) is a polynomial.
            # P'(x) = sum c_n h_n'(x) / h_n(0) ... wait, h_n are physicist Hermite.
            # h_n'(x) = n h_{n-1}(x)? No, h_n'(x) = 2n h_{n-1}(x).
            # Let's compute P'(x) explicitly using the coefficients.
            c = coeffs
            # P(x) = c0 h0 + c4 h4 + c8 h8 + c12 h12 (scaled)
            # P'(x) = c4 h4'(x)/N4 + c8 h8'(x)/N8 + c12 h12'(x)/N12
            # h4'(x) = 8 h3(x)
            # h8'(x) = 16 h7(x)
            # h12'(x) = 24 h11(x)
            return (c[1] * 8 * eval_hermite(3, x) / _HNORM[1] +
                    c[2] * 16 * eval_hermite(7, x) / _HNORM[2] +
                    c[3] * 24 * eval_hermite(11, x) / _HNORM[3])

        def integrand_ig2(x):
            dp = deriv_p(x)
            p = _eval_poly(coeffs, x)
            term = dp - x * p
            return (term * term) * np.exp(-x * x)
            
        # Perform Integrations
        # Use quad for precision. Range [-15, 15] is sufficient for Gaussians.
        i0_val, _ = quad(integrand_l2, -15, 15, limit=200)
        i2_val, _ = quad(lambda x: x * x * integrand_l2(x), -15, 15, limit=200)
        ig2_val, _ = quad(integrand_ig2, -15, 15, limit=200)
        
        if i0_val < 1e-15:
            return np.inf, False
            
        var_x = i2_val / i0_val
        var_k = ig2_val / i0_val
        
        # Heisenberg product
        bound = np.sqrt(var_x * var_k)
        
        return bound, True

    except Exception as e:
        return np.inf, False

def construct():
    start_time = time.time()
    timeout = 200  # Safety margin (limit is 210s)
    
    # Seed coefficients
    seed_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    
    # Evaluate seed
    best_bound, _ = _compute_bound(seed_coeffs)
    best_coeffs = seed_coeffs.copy()
    
    # Optimization Strategy:
    # The objective is smooth. We can use L-BFGS-B or Nelder-Mead.
    # L-BFGS-B requires gradients. We can approximate them or provide analytical ones.
    # Given the low dimension (3), Nelder-Mead is robust against local minima and doesn't need gradients.
    # However, it's slower. Let's try BFGS with numerical gradient first, maybe with a few random restarts if time permits.
    # Actually, the previous "failed" attempt mentioned in the prompt notes used BFGS/Nelder-Mead and scored 1.0.
    # Wait, the prompt says "CURRENT PROGRAM (score 1.0000)".
    # And "Measured: 0.152 -> 1. (strategy of a program that improved island 6)"?
    # No, the prompt says "Measured: 0.152 -> 1." in the context of "Knowledge from this run".
    # This implies that an objective correction improved the score from 0.152 to 1.0?
    # Or that the current score IS 1.0?
    # "The evaluator measures the raw value (lower is better). The search score is seed/value, so 1.0 equals the seed program".
    # So if the raw value is the same as seed, score is 1.0.
    # If we find a better (lower) raw value, score > 1.0.
    # The note "0.152 -> 1" is confusing. Maybe the seed value was 0.152 in some previous normalization?
    # Or maybe the "current program" is the seed, and it scores 1.0.
    # We need to find coefficients that yield a lower `_compute_bound` value.
    
    # Let's verify the seed value.
    # If the seed is optimal, we can't beat it. But the task implies we CAN improve.
    # The "Uncertainty inequality" bound is minimized when g is a Gaussian (or Hermite function).
    # The constraint g(0)=0 forces it to be a combination.
    # A pure h_2 (degree 2) has g(0) != 0.
    # A combination of h_0, h_4, h_8 allows more flexibility.
    
    # Let's try Nelder-Mead with a few restarts.
    # Define the objective for the optimizer
    def objective(coeffs):
        b, valid = _compute_bound(coeffs)
        if not valid:
            return 1e10
        return b
        
    # We need to make sure we don't time out.
    # Nelder-Mead can be expensive.
    # Let's try a gradient-based method first as it's faster if we are in the basin of attraction.
    # We will provide the gradient via finite differences or analytical if easy.
    # Since `_compute_bound` does integration, finite difference gradient will call `_compute_bound` multiple times.
    # 3 variables -> ~7-10 calls per gradient step.
    # Nelder-Mead also calls multiple times per step.
    
    # Let's try L-BFGS-B with numerical gradient (approx_grad=True is not supported in newer scipy? use finite difference diff in `minimize` options).
    # Options: {'maxiter': 50, 'ftol': 1e-10}
    
    # Initial point
    x0 = best_coeffs
    
    # Try 1: Local optimization from seed
    # Using 'Nelder-Mead' is safe but slow. 'L-BFGS-B' is faster.
    # Let's try 'TNC' or 'L-BFGS-B'.
    # L-BFGS-B uses finite difference gradients by default if jac is not provided.
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
        if res.success:
            new_bound, _ = _compute_bound(res.x)
            if new_bound < best_bound:
                best_bound = new_bound
                best_coeffs = res.x.copy()
    except Exception as e:
        pass

    # Try 2: A few random restarts if time allows.
    # The landscape might have local minima.
    # Given the structure, the global minimum might be unique or close to seed.
    # Let's add 3 random restarts.
    rng = np.random.default_rng(42)
    
    max_restarts = 5
    time_per_restart = 30 # 150s / 5
    # We have about 150s left (approx)
    
    for i in range(max_restarts):
        if time.time() - start_time > timeout - 10: # Keep 10s safety
            break
            
        # Random perturbation
        # Scale is roughly O(1)
        x_try = best_coeffs + rng.normal(0, 0.5, size=3)
        # Or explore larger range
        # x_try = rng.normal(0, 1.0, size=3)
        
        # Run minimization
        try:
            res = minimize(objective, x_try, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-10})
            if res.success:
                new_bound, _ = _compute_bound(res.x)
                if new_bound < best_bound:
                    best_bound = new_bound
                    best_coeffs = res.x.copy()
        except Exception as e:
            pass
            
    # Final Check: Return the coefficients
    # The task asks for the three coefficients of the Hermite combination (degrees 0, 4, 8).
    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
