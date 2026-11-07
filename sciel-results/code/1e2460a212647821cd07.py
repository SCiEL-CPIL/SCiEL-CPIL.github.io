"""
Strategy: Physics-informed Ramberg-Osgood model with internal parameter sanitization.
This approach uses a non-linear form in parameters, which is standard for constitutive
models. To handle BFGS starting from all ones (which might be a poor start for
exponents or moduli), we apply soft constraints (clipping) to the parameters inside
the function. This ensures the function is always finite and well-behaved, allowing
BFGS to find a reasonable local minimum without crashing or diverging.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Uses a Ramberg-Osgood style model:
    sigma = E * eps_plastic_term + ... 
    Actually, the standard Ramberg-Osgood is:
    eps = sigma/E + (sigma/K)^n
    
    Inverting this is hard. Instead, we can model sigma directly using a form that
    approximates the inverse or a direct power law plasticity model:
    sigma = E * eps_elastic + H * (eps_plastic)^n
    
    Or more commonly for fitting, we can use a direct constitutive equation:
    sigma = E * eps + K * (eps - eps_yield)^n * (1 - exp(-alpha*(eps - eps_yield)))
    
    However, a very robust "non-linear" form that is easy to optimize and captures
    the yield plateau and hardening is:
    sigma = A * (1 - exp(-B * eps)) + C * eps^D
    
    Let's try a specific physics-based form:
    sigma = E * eps * (1 - exp(-H * eps^N)) / (1 - exp(-H * eps_max^N)) ... no, too complex.
    
    Let's use a modified power law that is smooth at zero:
    sigma = E * eps + K * (eps + delta)^n - K * delta^n
    
    Parameters:
    p0: E (Young's Modulus) - needs to be positive. Map: exp(p0)
    p1: K (Hardening Constant) - needs to be positive. Map: exp(p1)
    p2: n (Hardening Exponent) - needs to be positive. Map: exp(p2)
    p3: delta (Offset for smoothness) - needs to be small positive. Map: exp(p3)
    p4: Temp coeff for E - Map: 1 + 0.1 * p4
    p5: Temp coeff for K - Map: 1 + 0.1 * p5
    p6: Temp coeff for n - Map: 1 + 0.1 * p6
    p7: Yield offset? 
    p8: Higher order correction
    p9: Bias/Offset
    
    Let's refine the model to be robust.
    sigma = E(T) * eps + K(T) * (max(eps, 0) + delta)^n(T) - K(T) * delta^n(T)
    
    This ensures:
    1. At eps=0, sigma=0 (if bias is 0).
    2. Linear elastic region at small eps (dominated by E*eps).
    3. Power law hardening at large eps.
    4. Temperature dependence on E, K, n.
    """
    
    eps = np.asarray(strain, dtype=np.float64)
    T = np.asarray(temp, dtype=np.float64)
    
    # Extract and sanitize parameters
    # p0: log(E) -> E = exp(p0). Start p0=1 => E=2.7. Typical Al E ~ 70 GPa? 
    # The strain range is [-0.0019, 0.96]. Target stress [-0.005, 0.91].
    # If eps=0.001, sigma=0.005, E ~ 5.
    # If eps=0.9, sigma=0.9, K*0.9^n ~ 0.9.
    # So parameters of order 1-5 are reasonable.
    
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    # Soft constraints to keep parameters in physical ranges
    # E should be positive. exp(p0) is always positive.
    E = np.exp(p0)
    
    # K should be positive.
    K = np.exp(p1)
    
    # n (exponent) should be positive. 
    # If n is large, (eps+delta)^n can overflow.
    # Clamp n to a reasonable range, e.g., [0.1, 10].
    # Using a sigmoid or tanh mapping to [0.1, 10] from p2.
    # Let's just use exp(p2) but clip the result of the power calculation.
    # Or better, map p2 to n in [0.5, 5.0] using a smooth function.
    # n = 0.5 + 4.5 * (1 + np.tanh(p2 - 1.0)) / 2  # Maps p2=1 to n=2.5
    # Let's try: n = np.exp(p2). If p2=1, n=2.7.
    n = np.exp(p2)
    
    # delta (smoothing parameter) should be small and positive.
    delta = np.exp(p3) * 0.01 # Scale down to keep it small
    
    # Temperature effects
    # T is in [0.066, 1].
    # Let's assume E decreases with T, K decreases with T.
    # E_T = E * (1 - 0.1 * T * p4)  # p4 controls sensitivity
    # K_T = K * (1 - 0.1 * T * p5)
    # n_T = n * (1 + 0.1 * T * p6)
    
    # To prevent division by zero or negative moduli, we use soft clipping.
    # Let's use a simpler multiplicative factor that is always positive.
    # factor = 1.0 / (1.0 + p4 * T) ? No, let's keep it simple.
    
    # Let's define temperature-dependent parameters:
    # E_eff = E * (1.0 - 0.5 * T * np.tanh(p4))
    # K_eff = K * (1.0 - 0.5 * T * np.tanh(p5))
    # n_eff = n * (1.0 + 0.5 * T * np.tanh(p6))
    
    # Ensure E_eff and K_eff are positive.
    # If T is high and tanh(p4) is positive, E_eff can become negative if 0.5*T*tanh(p4) > 1.
    # T max is 1. So if tanh(p4) > 2, it breaks. tanh is in [-1,1].
    # So 1 - 0.5*1*1 = 0.5. Always positive. Good.
    
    E_eff = E * (1.0 - 0.5 * T * np.tanh(p4))
    K_eff = K * (1.0 - 0.5 * T * np.tanh(p5))
    n_eff = n * (1.0 + 0.5 * T * np.tanh(p6))
    
    # Calculate plastic strain term
    # We want a smooth transition.
    # term = (eps + delta)^n_eff
    # To avoid negative base if eps < -delta, we use abs or max.
    # Since strain can be negative (compression), we should handle sign.
    # For metals, compression is similar but let's assume symmetry or handle sign.
    # Let's use: sign(eps) * (abs(eps) + delta)^n_eff
    # But for small eps, we want linear behavior.
    # The model: sigma = E_eff * eps + K_eff * (sign(eps) * (abs(eps) + delta)^n_eff - delta^n_eff)
    
    # This ensures sigma=0 at eps=0.
    # For eps > 0: E*eps + K*((eps+delta)^n - delta^n)
    # For eps < 0: E*eps - K*((-eps+delta)^n - delta^n)
    
    abs_eps = np.abs(eps)
    sign_eps = np.sign(eps)
    
    # Calculate power term safely
    # (abs_eps + delta)^n_eff
    # If n_eff is large and abs_eps+delta > 1, this can be huge.
    # But eps is max 0.96, delta is small. 1.0^10 is 1.
    # If n_eff is 100, 1.01^100 is ~2.7.
    # It's safe for these ranges.
    
    base = abs_eps + delta
    power_term = np.power(base, n_eff)
    
    # Subtract the offset to make it zero at eps=0
    # delta^n_eff
    offset = np.power(delta, n_eff)
    
    plastic_part = sign_eps * (power_term - offset)
    
    # Elastic part
    elastic_part = E_eff * eps
    
    # Combine
    sigma = elastic_part + K_eff * plastic_part
    
    # Add higher order correction for fine tuning
    # p7, p8 can adjust the shape.
    # Let's add a small quadratic term to adjust curvature.
    # sigma += p7 * eps^2 + p8 * T * eps
    
    # We have 2 params left: p7, p8.
    # Let's use p7 as a global scaling factor and p8 as a bias.
    # Actually, the model is quite flexible.
    # Let's add: sigma = sigma * (1 + p7 * eps) + p8
    # This allows for non-linear scaling and offset.
    
    sigma = sigma * (1.0 + p7 * eps) + p8
    
    # Ensure finite output
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    return sigma
# EVOLVE-BLOCK-END
