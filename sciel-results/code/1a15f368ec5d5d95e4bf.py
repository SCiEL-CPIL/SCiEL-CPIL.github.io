import numpy as np
import time

def equation(strain, temp, params):
    """
    Symbolic regression for stress-strain of an aluminium rod.
    
    Strategy: Linear-in-Parameters Basis Expansion with Physically Motivated Terms.
    
    The previous approach used a rational function (Padé approximant) where parameters
    appear in the denominator. This creates a non-convex optimization landscape, making
    it difficult for BFGS to find the global minimum, especially starting from all ones.
    
    This strategy uses a model that is LINEAR in the parameters.
    Stress = sum_i w_i * phi_i(strain, temp)
    
    By choosing a set of basis functions phi_i that capture the physics of the
    stress-strain curve (elastic region, yield, plastic hardening, temperature effects),
    the optimization problem becomes a standard linear least squares problem.
    BFGS is guaranteed to converge to the global minimum for such problems (assuming
    the basis is well-conditioned), leading to a much more robust fit.
    
    Basis Functions:
    1. Elastic term: strain (linear at small strains)
    2. Elastic temp correction: strain * temp (modulus changes with temp)
    3. Plastic hardening: (strain - s_y)_+^2 (quadratic hardening after yield)
       We use a smooth approximation for the positive part to maintain differentiability:
       softplus(k*(strain - s_y))^2
    4. Plastic temp correction: plastic_term * temp
    5. Higher order elastic: strain^2 (slight curvature in elastic regime or initial yielding)
    6. Higher order elastic temp: strain^2 * temp
    7. Cubic hardening: (strain - s_y)_+^3
    8. Cubic hardening temp: cubic_term * temp
    9. Constant offset (noise/intercept): 1
    10. Temp offset: temp
    
    To keep it simple and robust, we will use a fixed yield strain s_y = 0.002 (typical for Al)
    and a sharpness k=100 for the softplus to approximate the kink.
    Actually, to ensure smoothness for BFGS, we use np.maximum(strain - 0.002, 0) which is 
    non-differentiable at the kink. BFGS might struggle. 
    Better: use a smooth approximation like 0.5 * (x + sqrt(x^2 + eps)) or just polynomial 
    terms that naturally go to zero or are small in the elastic region.
    
    Alternative Basis:
    Since BFGS is used on the parameters, and the function is linear in params, the Hessian 
    is constant (X^T X). This is ideal.
    
    Let's define basis functions that are smooth everywhere:
    phi_1 = strain
    phi_2 = strain * temp
    phi_3 = strain^2
    phi_4 = strain^2 * temp
    phi_5 = strain^3
    phi_6 = strain^3 * temp
    phi_7 = strain^4
    phi_8 = strain^4 * temp
    phi_9 = 1.0
    phi_10 = temp
    
    This is essentially a polynomial regression with temperature interaction terms.
    It is linear in parameters.
    Why this is better than the rational function:
    1. Convex optimization: BFGS will find the global minimum.
    2. No division by zero risks.
    3. Polynomial basis is universal approximator on compact sets.
    4. The "different strategy" is the shift from non-linear (in params) to linear (in params) 
       structural model, leveraging the convexity for reliable optimization.
    """
    s = strain
    t = temp
    p = params
    
    # Build basis functions
    # Using a mix of pure strain powers and strain-temp interactions
    # This allows the model to capture:
    # - Linear elastic response (p[0]*s + p[2]*s^2 + ...)
    # - Temperature dependent modulus (p[1]*s*t + p[3]*s^2*t + ...)
    # - Non-linear hardening (higher order s terms)
    # - Temperature dependent hardening (higher order s*t terms)
    
    # Calculate powers of strain
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    
    # Basis vector
    # 0: s
    # 1: s*t
    # 2: s^2
    # 3: s^2*t
    # 4: s^3
    # 5: s^3*t
    # 6: s^4
    # 7: s^4*t
    # 8: 1
    # 9: t
    
    # Construct the weighted sum
    stress = (p[0] * s + 
              p[1] * s * t + 
              p[2] * s2 + 
              p[3] * s2 * t + 
              p[4] * s3 + 
              p[5] * s3 * t + 
              p[6] * s4 + 
              p[7] * s4 * t + 
              p[8] + 
              p[9] * t)
    
    return stress
