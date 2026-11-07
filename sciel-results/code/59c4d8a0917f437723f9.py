import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Pure Bivariate Polynomial Chaos Expansion.
    
    Instead of physical constitutive laws (Hooke's law, power laws) or localized
    bumps (RBFs), we model the stress-strain-temperature surface as a global
    polynomial. This is a "model-agnostic" approach that relies on the flexibility
    of polynomial basis functions to fit the data.
    
    We use 10 parameters. To capture the essential physics (linear elastic response
    at low strain, hardening at high strain) and temperature dependence, we select
    a specific set of monomials from the full bivariate Taylor series.
    
    The basis functions are chosen to be orthogonal-like or at least well-conditioned
    within the input ranges:
    strain s in [-0.002, 0.96]
    temp t in [0.067, 1.0]
    
    We normalize strain to [0, 1] roughly to improve conditioning, but since we
    can't easily center/scale inside the equation without knowing the exact mean
    (though we can approximate), we will use the raw variables but keep degrees low
    to avoid ill-conditioning.
    
    Actually, let's use a slightly smarter set of basis functions that are
    standard in polynomial chaos but adapted for this domain.
    
    Basis functions:
    1. 1
    2. s
    3. s^2
    4. s^3
    5. s^4
    6. t
    7. s*t
    8. s^2*t
    9. t^2
    10. s^3*t
    
    This allows:
    - A polynomial in s (degree 4) to capture the non-linear hardening curve.
    - Linear and quadratic dependence on t.
    - Interaction terms (s*t, s^2*t, s^3*t) to allow the shape of the stress-strain
      curve to change with temperature (e.g., yield point drop, hardening rate change).
    
    This is a linear model in the parameters!
    stress = p0*1 + p1*s + p2*s^2 + p3*s^3 + p4*s^4 + p5*t + p6*s*t + p7*s^2*t + p8*t^2 + p9*s^3*t
    
    Wait, if it's linear in parameters, BFGS is overkill but will still work perfectly
    to find the least squares solution (it will converge in one step or very few).
    Actually, for a linear model, `scipy.optimize.minimize` with BFGS will find the
    global optimum exactly (within numerical precision) because the error surface
    is a convex quadratic. This is a HUGE advantage over non-linear models where
    BFGS can get stuck in local minima.
    
    The risk is that a degree-4 polynomial in s might not be flexible enough to
    capture the sharp yield point or the specific hardening behavior if it's not
    well-approximated by a low-degree polynomial. However, the stress-strain curve
    of aluminum is generally smooth after yielding. The "kink" at yield is the
    main issue. A degree-4 polynomial might smooth over the yield point too much.
    
    Let's check the degrees of freedom. 2161 data points. 10 parameters.
    We have plenty of data.
    
    Can we make it more flexible?
    We can use higher order terms in s.
    Let's try:
    1. 1
    2. s
    3. s^2
    4. s^3
    5. s^4
    6. s^5
    7. s^6
    8. t
    9. s*t
    10. s^2*t
    
    This gives a degree-6 polynomial in s, which is much more flexible for the
    hardening curve. The temperature dependence is simpler (linear in t, with
    interaction with s and s^2).
    
    Let's verify the conditioning.
    s is up to 1. s^6 is up to 1.
    t is up to 1.
    The matrix X will have columns [1, s, s^2, ..., s^6, t, s*t, s^2*t].
    This should be well-conditioned enough for BFGS.
    
    Let's implement this.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    s = strain
    t = temp
    
    # Precompute powers of s
    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2
    s5 = s4 * s
    s6 = s4 * s2
    
    # Basis functions
    # 1. Constant
    b0 = 1.0
    # 2-7. Powers of s
    b1 = s
    b2 = s2
    b3 = s3
    b4 = s4
    b5 = s5
    b6 = s6
    # 8. Temp
    b7 = t
    # 9. Interaction s*t
    b8 = s * t
    # 10. Interaction s^2*t
    b9 = s2 * t
    
    # Linear combination
    stress = (p0 * b0 + 
              p1 * b1 + 
              p2 * b2 + 
              p3 * b3 + 
              p4 * b4 + 
              p5 * b5 + 
              p6 * b6 + 
              p7 * b7 + 
              p8 * b8 + 
              p9 * b9)
    
    return stress
