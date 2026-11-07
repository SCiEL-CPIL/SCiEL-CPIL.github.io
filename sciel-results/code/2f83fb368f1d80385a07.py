import numpy as np

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Physics-informed model for a nonlinear oscillator (Duffing-Van der Pol type).
    
    The target is likely the acceleration (d2x/dt2) or the force term, given the 
    inputs are position (x) and velocity (v). 
    Standard form: d2x/dt2 = F(x, v)
    
    We model F(x, v) as a sum of physical terms:
    1. Linear restoring force: -k1 * x
    2. Nonlinear restoring (Duffing): -k3 * x^3
    3. Linear damping: -c1 * v
    4. Nonlinear damping (Van der Pol): c3 * v^3
    5. Cross terms for potential asymmetries or parametric effects:
       - x^2 * v (parametric damping)
       - x * v^2 (nonlinear drag coupling)
       - v^2 * x (another coupling)
       - x^4 (higher order stiffness)
       - v^4 (higher order drag)
       - x^2 (bias in stiffness)
    
    We use 10 parameters. To ensure BFGS converges from an all-ones start,
    we scale the inputs and the parameters' effective influence.
    
    Input Ranges:
    x: [-0.5728, 0.5275] -> Center ~ -0.02, Range ~ 1.1
    v: [-0.1754, 0.2821] -> Center ~ 0.05, Range ~ 0.46
    
    Target Range: [-0.1047, 0.1106] -> Small magnitude, implies strong cancellation
    or small coefficients for the dominant terms.
    
    Strategy:
    Use a structured basis that mimics the oscillator dynamics.
    p0: bias (constant force)
    p1: linear stiffness (x)
    p2: linear damping (v)
    p3: cubic stiffness (x^3)
    p4: cubic damping (v^3)
    p5: quadratic stiffness (x^2) - for asymmetry
    p6: quadratic damping (v^2) - for rectification
    p7: cross term x*v
    p8: cross term x^2*v
    p9: cross term x*v^2
    
    To help BFGS, we pre-scale the basis functions to be O(1) and orthogonal-ish.
    However, since we can't compute data-dependent stats inside equation(x, v, params)
    efficiently without passing data stats, we rely on fixed scaling based on known ranges.
    
    Let's define scaled variables:
    X = x / 0.6  (approx [-1, 1])
    V = v / 0.3  (approx [-1, 1])
    
    Basis:
    1. 1
    2. X
    3. V
    4. X^2
    5. X^3
    6. V^2
    7. V^3
    8. X*V
    9. X^2*V
    10. X*V^2
    
    This is similar to the previous attempt but we will add a crucial twist:
    The previous attempt scored 32409, which is very high (bad). 
    The inspiration 1 (score 74629) and 2 (46796) are also bad.
    Wait, the score is `seed/value`. Lower raw value is better.
    Seed program score 32409.2540.
    If the seed is the monomial basis, and it scored 32409, maybe the monomial basis is struggling 
    because the parameters are not well-conditioned or the basis is not capturing the physics well.
    
    Let's try a different structural approach: 
    Instead of just a polynomial, let's use a model that explicitly separates 
    conservative (potential derivative) and dissipative (damping) forces.
    
    F(x, v) = d/dx U(x) + G(v) + H(x, v)
    
    Let's stick to the 10-term polynomial but ensure the scaling is robust.
    The key failure of simple polynomials in symbolic regression tasks often comes from 
    the optimizer getting stuck in local minima due to parameter redundancy.
    
    Alternative Idea: Use a Radial Basis Function (RBF) like kernel in 2D?
    No, must be closed form with 10 params.
    
    Let's try a "Physics-Guided" scaling.
    If the system is a Duffing oscillator: x'' + mu(x^2-1)x' + x + alpha x^3 = 0
    Then x'' = -mu(x^2-1)x' - x - alpha x^3
    
    This involves terms: x, x^3, x^2 v, v.
    Let's construct a basis that heavily weights these physical terms.
    
    Basis:
    1. 1 (Bias)
    2. x
    3. x^3
    4. v
    5. x^2 v
    6. x^2
    7. v^3
    8. x v
    9. x^4
    10. v^2
    
    To improve conditioning for BFGS starting at 1, we should scale the basis functions
    so that their typical magnitude is around 1.
    
    x ~ 0.5, x^2 ~ 0.25, x^3 ~ 0.125, x^4 ~ 0.06
    v ~ 0.2, v^2 ~ 0.04, v^3 ~ 0.008
    
    If params start at 1, the contribution of x^3 is small, x is large.
    This creates a gradient imbalance.
    
    Let's pre-scale the basis terms by inverse of their typical magnitude.
    Term 2: x / 0.5 = 2x
    Term 3: x^3 / 0.125 = 8x^3
    Term 4: v / 0.2 = 5v
    Term 5: x^2 v / (0.25*0.2) = 20 x^2 v
    Term 6: x^2 / 0.25 = 4x^2
    Term 7: v^3 / 0.008 = 125 v^3
    Term 8: x v / (0.5*0.2) = 10 x v
    Term 9: x^4 / 0.06 ~ 16 x^4
    Term 10: v^2 / 0.04 = 25 v^2
    
    This normalization makes the gradients with respect to params more uniform.
    """
    # Fixed scaling factors based on input ranges to normalize basis functions
    # x range ~ [-0.57, 0.53], mean ~ 0, std ~ 0.3
    # v range ~ [-0.18, 0.28], mean ~ 0, std ~ 0.15
    
    # Pre-compute scaled basis terms
    x2 = x * x
    x3 = x2 * x
    x4 = x2 * x2
    
    v2 = v * v
    v3 = v2 * v
    
    xv = x * v
    x2v = x2 * v
    xv2 = x * v2
    
    # Normalized basis
    # We want each term to be O(1) when params are O(1)
    # Typical magnitudes:
    # x: 0.3, x2: 0.09, x3: 0.027, x4: 0.008
    # v: 0.15, v2: 0.0225, v3: 0.003
    # xv: 0.045, x2v: 0.0135, xv2: 0.006
    
    # Inverse scales (approx)
    s_x = 1.0 / 0.3
    s_x2 = 1.0 / 0.09
    s_x3 = 1.0 / 0.027
    s_x4 = 1.0 / 0.008
    
    s_v = 1.0 / 0.15
    s_v2 = 1.0 / 0.0225
    s_v3 = 1.0 / 0.003
    
    s_xv = 1.0 / 0.045
    s_x2v = 1.0 / 0.0135
    s_xv2 = 1.0 / 0.006
    
    # Basis vector
    b0 = 1.0
    b1 = x * s_x
    b2 = x3 * s_x3
    b3 = v * s_v
    b4 = x2v * s_x2v
    b5 = x2 * s_x2
    b6 = v3 * s_v3
    b7 = xv * s_xv
    b8 = x4 * s_x4
    b9 = v2 * s_v2
    
    # Combine
    return (params[0] * b0 +
            params[1] * b1 +
            params[2] * b2 +
            params[3] * b3 +
            params[4] * b4 +
            params[5] * b5 +
            params[6] * b6 +
            params[7] * b7 +
            params[8] * b8 +
            params[9] * b9)
