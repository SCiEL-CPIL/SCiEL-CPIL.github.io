import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Strategy: Multi-Start Global Optimization via Flexible Basis Expansion.
    
    The current program uses a fixed parametric form (RBFs with specific roles) 
    and relies on the evaluator's single BFGS run starting from ones. This often 
    gets stuck in local minima or fails to capture complex non-linearities if the 
    initial guess is far from the solution.
    
    This strategy redefines the 10 parameters to control a flexible linear 
    combination of 5 basis functions:
    1. Linear Elastic: (1 + a*t) * s
    2. Quadratic Hardening: s^2 * (1 + b*t)
    3. Cubic Softening/Hardening: s^3
    4. Yield Transition (Smoothstep-like): (s - c)^3 * (d - s)^3 ... actually, 
       let's use a simpler smooth transition: s * sigmoid(k*(s-c)) or similar.
       
    However, to keep it robust and fast, we will use a set of physically plausible 
    basis functions that span the space of typical stress-strain curves:
    B1 = s
    B2 = s^2
    B3 = s^3
    B4 = exp(-((s-c)/w)^2)  # Localized bump
    B5 = 1                   # Intercept
    
    We will use a multi-start approach. The 'params' array passed in by the 
    evaluator is just the initial guess for the FIRST start. But we can ignore 
    the quality of that initial guess and perform our own internal optimization 
    if we had data. Wait, we DO have data (strain, temp). We can compute the 
    target y? No, we don't have y. We only have strain and temp. 
    The evaluator provides y implicitly by computing the loss. But we cannot 
    access y inside equation().
    
    Correction: The evaluator calls equation(strain, temp, params) to get 
    predictions. It then computes MSE against the hidden y. We cannot optimize 
    inside equation() because we don't have y.
    
    Therefore, the "Multi-Start" strategy must be implemented differently. 
    We cannot do internal optimization without y.
    
    Alternative Strategy: **Robust Non-Linear Basis with Better Conditioning**.
    
    The issue with the current RBF strategy might be that the parameters 
    (centers, widths) are highly sensitive and hard for BFGS to tune from 
    ones. 
    
    New Strategy: **Spline-like Basis with Fixed Nodes and Optimizable Weights**.
    
    If we fix the centers and widths of the basis functions, the problem becomes 
    linear in the weights. But we have 10 params. 
    Let's use 5 Gaussian bumps with FIXED centers and widths (based on the 
    expected strain range) and 5 weights. Plus 5 more params for temperature 
    scaling or global terms.
    
    Actually, a better approach for symbolic regression with limited params is 
    to use a **Hybrid Power-Law + Exponential** form that is well-conditioned.
    
    Let's try: 
    sigma = E(t) * s + K(t) * s^n + A(t) * (1 - exp(-s/sat))
    
    Parameters:
    p0: E_base
    p1: E_temp_coeff
    p2: K_base
    p3: K_temp_coeff
    p4: n (exponent, typically 1-3 for metals)
    p5: A_base
    p6: A_temp_coeff
    p7: sat (saturation strain)
    p8: Intercept
    p9: Intercept_temp_coeff
    
    This form is physically motivated (Elastic + Power Law Hardening + Saturation) 
    and typically well-conditioned for BFGS.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    s = strain
    t = temp
    
    # Ensure numerical stability
    # Exponent n should be positive. Clip to [0.1, 10]
    n = np.clip(p4, 0.1, 10.0)
    
    # Saturation strain should be positive. Clip to [1e-4, 5.0]
    sat = np.clip(p7, 1e-4, 5.0)
    
    # Temperature coefficients
    # To avoid extreme values, we can scale t. 
    # t is in [0.06667, 1]. Let's use t directly.
    
    # 1. Elastic Term
    E = p0 + p1 * t
    term_elastic = E * s
    
    # 2. Power Law Hardening Term
    # s^n can be problematic if s is negative. 
    # Strain is in [-0.001911, 0.9636]. 
    # For negative strain (compression), power laws with non-integer n are invalid.
    # We should handle sign(s) or use |s|*sign(s) or just clip s to 0 for hardening?
    # Stress-strain is often symmetric in magnitude but different in sign.
    # However, for a rod under tension (implied by positive strain range mostly),
    # and small compression, we can use:
    # sign(s) * |s|^n
    # But BFGS might struggle with non-smooth abs().
    # Since the range is mostly positive (0 to 0.96) and small negative (-0.0019),
    # we can assume tension-dominant or use a smooth approximation.
    # Let's use s * exp(n * log(abs(s) + eps)) ? No, too complex.
    
    # Simpler: Since the negative strain is very small (-0.0019), and typical 
    # stress-strain curves are linear in that region, the power law term 
    # should be small there. 
    # We can define the power law term only for s > 0, or use a smooth transition.
    # Let's use: s * (1 + (n-1) * s) ... no.
    
    # Let's stick to a form that handles negative strain gracefully.
    # Term 2: K * s^2 (always positive, smooth)
    # Term 3: A * (1 - exp(-s/sat)) (smooth, handles negative s)
    
    # Revised Model:
    # sigma = (p0 + p1*t)*s + (p2 + p3*t)*s^2 + (p5 + p6*t)*(1 - exp(-s/p7)) + (p8 + p9*t)
    
    # This uses:
    # p0, p1: Linear slope
    # p2, p3: Quadratic coefficient (captures initial curvature/hardening)
    # p5, p6: Saturation amplitude
    # p7: Saturation strain scale
    # p8, p9: Intercept
    
    # This uses 9 params. We have 10.
    # Add a cubic term? Or a temp-dependent saturation?
    # Let's add a cubic term with its own temp coeff? No, that's 2 more.
    # Let's use p4 for a scaling of the saturation term or a higher order term.
    
    # Let's use p4 to scale the quadratic term's temperature dependence? 
    # Or add a third polynomial term: p4 * s^3?
    
    # Model:
    # sigma = (p0 + p1*t)*s + (p2 + p3*t)*s^2 + p4*s^3 + (p5 + p6*t)*(1 - exp(-s/p7)) + (p8 + p9*t)
    
    # This is 10 params.
    # p0, p1: Linear
    # p2, p3: Quadratic
    # p4: Cubic (no temp dep to save params)
    # p5, p6: Saturation amplitude
    # p7: Saturation scale
    # p8, p9: Intercept
    
    # This is a very flexible, smooth, well-conditioned function.
    # The exponential term captures the asymptotic behavior (yield/plateau).
    # The polynomial terms capture the local curvature.
    
    # Calculate terms
    term_linear = (p0 + p1 * t) * s
    term_quad = (p2 + p3 * t) * s * s
    term_cubic = p4 * s * s * s
    
    # Exponential saturation term
    # Clip argument to avoid overflow
    arg = -s / sat
    arg = np.clip(arg, -100, 100)
    exp_term = 1.0 - np.exp(arg)
    term_sat = (p5 + p6 * t) * exp_term
    
    term_intercept = p8 + p9 * t
    
    stress = term_linear + term_quad + term_cubic + term_sat + term_intercept
    
    return stress
