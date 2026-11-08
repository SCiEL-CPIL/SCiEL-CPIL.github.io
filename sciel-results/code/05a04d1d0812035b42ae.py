"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Strategy: Smooth Differentiable Spline (Logistic Basis).
    Unlike hard hinge functions (max(0, x-k)), this model uses logistic 
    sigmoid functions to create smooth transitions. This ensures the loss 
    landscape is differentiable everywhere, which is critical for BFGS 
    convergence. The model captures the elastic region, yield point, and 
    plastic hardening/softening behavior.
    
    Inputs:
    strain: [-0.001911, 0.9636]
    temp:   [0.06667, 1]
    
    Parameters (10 total):
    p[0]: Base elastic modulus (slope before yield)
    p[1]: Temperature coefficient for modulus
    p[2]: Yield strain (knot position 1)
    p[3]: Yield stress offset (intercept shift at yield)
    p[4]: Plastic slope change at yield (hardening/softening)
    p[5]: Second transition strain (knot position 2, e.g., necking or saturation)
    p[6]: Slope change at second transition
    p[7]: Steepness of first transition (controls sharpness of yield)
    p[8]: Steepness of second transition
    p[9]: Global temperature interaction term (linear effect on stress)
    
    Model Structure:
    E_eff = p0 + p1 * (temp - 0.5)
    Base_Stress = E_eff * strain
    
    Transition 1 (Yield): 
    S1 = p3 + p4 * sigmoid(p7 * (strain - p2))
    This adds a stress offset and changes the slope as strain passes p2.
    
    Transition 2 (Plastic regime change):
    S2 = p6 * sigmoid(p8 * (strain - p5))
    This adds a further slope change.
    
    Total Stress = Base_Stress + S1 * strain? 
    Wait, a pure additive sigmoid on strain doesn't change the slope of the 
    linear part unless multiplied by strain or structured as a derivative.
    
    Let's refine the structure to be physically consistent:
    Stress = E(T) * strain + Offset(T) + Nonlinear_Correction
    
    Nonlinear_Correction should be a function that is ~0 in elastic region 
    and grows in plastic region.
    
    Better formulation:
    Let S(x) be a smooth step function.
    Stress = (E0 + E1*T) * strain 
           + C0 + C1*T
           + K1 * (strain - k1) * sigmoid(a1 * (strain - k1))  # Hardening term
           + K2 * (strain - k2) * sigmoid(a2 * (strain - k2))  # Saturation/Necking
    
    This form ` (x-k)*sigmoid(a(x-k)) ` is smooth, zero at x=k, and behaves 
    like `0.5*a*(x-k)^2` near k and `x-k` for large x (if a>0). 
    However, for aluminium, after yield, the slope usually *decreases* 
    (strain hardening is less than elastic modulus) or stays low.
    
    If we use ` (x-k) * sigmoid(a(x-k)) `:
    For x >> k, sigmoid -> 1, term -> x-k. Slope contribution -> 1.
    For x << k, sigmoid -> 0, term -> 0. Slope contribution -> 0.
    
    So the total slope is:
    Elastic: E(T)
    Plastic: E(T) + K1 + K2
    
    If K1 is negative, the plastic slope is lower than elastic, which is 
    physically correct for metals (tangent modulus < Young's modulus).
    
    Let's use this structure:
    p[0]: E0 (Elastic Modulus)
    p[1]: dE/dT (Temperature dependence of Modulus)
    p[2]: k1 (Yield Strain)
    p[3]: K1 (Slope change at yield, likely negative)
    p[4]: a1 (Sharpness of yield transition)
    p[5]: k2 (Second transition strain, e.g., onset of necking/saturation)
    p[6]: K2 (Slope change at second transition)
    p[7]: a2 (Sharpness of second transition)
    p[8]: C0 (Stress offset/Intercept)
    p[9]: C1 (Temperature interaction on intercept)
    """
    
    # Extract parameters
    E0 = params[0]
    dE_dT = params[1]
    k1 = params[2]
    K1 = params[3]
    a1 = params[4]
    k2 = params[5]
    K2 = params[6]
    a2 = params[7]
    C0 = params[8]
    C1 = params[9]
    
    # Center temperature for numerical stability
    T_c = temp - 0.5
    
    # Effective Elastic Modulus
    E_eff = E0 + dE_dT * T_c
    
    # Linear Elastic Stress
    stress_linear = E_eff * strain
    
    # Smooth Nonlinear Correction Terms
    # Term 1: Yield/Hardening
    # Function: (strain - k1) * sigmoid(a1 * (strain - k1))
    # This is 0 at k1, has slope 0 at k1, and slope 1 for strain >> k1 (if a1>0).
    # By scaling with K1, we add a slope of K1 in the plastic region.
    x1 = strain - k1
    # Clip to prevent overflow in exp
    x1_clip = np.clip(x1 * a1, -50, 50)
    sig1 = 1.0 / (1.0 + np.exp(-x1_clip))
    corr1 = K1 * x1 * sig1
    
    # Term 2: Second Transition (e.g., Saturation or Necking)
    x2 = strain - k2
    x2_clip = np.clip(x2 * a2, -50, 50)
    sig2 = 1.0 / (1.0 + np.exp(-x2_clip))
    corr2 = K2 * x2 * sig2
    
    # Intercept Terms
    stress_offset = C0 + C1 * T_c
    
    # Total Stress
    stress = stress_linear + corr1 + corr2 + stress_offset
    
    return stress
# EVOLVE-BLOCK-END
