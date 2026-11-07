"""
Strategy: Localized Gaussian Radial Basis Function (RBF) basis.

The current program uses a global polynomial series. Global polynomials are
often poor at fitting the sharp 'knee' (yield point) of metallic stress-strain
curves because they tend to oscillate (Runge's phenomenon) or require high
degrees to capture local non-linearity, leading to poor generalization.

This new strategy uses a combination of:
1. A linear elastic term (global stiffness) to capture the initial slope.
2. A set of Gaussian "bumps" (RBFs) centered at specific locations in the
   (strain, temperature) space. These bumps allow the model to locally adjust
   the curvature to fit the yield point and hardening region without affecting
   the elastic region too much.

We use 4 RBF centers. Each center has an x (strain) and y (temp) coordinate
fixed to reasonable physical locations (e.g., near yield strain and various
temperatures) to provide a structured inductive bias, and a fitted amplitude.
This uses 1 (linear) + 4 (amplitudes) = 5 parameters.
We add 5 more parameters for a low-order polynomial background to ensure
smoothness and capture any remaining global trends not covered by the bumps.

Total parameters: 10.
- p0: Linear strain coefficient (Young's Modulus approx)
- p1..p4: Amplitudes of the 4 Gaussian bumps
- p5..p9: Coefficients for a quadratic background in strain and temp
  (p5: const, p6: strain, p7: temp, p8: strain^2, p9: temp^2)
  
Note: To avoid collinearity and optimization difficulties, the Gaussian
centers are fixed based on the input ranges provided in the task description.
Strain range: [-0.001911, 0.9636]
Temp range: [0.06667, 1]

Yield point in aluminum is typically around strain 0.002 - 0.01.
We place bumps at:
1. (0.005, 0.5)  : Near yield, mid-temp
2. (0.02, 0.5)   : Post-yield, mid-temp
3. (0.1, 0.5)    : Hardening region, mid-temp
4. (0.005, 0.9)  : Near yield, high-temp (thermal softening)

Widths (sigma) are fixed to ensure the bumps are localized enough to capture
features but wide enough to be stable.
"""
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress prediction using a hybrid Linear + Gaussian RBF model.
    
    Parameters:
    strain: np.ndarray, shape (N,)
    temp: np.ndarray, shape (N,)
    params: np.ndarray, shape (10,)
    
    Returns:
    stress: np.ndarray, shape (N,)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    s = strain
    t = temp
    
    # 1. Gaussian RBF terms
    # Fixed centers (strain, temp) and widths (sigma_s, sigma_t)
    # We use anisotropic Gaussians to allow different scales in strain vs temp
    
    # Center 1: Yield region, mid temp
    c1_s, c1_t = 0.005, 0.5
    w1_s, w1_t = 0.01, 0.2
    g1 = np.exp(-0.5 * ((s - c1_s) / w1_s)**2 - 0.5 * ((t - c1_t) / w1_t)**2)
    
    # Center 2: Post-yield, mid temp
    c2_s, c2_t = 0.02, 0.5
    w2_s, w2_t = 0.05, 0.2
    g2 = np.exp(-0.5 * ((s - c2_s) / w2_s)**2 - 0.5 * ((t - c2_t) / w2_t)**2)
    
    # Center 3: Hardening, mid temp
    c3_s, c3_t = 0.1, 0.5
    w3_s, w3_t = 0.2, 0.2
    g3 = np.exp(-0.5 * ((s - c3_s) / w3_s)**2 - 0.5 * ((t - c3_t) / w3_t)**2)
    
    # Center 4: Yield region, high temp (thermal softening effect)
    c4_s, c4_t = 0.005, 0.9
    w4_s, w4_t = 0.01, 0.1
    g4 = np.exp(-0.5 * ((s - c4_s) / w4_s)**2 - 0.5 * ((t - c4_t) / w4_t)**2)
    
    # 2. Background Polynomial terms
    # This ensures the model can represent the general trend even if RBFs fail
    # or to capture elastic slope better than just p0*s.
    
    s2 = s * s
    t2 = t * t
    
    # p5: constant
    # p6: linear strain (additive to p0*s for flexibility, or p0 is the main slope)
    # Let's make p0 the primary elastic slope.
    # p6..p9 are correction terms.
    
    bg = p5 + p6 * s + p7 * t + p8 * s2 + p9 * t2
    
    # Combine
    stress = p0 * s + p1 * g1 + p2 * g2 + p3 * g3 + p4 * g4 + bg
    
    return stress
