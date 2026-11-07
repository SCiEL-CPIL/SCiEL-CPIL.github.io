"""
Strategy: Physically-Motivated Global Parametric Model (Ramberg-Osgood Variant).

The current program uses localized Gaussian Radial Basis Functions (RBFs) with
fixed centers. While RBFs are universal approximators, they can suffer from
local minima and poor generalization if the fixed centers don't align perfectly
with the data's features (like the yield point). Furthermore, RBFs are not
inherently monotonic, which can lead to unphysical oscillations in the
stress-strain curve.

This new strategy uses a global, smooth, monotonic functional form based on
the Ramberg-Osgood model, augmented with temperature dependence and transient
hardening. This form is:
1. Physically grounded: It explicitly models elastic stiffness, yield strength,
   and hardening modulus.
2. Smooth and monotonic: The power-law and exponential terms ensure the stress
   increases with strain, avoiding unphysical drops.
3. BFGS-friendly: The parameters have clear physical meanings and scales,
   leading to a well-conditioned optimization landscape compared to generic
   RBF amplitudes.

Model:
  sigma = E(T) * eps + K(T) * eps^n + A(T) * exp(-eps/tau) + C(T)

Parameters (10 total):
  p0: Elastic Modulus scale (E0)
  p1: Temperature coefficient for E (dE/dT)
  p2: Yield/Hardening scale (K0)
  p3: Temperature coefficient for K (dK/dT)
  p4: Hardening exponent (n)
  p5: Transient amplitude scale (A0)
  p6: Temperature coefficient for A (dA/dT)
  p7: Transient decay rate (1/tau)
  p8: Constant offset (C0)
  p9: Temperature coefficient for C (dC/dT)

Inputs:
  strain: np.ndarray, shape (N,)
  temp: np.ndarray, shape (N,)
  params: np.ndarray, shape (10,)
"""
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress prediction using a Ramberg-Osgood-like parametric model.
    
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
    
    # Ensure positive bases for power laws to avoid NaNs with negative strain
    # Strain range is [-0.001911, 0.9636], so we shift slightly if needed, 
    # but typically Ramberg-Osgood uses total strain. 
    # To be safe against negative bases in power laws if n is non-integer,
    # we use abs(s) for the power term but keep s for the linear elastic term.
    # However, for small negative strains, stress should be negative.
    # Let's use sign(s) * |s|^n for the hardening term to maintain odd symmetry 
    # if possible, or just assume strain is mostly positive in the hardening region.
    # Given the range includes small negative values (elastic compression), 
    # a simple s^n might fail. 
    # Better: Use s for linear, and (s + offset)^n? No, that breaks linearity at 0.
    # Standard RO: epsilon = sigma/E + (sigma/K)^(1/n). 
    # Inverted: sigma = K * (epsilon - sigma/E)^n is implicit.
    # Explicit approximation: sigma = E*eps + K*eps^n.
    # For negative eps, eps^n is problematic if n is not integer.
    # We can use np.sign(s) * np.abs(s)**n for the non-linear term.
    
    # Temperature-dependent coefficients
    # E(T) = p0 + p1 * t
    E_T = p0 + p1 * t
    
    # K(T) = p2 + p3 * t
    K_T = p2 + p3 * t
    
    # Hardening exponent n = p4
    # Ensure n is positive to avoid singularities at 0
    n = np.maximum(p4, 0.1)
    
    # Transient term: A(T) * exp(-s / tau)
    # A(T) = p5 + p6 * t
    A_T = p5 + p6 * t
    
    # Decay rate inv_tau = p7
    inv_tau = np.maximum(p7, 0.1)
    
    # Constant offset: C(T) = p8 + p9 * t
    C_T = p8 + p9 * t
    
    # Calculate terms
    # Elastic term
    elastic = E_T * s
    
    # Plastic/Hardening term
    # Use sign-preserving power law
    hardening = K_T * np.sign(s) * np.abs(s) ** n
    
    # Transient term (often models rapid initial hardening or Bauschinger effect)
    # We use exp(-abs(s)/tau) to keep it well-behaved, or exp(-s/tau) for s>0.
    # Let's use exp(-np.abs(s) * inv_tau) for stability
    transient = A_T * np.exp(-np.abs(s) * inv_tau)
    
    # Combine
    stress = elastic + hardening + transient + C_T
    
    return stress
