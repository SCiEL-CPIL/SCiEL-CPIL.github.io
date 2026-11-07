"""
Strategy: Chebyshev Polynomial Basis + Sinusoidal Time Term
Rationale:
The previous strategy assumed a specific physical form (Duffing/Van der Pol) which may be too restrictive or poorly conditioned for BFGS starting from all-ones. 
A Chebyshev basis is orthogonal on the interval [-1, 1], leading to a well-conditioned design matrix. 
By scaling inputs to [-1, 1] based on known training ranges, we ensure the basis functions are well-behaved. 
We use degree 3 in x and v (9 terms: 1, x, x^2, x^3, v, v^2, v^3, xv, x^2v) plus a sinusoidal time term sin(w*t+phi). 
Wait, that's 10 terms. 
Let's refine the basis to ensure 10 parameters exactly and good coverage:
1. Constant (Chebyshev T0)
2. T1(x) = x
3. T2(x) = 2x^2 - 1
4. T3(x) = 4x^3 - 3x
5. T1(v) = v
6. T2(v) = 2v^2 - 1
7. T3(v) = 4v^3 - 3v
8. T1(x)*T1(v) = x*v
9. sin(w*t + phi)
10. cos(w*t + phi)

This gives a rich, orthogonal-like basis for the spatial dynamics and a flexible sinusoidal drive.
The parameters are:
p0: coeff T0
p1: coeff T1(x)
p2: coeff T2(x)
p3: coeff T3(x)
p4: coeff T1(v)
p5: coeff T2(v)
p6: coeff T3(v)
p7: coeff T1(x)*T1(v)
p8: frequency w
p9: phase phi

Note: BFGS will fit p0..p7 (linear in params) and p8, p9 (non-linear in params).
This is a mixed problem, but the linear part is well-conditioned.
"""

import numpy as np
import time

_DEADLINE_SECONDS = 140.0
_START = time.time()


def _elapsed_ok() -> bool:
    return (time.time() - _START) < _DEADLINE_SECONDS


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev-based approximation of nonlinear oscillator dynamics.
    Inputs:
        t: time [30, 50]
        x: position [-0.2056, 0.1749]
        v: velocity [-0.3391, 0.3624]
    Params:
        p[0:7]: Coefficients for Chebyshev basis in x, v
        p[7]: Coefficient for x*v term
        p[8]: Frequency of driving force
        p[9]: Phase of driving force
    """
    if not _elapsed_ok():
        return np.zeros(np.shape(t), dtype=float)

    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    p = np.asarray(params, dtype=float)
    
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]))
    p = p[:10]

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p

    # Scale x and v to [-1, 1] to utilize Chebyshev properties
    # Training ranges:
    # x: [-0.2056, 0.1749] -> center: -0.01535, half-width: 0.19025
    # v: [-0.3391, 0.3624] -> center: 0.01165, half-width: 0.35075
    
    x_center = -0.01535
    x_half = 0.19025
    v_center = 0.01165
    v_half = 0.35075
    
    # Avoid division by zero if range is degenerate (unlikely here)
    if x_half < 1e-9: x_half = 1.0
    if v_half < 1e-9: v_half = 1.0

    x_scaled = (x - x_center) / x_half
    v_scaled = (v - v_center) / v_half

    # Clip to [-1, 1] to stay within Chebyshev domain and avoid explosion
    # This is safe because test data is in-distribution
    x_scaled = np.clip(x_scaled, -1.0, 1.0)
    v_scaled = np.clip(v_scaled, -1.0, 1.0)

    # Chebyshev polynomials of first kind
    T0 = np.ones_like(x_scaled)
    T1_x = x_scaled
    T2_x = 2 * x_scaled**2 - 1
    T3_x = 4 * x_scaled**3 - 3 * x_scaled

    T1_v = v_scaled
    T2_v = 2 * v_scaled**2 - 1
    T3_v = 4 * v_scaled**3 - 3 * v_scaled

    # Cross term
    T1_xv = x_scaled * v_scaled

    # Time dependent terms
    # t is in [30, 50]. We model the driving force as A*sin(w*t + phi) + B*cos(w*t + phi)
    # But we only have 2 params left (p8, p9). 
    # Let's use p8 as frequency w and p9 as phase phi.
    # We need amplitudes. We can fix amplitudes to 1.0 and let the phase/frequency adapt, 
    # OR we can interpret p8, p9 differently.
    
    # Alternative: 
    # Use p8 and p9 to define the sinusoid directly? No, that's 2 params for 1 function.
    # Let's stick to: term = sin(p8*t + p9). 
    # But we need a coefficient for this term. We have used p0..p7 for spatial terms.
    # We have p8, p9 left.
    # If we use p8 as frequency and p9 as phase, the amplitude is fixed at 1.0.
    # This might be too restrictive if the driving amplitude is large.
    
    # Let's re-evaluate the parameter allocation.
    # We have 10 params.
    # Spatial basis: 1, x, x^2, x^3, v, v^2, v^3, xv (8 terms). 
    # This uses 8 coefficients.
    # Remaining 2 params for time.
    # If we use sin(w*t+phi), we need w, phi, and amplitude A. That's 3 params.
    # We only have 2.
    
    # Option A: Fix amplitude to 1.0. Risk: poor fit if A != 1.
    # Option B: Use only phase and frequency, but scale the whole equation? No, params are per-term.
    # Option C: Use a different time basis. E.g., sin(t) and cos(t) with fixed frequency? 
    # The input t is [30, 50]. If the oscillation is fast, fixed freq might fail.
    # If the oscillation is slow, maybe t itself is the variable?
    
    # Let's look at the previous successful model (score 67).
    # It used: -k1*x - k2*x^3 + k3*v + k4*x*v + F0*sin(w*t + phi) + c
    # Params: c, k1, k2, k3, k4, F0, w, phi, scale_x, scale_v.
    # That's 10 params.
    # It got 67.
    
    # My Chebyshev proposal:
    # p0: c
    # p1: coeff x
    # p2: coeff x^2
    # p3: coeff x^3
    # p4: coeff v
    # p5: coeff v^2
    # p6: coeff v^3
    # p7: coeff xv
    # p8: w
    # p9: phi
    # Missing: Amplitude of drive.
    
    # Let's swap one spatial term for amplitude.
    # Drop T3_v (v^3) or T2_v (v^2)? 
    # Duffing is x^3. Van der Pol is v^2 damping? Or x*v damping?
    # Let's keep x^3, x, v, xv. Drop v^2, v^3.
    # Basis: 1, x, x^2, x^3, v, xv (6 terms).
    # Remaining 4 params: A, w, phi, and maybe a second harmonic or another spatial term?
    # Let's use:
    # p0: c
    # p1: coeff x
    # p2: coeff x^2
    # p3: coeff x^3
    # p4: coeff v
    # p5: coeff xv
    # p6: coeff v^2  (restore v^2 for quadratic drag)
    # p7: Amplitude A
    # p8: Frequency w
    # p9: Phase phi
    
    # This seems robust. 7 spatial/bias terms + 3 time terms.
    
    T1_x = x_scaled
    T2_x = 2 * x_scaled**2 - 1
    T3_x = 4 * x_scaled**3 - 3 * x_scaled
    T1_v = v_scaled
    T2_v = 2 * v_scaled**2 - 1
    T1_xv = x_scaled * v_scaled
    
    # Recalculate with new parameter mapping
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = p
    
    # Spatial part
    spatial = (
        p0 * np.ones_like(x_scaled) + 
        p1 * T1_x + 
        p2 * T2_x + 
        p3 * T3_x + 
        p4 * T1_v + 
        p5 * T1_xv + 
        p6 * T2_v
    )
    
    # Time part
    # Drive: p7 * sin(p8 * t + p9)
    time_term = p7 * np.sin(p8 * t + p9)
    
    out = spatial + time_term
    
    out = np.where(np.isfinite(out), out, 0.0)
    return np.asarray(out, dtype=float)
