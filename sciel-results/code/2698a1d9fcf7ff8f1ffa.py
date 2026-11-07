"""Chebyshev polynomial basis with sinusoidal time terms for nonlinear oscillator acceleration."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: evaluator kills after 150s wall-clock. Keep a safety margin.
_DEADLINE_SECONDS = 140.0
_START = time.time()


def _elapsed_ok() -> bool:
    return (time.time() - _START) < _DEADLINE_SECONDS


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator using a Chebyshev Polynomial Basis.
    
    This model uses a global approximation strategy rather than local bumps.
    It scales the inputs (t, x, v) to the interval [-1, 1] and expands the
    acceleration in terms of Chebyshev polynomials of the first kind.
    
    The basis functions are:
    - Chebyshev polynomials in scaled position (x_s) and velocity (v_s) up to degree 2.
    - Sinusoidal time harmonics to capture periodic driving or natural frequency.
    
    Inputs ranges (from task description):
    t: [30, 50]
    x: [-0.2056, 0.1749]
    v: [-0.3391, 0.3624]
    
    We map these to [-1, 1] using linear scaling.
    
    Parameters (10 total):
    p[0]: Weight for T_0(x_s) * T_0(v_s)  (Constant in x,v)
    p[1]: Weight for T_1(x_s) * T_0(v_s)  (Linear in x)
    p[2]: Weight for T_0(x_s) * T_1(v_s)  (Linear in v)
    p[3]: Weight for T_1(x_s) * T_1(v_s)  (Bilinear)
    p[4]: Weight for T_2(x_s) * T_0(v_s)  (Quadratic in x)
    p[5]: Weight for T_0(x_s) * T_2(v_s)  (Quadratic in v)
    p[6]: Weight for T_2(x_s) * T_2(v_s)  (Biquadratic)
    p[7]: Amplitude for cos(omega * t)
    p[8]: Amplitude for sin(omega * t)
    p[9]: Frequency multiplier (omega)
    
    Note: To allow the model to capture time-dependent variations in the state-space
    coefficients (e.g., parametric excitation), we could mix time harmonics with state
    polynomials, but with only 10 params, we separate the time dependence into
    sinusoidal terms added to the state-dependent polynomial surface.
    
    However, a pure additive separation a = f(x,v) + g(t) might be insufficient if
    the oscillator is parametrically driven (e.g., Duffing with time-dependent stiffness).
    
    Let's refine the basis to include some cross terms between time and state, or just
    ensure the time terms are rich enough.
    
    Alternative Allocation:
    Since the system is likely a driven oscillator, the acceleration a(t,x,v) often has
    a form like: a = F(t) + D(v,x) + S(x)
    
    Let's use:
    1. Time harmonics: A*cos(w*t) + B*sin(w*t)  (2 params: A, B, and w is p[9]) -> 3 params
    2. State polynomial surface (Chebyshev):
       T_0(x)T_0(v) = 1
       T_1(x)T_0(v) = x_s
       T_0(x)T_1(v) = v_s
       T_1(x)T_1(v) = x_s*v_s
       T_2(x)T_0(v) = 2*x_s^2 - 1
       T_0(x)T_2(v) = 2*v_s^2 - 1
       
       This gives 6 basis functions in (x,v).
       Total params: 3 (time) + 6 (state) = 9.
       We have 1 left. Let's add T_2(x)T_1(v) or T_1(x)T_2(v) or a higher order time term?
       Let's add a cubic term in x: T_3(x_s) = 4*x_s^3 - 3*x_s.
       
    Revised Params:
    p[0]: w_cos (Amplitude of cos term)
    p[1]: w_sin (Amplitude of sin term)
    p[2]: omega (Frequency)
    p[3]: c0 (Bias)
    p[4]: c1 (Coeff for x_s)
    p[5]: c2 (Coeff for v_s)
    p[6]: c3 (Coeff for x_s*v_s)
    p[7]: c4 (Coeff for T_2(x_s))
    p[8]: c5 (Coeff for T_2(v_s))
    p[9]: c6 (Coeff for T_3(x_s))
    
    This seems robust. The Chebyshev polynomials are orthogonal on [-1,1], which helps
    BFGS converge by reducing multicollinearity compared to monomials.
    """
    # Safety margin guard
    if not _elapsed_ok():
        return np.zeros(np.shape(t), dtype=float)

    # Ensure float arrays and consistent shape
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)

    p = np.asarray(params, dtype=float)
    # Defensive: if params is short, pad with zeros; if long, ignore extras.
    if p.shape[0] < 10:
        p = np.pad(p, (0, 10 - p.shape[0]))
    p = p[:10]

    (w_cos, w_sin, omega, c0, c1, c2, c3, c4, c5, c6) = p

    # 1. Scale inputs to [-1, 1]
    # t: [30, 50] -> center 40, range 10
    # x: [-0.2056, 0.1749] -> center -0.01535, range 0.3805
    # v: [-0.3391, 0.3624] -> center 0.01165, range 0.7015
    
    # To avoid division by zero or extreme scaling issues, we use fixed ranges
    # based on the training data provided in the prompt.
    
    t_mid = 40.0
    t_half_range = 10.0
    
    x_mid = (-0.2056 + 0.1749) / 2.0  # -0.01535
    x_half_range = (0.1749 - (-0.2056)) / 2.0  # 0.19025
    
    v_mid = (-0.3391 + 0.3624) / 2.0  # 0.01165
    v_half_range = (0.3624 - (-0.3391)) / 2.0  # 0.35075
    
    # Clip scaled values to [-1.5, 1.5] to prevent Chebyshev polynomials from
    # blowing up for any slight out-of-distribution inputs, while maintaining
    # linearity within the training range.
    t_s = (t - t_mid) / t_half_range
    x_s = (x - x_mid) / x_half_range
    v_s = (v - v_mid) / v_half_range
    
    t_s = np.clip(t_s, -1.5, 1.5)
    x_s = np.clip(x_s, -1.5, 1.5)
    v_s = np.clip(v_s, -1.5, 1.5)

    # 2. Compute Chebyshev Polynomials
    # T_0(z) = 1
    # T_1(z) = z
    # T_2(z) = 2*z^2 - 1
    # T_3(z) = 4*z^3 - 3*z
    
    T0_x = np.ones_like(x_s)
    T1_x = x_s
    T2_x = 2.0 * x_s**2 - 1.0
    T3_x = 4.0 * x_s**3 - 3.0 * x_s
    
    T0_v = np.ones_like(v_s)
    T1_v = v_s
    T2_v = 2.0 * v_s**2 - 1.0
    
    # 3. Compute Time Harmonics
    # We use the raw t for the phase to ensure the frequency parameter is interpretable,
    # but scaled t could also be used. Using raw t allows 'omega' to have physical units.
    # However, BFGS starting from 1.0 for omega might be too small or large for t in [30,50].
    # If omega=1, cos(t) oscillates very fast.
    # It is safer to scale t for the frequency parameter as well, or just let BFGS find it.
    # Let's use scaled t for the phase argument to keep omega in a reasonable range (e.g., 0-10).
    phase = omega * t_s
    
    cos_t = np.cos(phase)
    sin_t = np.sin(phase)

    # 4. Construct the Model
    # a = w_cos * cos_t + w_sin * sin_t 
    #   + c0 * T0_x * T0_v 
    #   + c1 * T1_x * T0_v 
    #   + c2 * T0_x * T1_v 
    #   + c3 * T1_x * T1_v 
    #   + c4 * T2_x * T0_v 
    #   + c5 * T0_x * T2_v 
    #   + c6 * T3_x * T0_v
    
    # Note: T0 is 1, so terms simplify.
    term_time = w_cos * cos_t + w_sin * sin_t
    term_state = (c0 * T0_x * T0_v + 
                  c1 * T1_x * T0_v + 
                  c2 * T0_x * T1_v + 
                  c3 * T1_x * T1_v + 
                  c4 * T2_x * T0_v + 
                  c5 * T0_x * T2_v + 
                  c6 * T3_x * T0_v)
    
    a = term_time + term_state
    
    # Guarantee finiteness and correct shape
    a = np.where(np.isfinite(a), a, 0.0)
    
    return np.asarray(a, dtype=float)
# EVOLVE-BLOCK-END
