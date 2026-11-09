"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Chebyshev polynomial basis in (x, v) with sinusoidal driving in t.
    
    The acceleration of a driven nonlinear oscillator is modeled as:
    a(t, x, v) = [sin(t), cos(t)] dot [A, B] + Chebyshev polynomial in (x, v)
    
    The Chebyshev basis provides well-conditioned global approximation on
    bounded domains, avoiding monomial ill-conditioning and RBF locality issues.
    
    params layout (10 values):
      p0:  sin drive amplitude
      p1:  cos drive amplitude
      p2:  drive frequency multiplier (applied to t)
      p3:  Chebyshev coeff for T1(x)
      p4:  Chebyshev coeff for T1(v)
      p5:  Chebyshev coeff for T2(x)
      p6:  Chebyshev coeff for T2(v)
      p7:  Chebyshev coeff for T1(x)*T1(v)
      p8:  Chebyshev coeff for T1(x)*T2(v)
      p9:  constant bias
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize x, v to [-1, 1] for Chebyshev polynomials
    x_norm = x / 0.21
    v_norm = v / 0.36

    # Clamp to [-1, 1] to avoid numerical issues outside training range
    x_c = np.clip(x_norm, -1.0, 1.0)
    v_c = np.clip(v_norm, -1.0, 1.0)

    # Chebyshev polynomials: T0=1, T1=z, T2=2z^2-1, T3=4z^3-3z
    T1_x = x_c
    T1_v = v_c
    T2_x = 2.0 * x_c**2 - 1.0
    T2_v = 2.0 * v_c**2 - 1.0

    # Sinusoidal driving in t
    tau = p2 * t
    sin_drive = np.sin(tau)
    cos_drive = np.cos(tau)

    # Combine: driving + Chebyshev phase-plane terms
    accel = (p0 * sin_drive + p1 * cos_drive +
             p3 * T1_x + p4 * T1_v +
             p5 * T2_x + p6 * T2_v +
             p7 * T1_x * T1_v +
             p8 * T1_x * T2_v +
             p9)

    return accel
# EVOLVE-BLOCK-END
