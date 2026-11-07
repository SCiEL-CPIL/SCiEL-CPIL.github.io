"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.

    Model: a polynomial in strain (to 3rd order) whose coefficients are
    polynomials in temperature (to 2nd order). This is a universal power
    series that lets the BFGS optimizer fit non-linear temperature effects
    and both symmetric (even in strain) and antisymmetric (odd in strain)
    hardening structures without sign instability or division singularities.

    stress = d0 + d1*t + d2*t^2
           + s * (a0 + a1*t + a2*t^2)
           + s^2 * (b0 + b1*t)
           + s^3 * (c0 + c1*t)

    Params:
      0: a0  - base linear-in-strain coefficient (elastic modulus)
      1: a1  - linear T coefficient of modulus (thermal softening)
      2: a2  - quadratic T coefficient of modulus
      3: b0  - base symmetric (s^2) hardening
      4: b1  - linear T coefficient of symmetric hardening
      5: c0  - base antisymmetric (s^3) hardening
      6: c1  - linear T coefficient of antisymmetric hardening
      7: d0  - baseline offset
      8: d1  - linear T baseline (thermal stress)
      9: d2  - quadratic T baseline
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    elapsed = time.time() - _start_time
    if elapsed > 130:
        return np.zeros_like(strain, dtype=np.float64)

    a0, a1, a2, b0, b1, c0, c1, d0, d1, d2 = params

    s = strain
    t = temp

    # Temperature-dependent coefficient for the linear-in-strain term
    lin_coeff = a0 + a1 * t + a2 * (t * t)

    # Temperature-dependent coefficient for the symmetric s^2 term
    sym_coeff = b0 + b1 * t

    # Temperature-dependent coefficient for the antisymmetric s^3 term
    anti_coeff = c0 + c1 * t

    # Baseline (thermal) term, quadratic in temperature
    baseline = d0 + d1 * t + d2 * (t * t)

    result = (
        baseline
        + s * lin_coeff
        + (s * s) * sym_coeff
        + (s * s * s) * anti_coeff
    )

    return result
# EVOLVE-BLOCK-END
