"""Rewritten program for E. coli growth rate symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s limit


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using piecewise-linear hinge basis functions.
    
    This model is LINEAR in parameters, making the MSE loss convex and
    enabling BFGS to find the global optimum from the all-ones start.
    
    Basis functions:
      - Substrate: log1p(s), hinge(s - 5), hinge(s - 15)
      - Temperature: hinge(temp - 25), hinge(35 - temp), hinge(temp - 30)
      - pH: hinge(pH - 6), hinge(8 - pH), hinge(pH - 7)
      - Density: log1p(b), hinge(b - 10), hinge(b - 20)
      - Constant: 1
    
    10 parameters total for the linear combination.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.zeros_like(b, dtype=np.float64)

    # Hinge function: max(0, x)
    def hinge(x):
        return np.maximum(0, x)

    # Substrate basis (s range: 1 to 40)
    log_s = np.log1p(np.clip(s, 0, None))
    s_hinge1 = hinge(s - 5.0)
    s_hinge2 = hinge(s - 15.0)

    # Temperature basis (temp range: 1 to 5)
    t_hinge1 = hinge(temp - 2.5)
    t_hinge2 = hinge(4.0 - temp)
    t_hinge3 = hinge(temp - 3.0)

    # pH basis (pH range: 1 to 12)
    ph_hinge1 = hinge(pH - 6.0)
    ph_hinge2 = hinge(8.0 - pH)
    ph_hinge3 = hinge(pH - 7.0)

    # Density basis (b range: 1 to 50)
    log_b = np.log1p(np.clip(b, 0, None))
    b_hinge1 = hinge(b - 10.0)
    b_hinge2 = hinge(b - 20.0)

    # Constant
    one = np.ones_like(b, dtype=np.float64)

    # Linear combination with 10 parameters
    mu = (params[0] * one +
          params[1] * log_s +
          params[2] * s_hinge1 +
          params[3] * s_hinge2 +
          params[4] * t_hinge1 +
          params[5] * t_hinge2 +
          params[6] * t_hinge3 +
          params[7] * ph_hinge1 +
          params[8] * ph_hinge2 +
          params[9] * log_b)

    return mu
# EVOLVE-BLOCK-END
