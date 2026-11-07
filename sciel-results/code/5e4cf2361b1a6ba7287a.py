"""Symbolic regression: recover hidden univariate f on [-3,3] with 10 free params.

STRATEGIC CHANGE:
The previous linear basis approach scored very high (~1.9e9), implying it could not
approximate the target well. A purely linear model in 10 fixed basis functions is
too restrictive for many non-linear targets.

Here we switch to a NON-LINEAR parametric family with 10 parameters that can
represent a much wider class of univariate functions:

    f(x) = a0 + a1*x + a2*x^2 + a3*x^3
          + A * sin(f1 * x + p1) * cos(f2 * x + p2)
          + B * tanh(s * x + t)
          + C / (1 + (x - c)^2)

Parameters (10 total):
  a0, a1, a2, a3, A, f1, p1, f2, p2, B, s, t, C, c  -> that's 14, too many.

Let me restructure to exactly 10 params:
  p0: a0 (constant)
  p1: a1 (linear)
  p2: a2 (quadratic)
  p3: a3 (cubic)
  p4: amplitude A
  p5: frequency f
  p6: phase phi
  p7: tanh slope s
  p8: rational center c
  p9: rational width w

Model:
  f(x) = p0 + p1*x + p2*x^2 + p3*x^3
       + p4 * sin(p5*x + p6)
       + p7 * tanh(p7*x)   -- wait, p7 used twice.

Let me use 10 distinct params:
  p0, p1, p2, p3: polynomial coefficients
  p4: amplitude
  p5: frequency
  p6: phase
  p7: tanh scale
  p8: rational center
  p9: rational width

Model:
  f(x) = p0 + p1*x + p2*x^2 + p3*x^3
       + p4 * sin(p5*x + p6)
       + p7 * tanh(x)
       + 1.0 / (1.0 + ((x - p8) / p9)^2)

This is smooth, bounded on [-3,3] (tanh bounded, rational bounded, sin bounded,
polynomial bounded on [-3,3]), and non-linear in params, giving BFGS a more
expressive landscape. The non-linearity allows the model to adapt frequency,
phase, and shape to the target.

To handle potential local minima from non-linearity, we could try multiple
starting points, but BFGS from all-ones is mandated. However, the structure is
simpler than the seed (no abs() kinks), so BFGS should behave reasonably.

We add a deadline guard: track elapsed time and if we exceed a safety margin
(e.g., 140s out of 150s), we return the current best estimate. But since the
equation function is called repeatedly by BFGS, we cannot easily track global
state across calls without side effects. Instead, we keep the function pure and
fast. The 150s limit applies to the entire evaluation including BFGS iterations.
Since each call is O(n) with small n=200, and BFGS typically takes <100 iterations,
total time should be well under 150s. No explicit guard needed in equation itself.
"""
import numpy as np
import time

# Module-level state for deadline guard
_start_time = None
_deadline = 145.0  # safety margin: return by 145s out of 150s


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Non-linear parametric model with 10 free parameters."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    # Extract parameters
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
    p4, p5, p6 = p[4], p[5], p[6]
    p7 = p[7]
    p8, p9 = p[8], p[9]

    # Polynomial part
    y = p0 + p1 * x + p2 * x * x + p3 * x * x * x

    # Sinusoidal part: bounded by |p4|
    y = y + p4 * np.sin(p5 * x + p6)

    # Tanh part: bounded by |p7|
    y = y + p7 * np.tanh(x)

    # Rational bump: bounded, centered at p8, width p9
    # Use safe width to avoid division by zero
    width = max(abs(p9), 0.1)
    y = y + 1.0 / (1.0 + ((x - p8) / width) ** 2)

    # Ensure finiteness
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

    return y
