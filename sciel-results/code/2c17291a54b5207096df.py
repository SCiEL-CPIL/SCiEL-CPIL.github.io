"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We'll implement a custom multi-start optimization inside equation()
# to improve the chance of finding a good local minimum.
# The evaluator calls equation() many times during BFGS fitting.
# We can't intercept the BFGS loop directly, but we CAN make the
# landscape friendlier by using a well-conditioned basis.

# Alternative approach: use a richer but well-conditioned basis
# with terms that are natural for a nonlinear oscillator.
# Key insight: the target is acceleration a = f(x, v) for a damped
# nonlinear oscillator. Typical form: a = -k*x - c*v + nonlinear terms + drive

# Let's use a basis that includes:
# - Linear terms (stiffness, damping)
# - Cubic terms (Duffing)
# - Cross terms (coupling)
# - Sine terms (natural periodicity)
# - All scaled to O(1) for conditioning

def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Acceleration of a damped nonlinear oscillator with driving force, from position and velocity."""
    # Scale inputs to O(1) for better BFGS conditioning
    # x range: [-0.5728, 0.5275] -> center ~ -0.02265, scale ~ 0.55
    # v range: [-0.1754, 0.2821] -> center ~ 0.05335, scale ~ 0.23
    
    xs = (x + 0.02265) / 0.55
    vs = (v - 0.05335) / 0.23
    
    # Precompute powers and cross terms
    x2 = xs * xs
    v2 = vs * vs
    x3 = x2 * xs
    v3 = v2 * vs
    x2v = x2 * vs
    xv2 = xs * v2
    xv = xs * vs
    x4 = x2 * x2
    v4 = v2 * v2
    
    # Trigonometric terms (natural for oscillators)
    sx = np.sin(xs)
    sv = np.sin(vs)
    sx2 = np.sin(2.0 * xs)
    sv2 = np.sin(2.0 * vs)
    
    # Cosine terms
    cx = np.cos(xs)
    cv = np.cos(vs)
    
    # Exponential damping-like terms (smooth, bounded)
    exp_neg_x2 = np.exp(-0.5 * x2)
    exp_neg_v2 = np.exp(-0.5 * v2)
    
    # Build a 10-term basis that spans the important physics:
    # params[0]: linear restoring force (stiffness)
    # params[1]: cubic restoring force (Duffing)
    # params[2]: linear damping
    # params[3]: cubic damping
    # params[4]: x^2 * v (velocity-dependent stiffness)
    # params[5]: x * v^2 (velocity-squared coupling)
    # params[6]: sin(x) (periodic restoring)
    # params[7]: sin(v) (periodic damping)
    # params[8]: x * v (cross coupling)
    # params[9]: bias / drive
    
    return (params[0] * xs +
            params[1] * x3 +
            params[2] * vs +
            params[3] * v3 +
            params[4] * x2v +
            params[5] * xv2 +
            params[6] * sx +
            params[7] * sv +
            params[8] * xv +
            params[9])
# EVOLVE-BLOCK-END
