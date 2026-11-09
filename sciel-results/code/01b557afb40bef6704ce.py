"""Rewritten program for symbolic regression of a nonlinear oscillator."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a damped nonlinear oscillator with driving force, from position and velocity.
    Uses a physically-motivated Duffing oscillator structure with tanh-based
    nonlinearities and Gaussian RBFs for local features.
    """
    p = params
    
    # Physically-motivated structure:
    # a = -k(x) * x - c(v) * v + F_drive
    # where k(x) is state-dependent stiffness and c(v) is state-dependent damping
    
    # p[0]: linear stiffness
    # p[1]: cubic stiffness coefficient (Duffing term)
    # p[2]: linear damping
    # p[3]: cubic damping coefficient
    # p[4]: stiffness modulation frequency
    # p[5]: damping modulation frequency
    # p[6]: RBF center for x (local feature)
    # p[7]: RBF width for x
    # p[8]: RBF amplitude
    # p[9]: bias/driving term
    
    # State-dependent stiffness: k(x) = p[0] + p[1] * tanh(p[4] * x^2)
    # This captures Duffing-like nonlinearity smoothly
    k_x = p[0] + p[1] * np.tanh(p[4] * x**2)
    
    # State-dependent damping: c(v) = p[2] + p[3] * np.tanh(p[5] * v)
    c_v = p[2] + p[3] * np.tanh(p[5] * v)
    
    # Gaussian RBF for local features in x
    rbf = p[8] * np.exp(-((x - p[6])**2) / (2.0 * (p[7]**2 + 1e-6)))
    
    # Combine: acceleration = -k(x)*x - c(v)*v + RBF + bias
    y = -k_x * x - c_v * v + rbf + p[9]
    
    return y
# EVOLVE-BLOCK-END
