# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress-strain model for aluminium rod using piecewise-linear ReLU activations.
    
    Structure:
    - Linear elastic base: E(T) * eps
    - Multiple ReLU hinge terms to capture yield transition and hardening
    - Temperature-dependent knot positions and slopes
    
    Parameters (10 total):
    p0: base elastic modulus
    p1: temperature effect on modulus
    p2: yield stress scale
    p3: base yield strain (knot position)
    p4: temperature effect on yield strain
    p5: first hardening slope
    p6: second hardening slope  
    p7: third hardening slope (saturation)
    p8: knot spacing parameter
    p9: intercept/offset
    """
    eps = strain
    T = temp

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Temperature-dependent elastic modulus
    E_T = p0 + p1 * T

    # Elastic contribution
    elastic = E_T * eps

    # Temperature-dependent yield strain (knot position)
    # Use softplus to keep knot positions positive and differentiable
    eps_yield = p3 + p4 * T + 0.01 * np.log1p(np.exp(0))  # small offset for stability

    # Hinge 1: primary yield transition
    # ReLU(eps - eps_yield) * slope
    hinge1 = np.maximum(eps - eps_yield, 0.0)
    slope1 = p2 * p5

    # Hinge 2: secondary hardening at higher strain
    eps_knot2 = eps_yield + p8
    hinge2 = np.maximum(eps - eps_knot2, 0.0)
    slope2 = p2 * p6

    # Hinge 3: tertiary/saturation region
    eps_knot3 = eps_yield + 2.0 * p8
    hinge3 = np.maximum(eps - eps_knot3, 0.0)
    slope3 = p2 * p7

    # Combine all contributions
    stress = elastic + slope1 * hinge1 + slope2 * hinge2 + slope3 * hinge3 + p9

    return stress
# EVOLVE-BLOCK-END
