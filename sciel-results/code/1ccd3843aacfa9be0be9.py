"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using normalized sparse quadratic polynomial basis.
    
    Structural change: Replace mechanistic Monod/Gaussian features with a
    normalized polynomial basis. Normalizing inputs to ~[-1,1] range ensures
    BFGS starting point (all ones) is well-conditioned. The quadratic terms
    provide flexible curvature for temperature/pH optima without explicit
    Gaussian exponentials that can cause vanishing gradients.
    
    Feature selection (10 monomials from 4 normalized inputs):
    - 1 bias
    - 4 linear: bn, sn, tn, phn
    - 4 pairwise interactions: bn*sn, bn*tn, sn*tn, sn*phn
    - 1 quadratic in temp: tn^2 (captures thermal optimum curvature)
    - 1 quadratic in substrate: sn^2 (captures saturation curvature)
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Normalize inputs to roughly [-1, 1] based on training ranges:
    # b: [1, 4.998] -> center 3, scale 2
    # s: [1, 4.999] -> center 3, scale 2  (wait, s range is [1, 4.999] but also [1.002, 39.98])
    # Actually re-reading: inputs in order [1, 4.998], [1, 4.999], [1.002, 39.98], [2, 12]
    # So b: [1, 4.998], s: [1, 4.999], temp: [1.002, 39.98], pH: [2, 12]
    
    bn = (b - 3.0) / 2.0
    sn = (s - 3.0) / 2.0
    tn = (temp - 20.5) / 19.5
    phn = (pH - 7.0) / 5.0

    # Basis functions
    F0 = np.ones_like(b)       # bias
    F1 = bn                    # linear density
    F2 = sn                    # linear substrate
    F3 = tn                    # linear temp
    F4 = phn                   # linear pH
    F5 = bn * sn               # density * substrate interaction
    F6 = bn * tn               # density * temp interaction
    F7 = sn * tn               # substrate * temp interaction
    F8 = sn * phn              # substrate * pH interaction
    F9 = tn * tn               # quadratic temp (curvature for optimum)

    result = (p0 * F0 +
              p1 * F1 +
              p2 * F2 +
              p3 * F3 +
              p4 * F4 +
              p5 * F5 +
              p6 * F6 +
              p7 * F7 +
              p8 * F8 +
              p9 * F9)

    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
