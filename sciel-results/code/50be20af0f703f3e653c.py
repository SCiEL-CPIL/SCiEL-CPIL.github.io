import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5] -> shift by 1, divide by 4 => [0, 1]
    # s: [1, 5] -> shift by 1, divide by 4 => [0, 1]
    # temp: [1, 40] -> shift by 1, divide by 39 => [0, 1]
    # pH: [2, 12] -> shift by 2, divide by 10 => [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0

    p = params

    # 4th-order multivariate polynomial with 10 coefficients
    # The full 4th-order polynomial in 4 variables would have C(4+4,4) = 35 terms.
    # We use a sparse set of 10 terms that includes the most important main effects,
    # key quadratic terms, and selected interactions/cubic/quartic terms for flexibility.
    #
    # Terms selected:
    # 1. Intercept
    # 2. b^4 (quartic biomass - captures saturation/inhibition at high b)
    # 3. b^3 (cubic biomass)
    # 4. b^2 (quadratic biomass)
    # 5. b (linear biomass)
    # 6. s^2 (quadratic substrate - captures saturation)
    # 7. s (linear substrate)
    # 8. t^4 (quartic temperature - captures bell-shape)
    # 9. t^3 (cubic temperature)
    # 10. ph^2 (quadratic pH - captures bell-shape)
    #
    # This set focuses on the key non-linearities in each variable without
    # requiring many interaction terms, allowing the optimizer to capture
    # the main functional shapes.

    b2 = b_norm * b_norm
    b3 = b2 * b_norm
    b4 = b2 * b2
    s2 = s_norm * s_norm
    t2 = t_norm * t_norm
    t3 = t2 * t_norm
    t4 = t2 * t2
    ph2 = ph_norm * ph_norm

    y = (p[0] * 1.0 +
         p[1] * b4 +
         p[2] * b3 +
         p[3] * b2 +
         p[4] * b_norm +
         p[5] * s2 +
         p[6] * s_norm +
         p[7] * t4 +
         p[8] * t3 +
         p[9] * ph2)

    return y

BET: {mechanism: "Replaces multiplicative Monod/Gaussian model with a sparse 4th-order polynomial in normalized inputs, using 10 parameters to capture higher-order non-monotonic dependencies in biomass, substrate, temperature, and pH without assuming specific biological functional forms. The quartic/cubic terms for temperature and biomass allow flexible bell-shape and saturation curves.", q: 0.6, delta: [0.0, 1.0]}
