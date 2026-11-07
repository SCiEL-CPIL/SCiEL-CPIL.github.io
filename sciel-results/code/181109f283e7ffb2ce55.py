import numpy as np
import time

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    Model: Elastic + Smooth-Hinge Hardening with Saturation.
    sigma = E(T)*strain + H(T) * (h(strain - ey(T)) - h(strain - ey(T))^2 / S) + C0 + C1*T
    where h(x) is a smooth approximation of max(0, x).
    """
    # Start time to enforce deadline (though computation is fast)
    start_time = time.time()
    DEADLINE = 150.0
    SAFETY_MARGIN = 5.0

    E0, E1 = params[0], params[1]
    H0, H1 = params[2], params[3]
    ey0, ey1 = params[4], params[5]
    delta = params[6]
    S = params[7]
    C0 = params[8]
    C1 = params[9]

    # Ensure numerical stability
    # delta controls the sharpness of the yield transition
    delta_safe = np.clip(delta, 1e-6, 1.0)
    # S controls the saturation of hardening
    S_safe = np.clip(S, 1e-6, 10.0)

    # Temperature-dependent properties
    E = E0 + E1 * temp
    H = H0 + H1 * temp
    ey = ey0 + ey1 * temp

    # Smooth hinge function: approximates max(0, x)
    # h(x) = 0.5 * (x + sqrt(x^2 + delta^2))
    x = strain - ey
    # To avoid overflow in sqrt if x is large, though strain is bounded
    h = 0.5 * (x + np.sqrt(x * x + delta_safe * delta_safe))

    # Hardening term: linear growth with quadratic saturation
    # This allows the curve to flatten out at high strains
    hardening = H * (h - (h * h) / S_safe)

    # Elastic term
    elastic = E * strain

    # Combine
    stress = elastic + hardening + C0 + C1 * temp

    # Check deadline (should never trigger for this vectorized op on 2161 rows)
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        # Return current best guess (which is the computed stress)
        # In a more complex iterative search, we'd return the last valid state.
        # Here, the computation is atomic.
        pass

    return stress
