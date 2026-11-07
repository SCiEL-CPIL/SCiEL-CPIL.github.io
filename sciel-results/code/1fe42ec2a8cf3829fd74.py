import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params: [c_v, c_x, c_tanhx, c_tanhv, c_x2v, c_v2x, c_cos1, c_sin1, omega, c_bias]
    c_v = params[0]
    c_x = params[1]
    c_tanhx = params[2]
    c_tanhv = params[3]
    c_x2v = params[4]
    c_v2x = params[5]
    c_cos1 = params[6]
    c_sin1 = params[7]
    omega = params[8]
    c_bias = params[9]

    # Damping/velocity term
    term_v = c_v * v

    # Restoring force: linear + soft-saturation via tanh
    term_x = c_x * x
    term_tanhx = c_tanhx * np.tanh(3.0 * x)  # scale factor 3.0 to match typical x range

    # Velocity-dependent nonlinearity: tanh saturation
    term_tanhv = c_tanhv * np.tanh(2.0 * v)

    # Cross terms: state-dependent damping
    term_x2v = c_x2v * (x ** 2) * v
    term_v2x = c_v2x * (v ** 2) * x

    # Driving force with fundamental frequency
    drive = c_cos1 * np.cos(omega * t) + c_sin1 * np.sin(omega * t)

    result = term_v + term_x + term_tanhx + term_tanhv + term_x2v + term_v2x + drive + c_bias

    return result
