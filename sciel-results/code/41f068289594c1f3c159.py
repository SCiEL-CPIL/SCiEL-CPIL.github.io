import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Monod-like substrate saturation
    u_s = s / (1.0 + params[0] * s)
    
    # Temperature: quadratic polynomial (optimal growth at intermediate temp)
    t_poly = params[1] + params[2] * temp + params[3] * temp**2
    
    # pH: quadratic polynomial (optimal pH)
    pH_poly = params[4] + params[5] * pH + params[6] * pH**2
    
    # Density: linear
    y_b = params[7] * b
    
    # Interactions: substrate x temp, substrate x pH
    y_s_T = params[8] * u_s * temp
    y_s_pH = params[9] * u_s * pH
    
    # Main model: substrate-gated temperature effect + pH effect + density + interactions
    y = u_s * t_poly + pH_poly + y_b + y_s_T + y_s_pH
    
    return y
