import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
        c_v = params[0]
        c_x = params[1]
        c_x3 = params[2]
        c_v2 = params[3]
        c_x2 = params[4]
        c_xv = params[5]
        c_c = params[6]
        c_s = params[7]
        omega_scaled = params[8]
        c_b = params[9]
        
        # Scale time to O(1) for stable gradients
        t_scaled = t / 40.0
        
        # Nonlinear terms
        x2 = x * x
        x3 = x2 * x
        v2 = v * v
        xv = x * v
        
        # Drive terms
        phase = omega_scaled * t_scaled
        drive = c_c * np.cos(phase) + c_s * np.sin(phase)
        
        # Combine
        result = c_v * v + c_x * x + c_x3 * x3 + c_v2 * v2 + c_x2 * x2 + c_xv * xv + drive + c_b
        
        return result
