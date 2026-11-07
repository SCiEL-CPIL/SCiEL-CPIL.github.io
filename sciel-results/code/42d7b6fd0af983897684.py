import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
        # Scale x and v to approx [-1, 1]
        # x range: [-0.3391, 0.3624] -> center 0.01165, half-range 0.35075
        # v range: [-0.7108, 0.6604] -> center -0.0252, half-range 0.6856
        
        x_center = 0.01165
        x_scale = 0.35075
        v_center = -0.0252
        v_scale = 0.6856
        
        x_s = (x - x_center) / x_scale
        v_s = (v - v_center) / v_scale
        
        # Clip to avoid numerical issues if data is slightly out of range
        x_s = np.clip(x_s, -1.5, 1.5)
        v_s = np.clip(v_s, -1.5, 1.5)
        
        p = params
        
        # Drive: p0*cos(w*t) + p1*sin(w*t)
        w = p[2]
        drive = p[0] * np.cos(w * t) + p[1] * np.sin(w * t)
        
        # Nonlinear terms in scaled x and v
        # We use up to cubic terms to capture Duffing-like behavior
        term_x1 = p[3] * x_s
        term_x3 = p[4] * x_s**3
        
        term_v1 = p[5] * v_s
        term_v3 = p[6] * v_s**3
        
        term_xv = p[7] * x_s * v_s
        term_x2 = p[8] * x_s**2
        term_v2 = p[9] * v_s**2
        
        return drive + term_x1 + term_x3 + term_v1 + term_v3 + term_xv + term_x2 + term_v2
