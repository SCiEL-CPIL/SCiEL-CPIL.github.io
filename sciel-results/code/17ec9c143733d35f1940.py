import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Physics-based basis for nonlinear oscillator acceleration
    # a(t, x, v) = Drive(t) + Restoring(x) + Damping(v) + Coupling(x,v)
    
    # Time-dependent drive terms
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    
    # Restoring force terms (nonlinear)
    sin_x = np.sin(x)
    x_term = x
    x_cubed = x**3
    
    # Damping terms
    v_lin = v
    v_quad = v**2
    v_cubed = v**3
    
    # Coupling terms
    x_v = x * v
    x_sin = x * np.sin(x)
    v_abs_approx = np.sign(v) * v**2  # smooth-ish odd damping
    
    # Combine with 10 params
    # p0: sin(t) drive
    # p1: cos(t) drive
    # p2: sin(x) restoring
    # p3: x linear restoring
    # p4: x^3 Duffing term
    # p5: v linear damping
    # p6: v^2 quadratic damping
    # p7: v^3 cubic damping
    # p8: x*v coupling
    # p9: x*sin(x) nonlinear coupling
    
    result = (params[0] * sin_t +
              params[1] * cos_t +
              params[2] * sin_x +
              params[3] * x_term +
              params[4] * x_cubed +
              params[5] * v_lin +
              params[6] * v_quad +
              params[7] * v_cubed +
              params[8] * x_v +
              params[9] * x_sin)
    
    return result
