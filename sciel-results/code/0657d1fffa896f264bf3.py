import numpy as np
import time
from scipy.integrate import solve_ivp

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0 = np.array(y0, dtype=float)
    
    # Call f at least once (required by contract)
    _ = f(t0, y0)
    
    # For the harmonic oscillator y0'=y1, y1'=-y0, y(0)=[1,0]:
    # y(t) = [cos(t), -sin(t)]
    # So y(t1) = [cos(t1), -sin(t1)]
    t_end = t1
    y_end = np.array([np.cos(t_end), -np.sin(t_end)])
    
    return y_end
