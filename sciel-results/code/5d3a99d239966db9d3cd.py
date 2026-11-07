import numpy as np
from scipy.integrate import odeint

def integrate(f, t_span, y0):
    t = np.linspace(t_span[0], t_span[1], 1000) # Need to specify points?
    # odeint requires t array.
    # If I use a dense t array, it will be accurate.
    # But odeint is adaptive internally.
    # I can just pass the two points? No, odeint integrates over the range of t.
    # If I pass t=[0, 5], it will take steps.
    # To ensure high accuracy, I should probably use a fine grid or let it adapt.
    # odeint has rtol and atol arguments.
    
    # To match the "end of t_span", I just take the last point.
    t = np.array([t_span[0], t_span[1]])
    sol = odeint(f, y0, t, rtol=1e-13, atol=1e-15)
    return sol[-1]
