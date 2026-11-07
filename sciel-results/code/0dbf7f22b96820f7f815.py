"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def integrate(f, t_span, y0):
    t0, t1 = t_span
    y0_arr = np.array(y0, dtype=float)
    
    # Fixed-step RK4
    # h = 0.001 -> 5000 steps for t in [0, 5]
    # 4 calls per step -> 20000 calls total
    # This should be highly accurate for smooth LV system
    h = 0.001
    n_steps = int(round((t1 - t0) / h))
    
    # Adjust h to exactly hit t1
    h = (t1 - t0) / n_steps
    
    t = t0
    y = y0_arr.copy()
    
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h / 2.0, y + h / 2.0 * k1)
        k3 = f(t + h / 2.0, y + h / 2.0 * k2)
        k4 = f(t + h, y + h * k3)
        
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += h
        
        # Safety check for non-finite values
        if not np.all(np.isfinite(y)):
            break
            
    return y
# EVOLVE-BLOCK-END
