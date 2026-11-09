import numpy as np
from scipy.optimize import least_squares

def find_root_system(F, x0):
    x0 = np.asarray(x0, dtype=float)
    
    # Check initial point
    Fx0 = F(x0)
    if np.linalg.norm(Fx0) <= 1e-8:
        return x0

    # Use least_squares with trust-region reflective method
    # This is a robust trust-region method for least-squares problems.
    try:
        res = least_squares(
            F,
            x0,
            method='trf',
            ftol=1e-15,
            xtol=1e-15,
            gtol=1e-15,
            max_nfev=200
        )
        x = res.x
    except Exception:
        x = x0

    # Final check
    Fx = F(x)
    if np.linalg.norm(Fx) <= 1e-8:
        return x

    # Fallback: Newton with FD Jacobian
    for _ in range(50):
        Fx = F(x)
        norm_F = np.linalg.norm(Fx)
        if norm_F <= 1e-8:
            break
        
        J = np.zeros((2, 2))
        h = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - Fx) / h
        
        try:
            dx = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(J, Fx, rcond=None)[0]
        
        alpha = 1.0
        for _ in range(20):
            x_trial = x - alpha * dx
            F_trial = F(x_trial)
            if np.linalg.norm(F_trial) < norm_F:
                x = x_trial
                break
            alpha *= 0.5
        else:
            break
    
    return x
